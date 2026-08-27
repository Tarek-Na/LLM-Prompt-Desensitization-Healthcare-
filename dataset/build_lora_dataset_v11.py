"""
Builds merged_clinical_phi_v11.{train,validation,test}.jsonl on top of v10, targeting two
things found after scoring the v10-trained model:

1. v10's CoNLL-2003 real-text fix for NAME/LOCATION worked (relaxed F1 30%->98% for NAME,
   10%->93% for LOCATION), confirming the core diagnosis. But the hard-negative mining for
   GENE/CHEMICAL/SPECIES was too aggressive: those same-domain negatives (real all-O
   sentences drawn from the exact corpora used for eval) sharpened the decision boundary
   more than intended and cost real recall (GENE 30.64%->12.41%, CHEMICAL 42.76%->29.86%,
   SPECIES 70.40%->47.85%, all relaxed). This script dials that volume down.

2. The user wants PHI (NAME/DATE/LOCATION) detection to specifically avoid misclassifying
   non-PHI numbers -- phone numbers, SSNs, MRNs, IP addresses -- as PHI. v10 already
   contains 16,022 synthetic_pii records with those spans correctly labeled O (from
   remapping the 30-category file down to 9 categories), but those sentences never
   co-occur with a real DATE/NAME/LOCATION mention in the SAME sentence, so the model never
   got direct in-context signal to discriminate "this looks like a date" from "this looks
   like a date but is actually a phone number two words later." That's exactly the failure
   pattern seen in LoRa-Raw.py's own pasted output: phone-number and IP-address fragments
   ("312", "555", ".1.") getting tagged as LOCATION/NAME/DATE. This script adds new
   contrastive templates that combine a real PHI mention with an adjacent non-PHI numeric
   identifier, all reusing build_dataset.py's own generators for phone/SSN/MRN/IP/account/
   device numbers so the surface forms match exactly what a real 30-category export would
   produce (and are known to be realistic, e.g. the credit-card generator is Luhn-valid).

Does not touch build_dataset.py or benchmark_clinical_phi.jsonl.
"""
import os
import json
import random
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
VERSION = "v11"

BASE_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, "merged_clinical_phi_v10.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, "merged_clinical_phi_v10.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, "merged_clinical_phi_v10.test.jsonl"),
}
OUTPUT_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.test.jsonl"),
}
LABELS_PATH = os.path.join(REPO_ROOT, f"labels_{VERSION}.json")

# Dial the v10 same-domain hard negatives down to this fraction (they hurt GENE/CHEMICAL/
# SPECIES recall more than intended). The conll2003_real source (both its positive NAME/
# LOCATION examples and its own natural negatives) is left untouched -- that one worked.
SELF_DOMAIN_NEGATIVE_SOURCES = {"bc2gm_negative", "bc4chemd_negative", "species800_negative"}
SELF_DOMAIN_NEGATIVE_KEEP_FRACTION = float(os.environ.get("PHI_NEGATIVE_KEEP_FRACTION", "0.35"))

# How many new PHI-vs-numeric-identifier contrastive examples to generate.
N_CONTRAST_EXAMPLES = int(os.environ.get("PHI_CONTRAST_EXAMPLES", "6000"))

master_labels = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-CHEMICAL", "I-CHEMICAL",
    "B-GENE", "I-GENE",
    "B-CELL", "I-CELL",
    "B-SPECIES", "I-SPECIES",
    "B-VARIANT", "I-VARIANT",
    "B-NAME", "I-NAME",
    "B-DATE", "I-DATE",
    "B-LOCATION", "I-LOCATION",
]
master_label2id = {label: i for i, label in enumerate(master_labels)}

# =====================================================================
# Vocab + PHI-slot generators, copied verbatim from dataset/build_dataset.py so the
# surface forms match exactly what the rest of the pipeline already produces.
# =====================================================================
FIRST_NAMES = ["James", "Maria", "Robert", "Linda", "Michael", "Patricia", "David",
               "Jennifer", "William", "Elizabeth", "Ahmed", "Fatima", "Wei", "Yuki",
               "Carlos", "Sofia", "Omar", "Aisha", "Noah", "Emma", "Gregory", "Sarah",
               "Richard", "Mary", "John", "Jane", "Thomas", "Susan", "Daniel", "Karen",
               "Mark", "Laura", "Steven", "Nancy", "Kevin", "Betty", "Jason", "Sandra",
               "Anthony", "Ashley"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Martinez", "Chen", "Khan", "Nguyen", "Kim",
              "Patel", "Osei", "Ivanov", "Silva", "Muller", "Andersson", "House",
              "Scott", "Connor", "Jenkins", "Doe", "Wilson", "Anderson", "Taylor",
              "Thomas", "Moore", "Jackson", "White", "Harris", "Clark", "Lewis"]
CITIES = ["Springfield", "Boston", "Austin", "Denver", "Portland", "Nashville",
          "Cleveland", "Tampa", "Fresno", "Buffalo", "Seattle", "Chicago", "Houston",
          "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Atlanta", "Miami"]
STATES = ["IL", "MA", "TX", "CO", "OR", "TN", "OH", "FL", "CA", "NY", "WA", "GA", "PA", "AZ"]
HOSPITALS = [
    "St. Mary's Medical Center", "Mercy General Hospital", "Riverside Community Clinic",
    "Lakeside Regional Hospital", "Northside Health Center", "Sunrise Family Practice",
    "Boston Medical Center", "Johns Hopkins Hospital", "Cleveland Clinic", "Mayo Clinic",
    "UCLA Medical Center", "Mount Sinai Hospital", "Massachusetts General Hospital",
    "Stanford Health Care", "Duke University Hospital", "Princeton-Plainsboro Teaching Hospital",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
PREFIXES = ["Dr.", "Mr.", "Mrs.", "Ms."]
ID_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

def _surname_token(rng):
    if rng.random() < 0.2:
        style = rng.choice(["apostrophe", "hyphenated"])
        if style == "apostrophe":
            return "O'" + rng.choice(LAST_NAMES)
        return f"{rng.choice(LAST_NAMES)}-{rng.choice(LAST_NAMES)}"
    return rng.choice(LAST_NAMES)

def _phi_name_tokens(rng):
    parts = []
    if rng.random() < 0.3:
        parts.append(rng.choice(PREFIXES))
    parts.append(rng.choice(FIRST_NAMES))
    if rng.random() < 0.6:
        parts.append(_surname_token(rng))
    return parts

def _phi_date_tokens(rng):
    month, day, year = rng.choice(MONTHS), rng.randint(1, 28), rng.randint(1960, 2025)
    fmt = rng.choice(["slash", "iso", "month_name"])
    if fmt == "slash":
        return [f"{MONTHS.index(month) + 1:02d}/{day:02d}/{year}"]
    if fmt == "iso":
        return [f"{year}-{MONTHS.index(month) + 1:02d}-{day:02d}"]
    return [month, f"{day},", str(year)]

def _phi_loc_tokens(rng):
    kind = rng.choice(["city_state", "hospital", "address"])
    if kind == "city_state":
        return [rng.choice(CITIES) + ",", rng.choice(STATES)]
    if kind == "hospital":
        return rng.choice(HOSPITALS).split()
    return [str(rng.randint(100, 9999))] + rng.choice(STREET_NAMES).split() + [",", rng.choice(CITIES)]

STREET_NAMES = ["Oak Street", "Maple Avenue", "Elm Drive", "Sunset Boulevard", "Pine Lane"]

# =====================================================================
# Non-PHI numeric identifier generators, copied verbatim from build_dataset.py -- every
# one of these must be labeled O, never DATE/NAME/LOCATION, no matter how close it sits to
# a real PHI mention in the same sentence.
# =====================================================================
def _phone_tokens(rng):
    area, mid, last = rng.randint(200, 999), rng.randint(200, 999), rng.randint(1000, 9999)
    fmt = rng.choice(["paren", "dash", "dot", "intl"])
    if fmt == "paren":
        return [f"({area})", f"{mid}-{last}"]
    if fmt == "dash":
        return [f"{area}-{mid}-{last}"]
    if fmt == "dot":
        return [f"{area}.{mid}.{last}"]
    return [f"+1-{area}-{mid}-{last}"]

def _ssn_tokens(rng):
    return [f"{rng.randint(100, 899):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"]

def _mrn_tokens(rng):
    if rng.random() < 0.5:
        return [str(rng.randint(1000000, 9999999))]
    return [f"MRN-{rng.randint(100000, 999999)}"]

def _account_number_tokens(rng):
    return [str(rng.randint(10000000, 999999999999))]

def _device_id_tokens(rng):
    return [f"SN-{rng.randint(1000000, 9999999)}{rng.choice(ID_LETTERS)}"]

def _ip_address_tokens(rng):
    return [f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"]

def _extension_tokens(rng):
    return [str(rng.randint(1000, 9999))]

def _employee_id_tokens(rng):
    return [f"E-{rng.randint(10000, 99999)}"]

def _lab_value_tokens(rng):
    kind = rng.choice(["percent", "mg_dl", "bpm"])
    if kind == "percent":
        return [f"{rng.uniform(4.0, 12.0):.1f}%"]
    if kind == "mg_dl":
        return [str(rng.randint(60, 400)), "mg/dL"]
    return [str(rng.randint(50, 160)), "bpm"]

NUMERIC_ID_BUILDERS = [
    ("PHONE_NUM", _phone_tokens), ("SSN_NUM", _ssn_tokens), ("MRN_NUM", _mrn_tokens),
    ("ACCT_NUM", _account_number_tokens), ("DEVICE_NUM", _device_id_tokens),
    ("IP_NUM", _ip_address_tokens), ("EXT_NUM", _extension_tokens),
    ("EMP_NUM", _employee_id_tokens), ("LAB_NUM", _lab_value_tokens),
]

# =====================================================================
# Contrastive templates: a real PHI mention plus a nearby non-PHI numeric identifier in
# the SAME sentence, directly recreating the failure pattern seen in LoRa-Raw.py's own
# pasted output (phone/IP fragments tagged as LOCATION/NAME/DATE).
# =====================================================================
CONTRAST_SENTENCE_TEMPLATES = [
    ["Contacted patient's family at", "PHONE_NUM", "regarding the results from", "DATE", "."],
    ["Patient", "NAME", "( MRN :", "MRN_NUM", ") was admitted on", "DATE", "."],
    ["SSN :", "SSN_NUM", "confirmed for patient", "NAME", "on", "DATE", "."],
    ["IP address from telemedicine portal visit :", "IP_NUM", ", session recorded", "DATE", "."],
    ["Dr.", "NAME", "can be reached at extension", "EXT_NUM", "regarding the", "DATE", "visit at", "LOC", "."],
    ["Lab value :", "LAB_NUM", "recorded on", "DATE", "for patient", "NAME", "at", "LOC", "."],
    ["Employee ID", "EMP_NUM", "for", "NAME", ", next visit scheduled", "DATE", "."],
    ["Billing account", "ACCT_NUM", "for", "NAME", "was opened on", "DATE", "."],
    ["Follow-up call to", "NAME", "at", "PHONE_NUM", "scheduled for", "DATE", "."],
    ["Device serial", "DEVICE_NUM", "was implanted in", "NAME", "on", "DATE", "at", "LOC", "."],
    ["Please verify MRN", "MRN_NUM", "matches", "NAME", "before the", "DATE", "procedure."],
    ["Patient reached at", "PHONE_NUM", "on", "DATE", "; resides near", "LOC", "."],
    [ "NAME", "was discharged on", "DATE", "from", "LOC", "; SSN on file :", "SSN_NUM", "."],
    ["Vitals for", "NAME", ":", "LAB_NUM", ", recorded", "DATE", "."],
]

def _build_contrast_example(rng):
    tokens, labels = [], []
    template = rng.choice(CONTRAST_SENTENCE_TEMPLATES)
    for slot in template:
        if slot == "NAME":
            slot_tokens = _phi_name_tokens(rng)
            slot_labels = ["B-NAME"] + ["I-NAME"] * (len(slot_tokens) - 1)
        elif slot == "DATE":
            slot_tokens = _phi_date_tokens(rng)
            slot_labels = ["B-DATE"] + ["I-DATE"] * (len(slot_tokens) - 1)
        elif slot == "LOC":
            slot_tokens = _phi_loc_tokens(rng)
            slot_labels = ["B-LOCATION"] + ["I-LOCATION"] * (len(slot_tokens) - 1)
        elif slot in dict(NUMERIC_ID_BUILDERS):
            slot_tokens = dict(NUMERIC_ID_BUILDERS)[slot](rng)
            slot_labels = ["O"] * len(slot_tokens)
        else:
            slot_tokens = slot.split()
            slot_labels = ["O"] * len(slot_tokens)
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[l] for l in labels],
            "source": "synthetic_phi_numeric_contrast"}

def generate_contrast_examples(n, rng):
    return [_build_contrast_example(rng) for _ in range(n)]

# =====================================================================
# Deterministic hash-of-tokens split, same convention as the rest of the pipeline.
# =====================================================================
def split_for(tokens):
    digest = hashlib.md5(" ".join(tokens).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"

def main():
    rng = random.Random(43)
    print(f"Building merged_clinical_phi_v11 on top of v10 -- dialing down self-domain "
          f"negatives to {SELF_DOMAIN_NEGATIVE_KEEP_FRACTION:.0%} and adding "
          f"{N_CONTRAST_EXAMPLES} PHI-vs-numeric-identifier contrastive examples.\n")

    base_by_split = {}
    for split_name, path in BASE_SPLIT_PATHS.items():
        kept, dropped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("source") in SELF_DOMAIN_NEGATIVE_SOURCES:
                    if rng.random() >= SELF_DOMAIN_NEGATIVE_KEEP_FRACTION:
                        dropped += 1
                        continue
                kept.append(rec)
        base_by_split[split_name] = kept
        print(f"{split_name:12s}: loaded {len(kept)} records from v10 "
              f"(dropped {dropped} self-domain negatives to reduce over-suppression).")

    print()
    contrast_examples = generate_contrast_examples(N_CONTRAST_EXAMPLES, rng)
    by_split = {"train": [], "validation": [], "test": []}
    for ex in contrast_examples:
        by_split[split_for(ex["tokens"])].append(ex)
    print(f"Generated {len(contrast_examples)} contrastive examples: "
          f"{len(by_split['train'])} train / {len(by_split['validation'])} validation / "
          f"{len(by_split['test'])} test.")

    print("\nWriting output files...")
    total_by_split = {}
    for split_name in ("train", "validation", "test"):
        merged = base_by_split[split_name] + by_split[split_name]
        rng.shuffle(merged)
        total_by_split[split_name] = merged
        with open(OUTPUT_SPLIT_PATHS[split_name], "w", encoding="utf-8") as f:
            for ex in merged:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name:12s}: {len(merged):6d} records -> {os.path.basename(OUTPUT_SPLIT_PATHS[split_name])}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "labels": master_labels,
            "label2id": master_label2id,
            "id2label": {str(i): l for i, l in enumerate(master_labels)},
        }, f, indent=2)
    print(f"\nLabel schema (unchanged, 9-category) written to {os.path.basename(LABELS_PATH)}")

    print("\nEntity spans per label, train split (B- tag counts):")
    entity_counts = {}
    for ex in total_by_split["train"]:
        for t in ex["ner_tags"]:
            label = master_labels[t]
            if label.startswith("B-"):
                entity_counts[label[2:]] = entity_counts.get(label[2:], 0) + 1
    for ent_type, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {ent_type:10s}: {cnt}")

    all_o_count = sum(1 for ex in total_by_split["train"] if all(t == 0 for t in ex["ner_tags"]))
    print(f"\nFully-negative (all-O) records in train split: {all_o_count} "
          f"({100*all_o_count/len(total_by_split['train']):.1f}% of train)")

if __name__ == "__main__":
    main()
