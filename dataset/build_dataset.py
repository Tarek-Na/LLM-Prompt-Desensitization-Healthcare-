# =====================================================================
# Merged clinical NER dataset builder
#
# Deliberately dependency-free: only the Python standard library is used
# (urllib, json, random). No `datasets` / `huggingface_hub` / `pyarrow`
# install is required -- `python build_dataset.py` just works, in VS Code
# or anywhere else with a plain Python 3 interpreter and network access.
# Source data is fetched directly over HTTP as JSON (raw dataset files
# where available, the HF datasets-server rows API otherwise), one local
# file (i2b2.jsonl) is read from disk, and two categories are filled in
# with synthetic generation where no viable open real-data source exists.
# Everything is merged, quality-filtered, balanced, and split, then
# written out as train/validation/test JSONL files.
# =====================================================================

import os
import sys
import json
import random
import re
import time
import hashlib
import urllib.request
import urllib.parse

MAX_EXAMPLES = int(os.environ.get("PHI_MAX_EXAMPLES", "0") or 0)
REQUEST_TIMEOUT = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATHS = {
    "train": os.path.join(BASE_DIR, "merged_clinical_phi.train.jsonl"),
    "validation": os.path.join(BASE_DIR, "merged_clinical_phi.validation.jsonl"),
    "test": os.path.join(BASE_DIR, "merged_clinical_phi.test.jsonl"),
}

# =====================================================================
# Master label schema
# NAME/DATE/LOCATION (not PHI_NAME/PHI_DATE/PHI_LOC) per instruction --
# simpler, generic category names.
# =====================================================================
# Schema v2: expanded from 9 to 29 entity types so this dataset can serve as a common
# label space when benchmarking against SOTA PII/PHI de-identification models. Previously
# only NAME/DATE/LOCATION covered PHI -- but SOTA de-id tools (i2b2-trained clinical
# de-identifiers, and general-purpose PII scrubbers like Presidio/AWS Comprehend PII/Google
# DLP) each detect a different subset of identifier types (phone, SSN, MRN, IP address,
# credit card, ...) that we previously had no label for at all -- meaning even a CORRECT
# detection from a SOTA model couldn't be scored against our data, since there was no
# ground-truth category to compare it to. This is a schema-breaking change: any adapter
# trained against the old 19-label schema is incompatible with this one.
#
#   - Clinical entities: unchanged (DISEASE/CHEMICAL/GENE/CELL/SPECIES/VARIANT).
#   - Core PHI: unchanged (NAME/DATE/LOCATION).
#   - HIPAA Safe Harbor identifiers not previously covered: AGE, PHONE, FAX, EMAIL, SSN,
#     MEDICAL_RECORD_NUMBER, HEALTH_PLAN_ID, ACCOUNT_NUMBER, LICENSE_NUMBER, VEHICLE_ID,
#     DEVICE_ID, URL, IP_ADDRESS, BIOMETRIC_ID, OTHER_ID. (HIPAA identifier #17, full-face
#     photographs, is image-only and has no text span, so it's excluded here.)
#   - i2b2 2014 schema additions beyond strict HIPAA (i2b2-trained de-id models output
#     these too): PROFESSION, ORGANIZATION.
#   - General-PII-only categories, not in HIPAA but standard in Presidio-style tools:
#     CREDIT_CARD, USERNAME, PASSPORT_NUMBER, CRYPTO_WALLET.
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
    "B-AGE", "I-AGE",
    "B-PHONE", "I-PHONE",
    "B-FAX", "I-FAX",
    "B-EMAIL", "I-EMAIL",
    "B-SSN", "I-SSN",
    "B-MEDICAL_RECORD_NUMBER", "I-MEDICAL_RECORD_NUMBER",
    "B-HEALTH_PLAN_ID", "I-HEALTH_PLAN_ID",
    "B-ACCOUNT_NUMBER", "I-ACCOUNT_NUMBER",
    "B-LICENSE_NUMBER", "I-LICENSE_NUMBER",
    "B-VEHICLE_ID", "I-VEHICLE_ID",
    "B-DEVICE_ID", "I-DEVICE_ID",
    "B-URL", "I-URL",
    "B-IP_ADDRESS", "I-IP_ADDRESS",
    "B-BIOMETRIC_ID", "I-BIOMETRIC_ID",
    "B-OTHER_ID", "I-OTHER_ID",
    "B-PROFESSION", "I-PROFESSION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-USERNAME", "I-USERNAME",
    "B-PASSPORT_NUMBER", "I-PASSPORT_NUMBER",
    "B-CRYPTO_WALLET", "I-CRYPTO_WALLET",
]
master_label2id = {label: i for i, label in enumerate(master_labels)}
ENTITY_TYPES = [
    "DISEASE", "CHEMICAL", "GENE", "CELL", "SPECIES", "VARIANT", "NAME", "DATE", "LOCATION",
    "AGE", "PHONE", "FAX", "EMAIL", "SSN", "MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_ID",
    "ACCOUNT_NUMBER", "LICENSE_NUMBER", "VEHICLE_ID", "DEVICE_ID", "URL", "IP_ADDRESS",
    "BIOMETRIC_ID", "OTHER_ID", "PROFESSION", "ORGANIZATION", "CREDIT_CARD", "USERNAME",
    "PASSPORT_NUMBER", "CRYPTO_WALLET",
]

# =====================================================================
# HTTP helpers (stdlib only, explicit timeouts + retries so a flaky
# connection fails fast instead of hanging indefinitely)
# =====================================================================
def _fetch_bytes(url, retries=5, backoff=2):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phi-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = backoff * 5 * (attempt + 1) if "429" in str(e) else backoff * (attempt + 1)
                time.sleep(wait)
    raise last_err

def fetch_json(url, **kw):
    return json.loads(_fetch_bytes(url, **kw))

def fetch_text(url, **kw):
    return _fetch_bytes(url, **kw).decode("utf-8")

# =====================================================================
# Entity-type normalization (shared across all sources)
# =====================================================================
ENTITY_TYPE_ALIASES = {
    "DISEASE": "DISEASE",
    "DISEASEORPHENOTYPICFEATURE": "DISEASE",
    "CHEMICAL": "CHEMICAL",
    "CHEMICALENTITY": "CHEMICAL",
    "DRUG": "CHEMICAL",
    "MEDICATION": "CHEMICAL",
    "GENE": "GENE",
    "PROTEIN": "GENE",
    "GENEORGENEPRODUCT": "GENE",
    "CELL": "CELL",
    "CELLTYPE": "CELL",
    "CELLLINE": "CELL",
    "SPECIES": "SPECIES",
    "ORGANISMTAXON": "SPECIES",
    "VARIANT": "VARIANT",
    "SEQUENCEVARIANT": "VARIANT",
}

def map_str_to_master(tag_str):
    tag_str = tag_str.upper()
    if tag_str == "O":
        return master_label2id["O"]
    prefix, body = (tag_str[:2], tag_str[2:]) if tag_str[:2] in ("B-", "I-") else ("B-", tag_str)
    body = body.replace("_", "").replace("-", "")
    canonical = ENTITY_TYPE_ALIASES.get(body)
    if canonical is None:
        return master_label2id["O"]
    return master_label2id.get(f"{prefix}{canonical}", 0)

# =====================================================================
# Source 1 & 2: tner/bc5cdr and tner/bionlp2004 -- both host raw
# JSON-lines files directly on the hub (dataset/{split}.json) plus a
# dataset/label.json giving the exact int -> label-string scheme, so no
# guessing is needed for either.
# =====================================================================
def fetch_tner_dataset(repo_id, source_name, splits=("train", "valid", "test")):
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main/dataset"
    try:
        label_map = fetch_json(f"{base}/label.json")
    except Exception as e:
        print(f"Warning: could not fetch label map for {repo_id}: {e}")
        return []
    id_to_label = {v: k for k, v in label_map.items()}

    examples = []
    for split in splits:
        try:
            text = fetch_text(f"{base}/{split}.json")
        except Exception as e:
            print(f"Warning: could not fetch {repo_id} split {split}: {e}")
            continue
        split_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        print(f"Loaded {repo_id} split {split}: {len(split_rows)} rows")
        for row in split_rows:
            tags = [map_str_to_master(id_to_label.get(t, "O")) for t in row.get("tags", [])]
            examples.append({"tokens": row.get("tokens", []), "ner_tags": tags, "source": source_name})
            if MAX_EXAMPLES > 0 and len(examples) >= MAX_EXAMPLES:
                return examples
    return examples

# =====================================================================
# Source 3 & 4: bigbio/blurb (ncbi_disease config) and disi-unibo-nlp/biored
# -- fetched page-by-page via the HF datasets-server rows API (plain JSON,
# no parquet/arrow decoding needed).
# =====================================================================
def get_split_row_count(dataset, config, split):
    url = (
        "https://datasets-server.huggingface.co/size"
        f"?dataset={urllib.parse.quote(dataset, safe='')}&config={urllib.parse.quote(config, safe='')}"
    )
    data = fetch_json(url)
    for s in data.get("size", {}).get("splits", []):
        if s["split"] == split:
            return s["num_rows"]
    return 0

def fetch_rows_via_datasets_server(dataset, config, split, page_size=100):
    try:
        total_rows = get_split_row_count(dataset, config, split)
    except Exception as e:
        print(f"Warning: could not get row count for {dataset}/{config}: {e}")
        return []
    if total_rows == 0:
        print(f"Warning: {dataset}/{config}/{split} reports 0 rows.")
        return []

    limit = total_rows if MAX_EXAMPLES <= 0 else min(total_rows, MAX_EXAMPLES)
    rows, offset = [], 0
    while offset < limit:
        length = min(page_size, limit - offset)
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(dataset, safe='')}&config={urllib.parse.quote(config, safe='')}"
            f"&split={split}&offset={offset}&length={length}"
        )
        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"Warning: failed fetching {dataset}/{config} rows at offset {offset}: {e}")
            break
        page_rows = [r["row"] for r in data.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        time.sleep(0.3)  # stay under the datasets-server rate limit across ~50+ pages
    print(f"Loaded {dataset}/{config}/{split}: {len(rows)} rows")
    return rows

# ncbi_disease's ClassLabel names are the bare ["O", "B", "I"] -- the entity
# type isn't encoded in the tag at all, it's implicit (disease-only dataset).
NCBI_ID_TO_BIO = {0: "O", 1: "B", 2: "I"}

def align_ncbi_row(row):
    tags = []
    for t in row.get("ner_tags", []):
        bio = NCBI_ID_TO_BIO.get(t, "O")
        tags.append(master_label2id["O"] if bio == "O" else master_label2id[f"{bio}-DISEASE"])
    return {"tokens": row.get("tokens", []), "ner_tags": tags, "source": "ncbi_disease"}

def align_biored_row(row):
    tags = [map_str_to_master(t) for t in row.get("ner_tags", [])]
    return {"tokens": row.get("tokens", []), "ner_tags": tags, "source": "biored"}

# =====================================================================
# Source 5 & 6: bigbio/tmvar_v2 (genetic variant mentions) and bigbio/linnaeus
# (species mentions) -- both address the two most underrepresented categories
# (VARIANT, SPECIES). Both ship in BigBio's char-offset "kb" schema (raw text +
# entity character spans) rather than pre-tokenized BIO, so they need their own
# offset-based tokenizer/aligner, unlike the pre-tokenized sources above.
# =====================================================================
TOKEN_RE = re.compile(r"\w+(?:[-'.]\w+)*|[^\w\s]")

def _tokenize_with_offsets(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

# Abbreviation-aware sentence splitter, used for full-document/long-passage
# sources (linnaeus, tmvar_v2, i2b2) that need chunking down to sentence-sized
# examples for BERT (a raw paragraph/article can run past BERT's 512-subword
# limit and silently truncate, potentially cutting through a labeled entity).
# A naive split on every "." would re-fragment on decimal numbers and
# abbreviations exactly like the bug already fixed for BC5CDR's own splitter.
_SENT_BOUNDARY_RE = re.compile(r"[.!?]\s+(?=[A-Z0-9])")
_SENT_ABBREVIATIONS = {
    "fig", "figs", "st", "dr", "mr", "mrs", "ms", "vs", "eg", "ie", "etc", "al",
    "no", "vol", "cf", "approx", "sp", "spp", "var", "cv", "ca", "resp", "prof",
    "inc", "co", "corp", "jr", "sr", "ed", "eds", "pp", "viz", "e.g", "i.e",
}

def _split_sentences_with_offsets(text):
    """Returns [(start, end), ...] char-offset spans of text, split on sentence-ending
    punctuation, but skipping splits after a decimal number, a single initial, or a known
    abbreviation."""
    spans = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        split_at = m.start() + 1  # keep the sentence-ending punctuation with the sentence
        word_match = re.search(r"(\S+)$", text[start:m.start() + 1])
        prev_word = word_match.group(1).rstrip(".!?") if word_match else ""
        if not prev_word:
            continue
        if prev_word.replace(".", "").isdigit():
            continue  # decimal number, e.g. "12.5"
        if prev_word.lower() in _SENT_ABBREVIATIONS or (len(prev_word) <= 2 and prev_word.isalpha()):
            continue  # abbreviation or a single initial like "E."
        spans.append((start, split_at))
        start = split_at
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]

def _align_span_entities(text, entities, target_type):
    """text: str: passage/chunk text (entity offsets are relative to this same text).
    entities: list of {"offsets": [[start, end]], ...} char-offset entities.
    target_type: single master entity type name (e.g. "VARIANT", "SPECIES") that every
    entity in this source maps to.
    """
    tokens_with_offsets = _tokenize_with_offsets(text)
    tokens = [t for t, _, _ in tokens_with_offsets]
    tags = [master_label2id["O"]] * len(tokens)
    for ent in entities:
        for est, eend in ent.get("offsets", []):
            first = True
            for i, (_, ts, te) in enumerate(tokens_with_offsets):
                if max(ts, est) < min(te, eend):
                    label = f"{'B' if first else 'I'}-{target_type}"
                    tags[i] = master_label2id[label]
                    first = False
    return tokens, tags

def _passage_to_sentence_examples(text, p_start, entities, target_type, source_name):
    """Splits one passage's text into sentence-sized chunks (first on real newlines --
    e.g. linnaeus's section headers/paragraphs each sit on their own line -- then each
    line is further split on sentence boundaries), maps entities onto each chunk by char
    offset, and yields one example per chunk. Keeps every example roughly sentence-sized
    so nothing silently truncates past BERT's subword limit later."""
    examples = []
    line_start = 0
    for line in text.split("\n"):
        for rel_start, rel_end in _split_sentences_with_offsets(line):
            chunk_start = line_start + rel_start
            chunk_end = line_start + rel_end
            local_entities = []
            for ent in entities:
                offs = ent.get("offsets", [])
                if offs:
                    gs, ge = offs[0]
                    ls, le = gs - p_start, ge - p_start
                    if chunk_start <= ls and le <= chunk_end:
                        local_entities.append({"offsets": [[ls - chunk_start, le - chunk_start]]})
            tokens, tags = _align_span_entities(text[chunk_start:chunk_end], local_entities, target_type)
            # Drop section-header/filler fragments (no entity, very short).
            if any(t != master_label2id["O"] for t in tags) or len(tokens) >= 5:
                examples.append({"tokens": tokens, "ner_tags": tags, "source": source_name})
        line_start += len(line) + 1  # +1 for the "\n" consumed by split()
    return examples

def fetch_tmvar_v2_examples():
    rows = fetch_rows_via_datasets_server("bigbio/tmvar_v2", "tmvar_v2_bigbio_kb", "train")
    examples = []
    for row in rows:
        entities = row.get("entities", [])
        for passage in row.get("passages", []):
            text = passage.get("text", [""])[0]
            p_start = passage.get("offsets", [[0, len(text)]])[0][0]
            examples.extend(_passage_to_sentence_examples(text, p_start, entities, "VARIANT", "tmvar_v2"))
            if MAX_EXAMPLES > 0 and len(examples) >= MAX_EXAMPLES:
                return examples[:MAX_EXAMPLES]
    return examples

def fetch_linnaeus_examples():
    rows = fetch_rows_via_datasets_server("bigbio/linnaeus", "linnaeus_bigbio_kb", "train")
    examples = []
    for row in rows:
        entities = row.get("entities", [])
        for passage in row.get("passages", []):
            text = passage.get("text", [""])[0]
            p_start = passage.get("offsets", [[0, len(text)]])[0][0]
            examples.extend(_passage_to_sentence_examples(text, p_start, entities, "SPECIES", "linnaeus"))
            if MAX_EXAMPLES > 0 and len(examples) >= MAX_EXAMPLES:
                return examples[:MAX_EXAMPLES]
    return examples

# =====================================================================
# Shared synthetic-value building blocks (used both by the standalone
# NAME/DATE/LOCATION generator and by i2b2's de-id bracket substitution).
# =====================================================================
PREFIXES = ["Dr.", "Mr.", "Mrs.", "Ms."]
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
STREET_NAMES = ["Oak Street", "Maple Avenue", "Elm Drive", "Sunset Boulevard", "Pine Lane"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

def _surname_token(rng):
    # Plain last names dominate LAST_NAMES; real-world surnames are frequently
    # apostrophe'd (O'Connor) or hyphenated compounds of two names (O'Connor-MacLeod).
    # Neither pattern existed anywhere in training, so the model had never seen a NAME
    # span continue across an apostrophe or an internal hyphen -- confirmed on real test
    # text where "O'Connor-MacLeod" broke into three disconnected spans, one of them a
    # lone apostrophe mistagged as LOCATION. This composes that pattern ~20% of the time.
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

# =====================================================================
# Source 7: synthetic clinical-note-style NAME/DATE/LOCATION examples.
# No viable open, freely-downloadable real de-identification corpus exists
# (i2b2 2014 / n2c2 require a signed data-use agreement); without ANY
# positive examples the model has zero chance of ever predicting these
# three categories, so they're synthesized directly with exact known spans.
# =====================================================================
PHI_SLOT_BUILDERS = {"NAME": _phi_name_tokens, "DATE": _phi_date_tokens, "LOC": _phi_loc_tokens}
PHI_SLOT_LABELS = {"NAME": "NAME", "DATE": "DATE", "LOC": "LOCATION"}

PHI_SENTENCE_TEMPLATES = [
    ["Patient", "NAME", "was admitted on", "DATE", "presenting with acute chest pain ."],
    ["Please schedule a follow-up for", "NAME", "on", "DATE", "at", "LOC", "."],
    ["NAME", "referred the patient to", "LOC", "for further evaluation ."],
    ["The patient , a resident of", "LOC", ", was born on", "DATE", "."],
    ["Discharge summary for", "NAME", ": follow up scheduled at", "LOC", "on", "DATE", "."],
    ["History and physical exam completed by", "NAME", "on", "DATE", "."],
    ["Contact next of kin", "NAME", "regarding the admission on", "DATE", "."],
    ["Patient transferred to", "LOC", "under the care of", "NAME", "."],
    ["Lab results were reviewed by", "NAME", "and forwarded to", "LOC", "."],
    ["Surgery is scheduled for", "DATE", "at", "LOC", "with attending physician", "NAME", "."],
    ["Progress note dictated by", "NAME", "at", "LOC", "on", "DATE", "."],
    ["The consult was requested by", "NAME", "for a visit on", "DATE", "."],
]

def _build_synthetic_phi_example(rng):
    tokens, labels = [], []
    for slot in rng.choice(PHI_SENTENCE_TEMPLATES):
        if slot in PHI_SLOT_BUILDERS:
            slot_tokens = PHI_SLOT_BUILDERS[slot](rng)
            slot_labels = [f"B-{PHI_SLOT_LABELS[slot]}"] + [f"I-{PHI_SLOT_LABELS[slot]}"] * (len(slot_tokens) - 1)
        else:
            slot_tokens = slot.split()
            slot_labels = ["O"] * len(slot_tokens)
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[lab] for lab in labels], "source": "synthetic_phi"}

def generate_synthetic_phi_examples(n):
    if n <= 0:
        return []
    print(f"Generating {n} synthetic clinical NAME/DATE/LOCATION examples...")
    rng = random.Random(42)
    return [_build_synthetic_phi_example(rng) for _ in range(n)]

# =====================================================================
# Source 7b: synthetic comprehensive-PII examples (schema v2).
#
# NAME/DATE/LOCATION covered only 3 of the categories a real de-identification tool needs
# to handle. Every other HIPAA Safe Harbor identifier -- phone, fax, email, SSN, medical
# record number, health plan ID, account number, license number, vehicle ID, device ID,
# URL, IP address, biometric ID, other unique ID -- plus i2b2's PROFESSION/ORGANIZATION
# categories and general-PII-only categories (credit card, username, passport, crypto
# wallet) had no training exposure and no label at all, so even a SOTA model correctly
# detecting a phone number couldn't be scored against this dataset. Formats are
# deliberately varied per category (multiple real-world punctuation/grouping conventions)
# since recognizing a category across format variation -- not one fixed pattern -- is the
# actual task.
# =====================================================================
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "protonmail.com",
                  "stmarys-health.org", "mercygeneral.org", "clevelandclinic.org"]
PROFESSIONS = ["teacher", "engineer", "nurse", "accountant", "electrician", "chef", "lawyer",
               "mechanic", "pilot", "farmer", "artist", "plumber", "consultant", "architect",
               "software developer", "police officer", "firefighter", "salesperson",
               "construction worker", "journalist"]
ORG_WORDS_1 = ["Apex", "Meridian", "Horizon", "Summit", "Blackstone", "Sunrise", "Ironwood",
               "Vanguard", "Crestview", "Pinnacle", "Cascade", "Redwood", "Granite", "Silverline"]
ORG_WORDS_2 = ["Logistics", "Technologies", "Manufacturing", "Consulting", "Freight",
               "Industries", "Solutions", "Enterprises", "Holdings", "Systems", "Partners"]
ID_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # excludes I/O to avoid digit confusion, matches real ID conventions

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

def _email_tokens(rng):
    sep = rng.choice(["", ".", "_"])
    user = f"{rng.choice(FIRST_NAMES).lower()}{sep}{rng.choice(LAST_NAMES).lower()}{rng.randint(1, 99)}"
    return [f"{user}@{rng.choice(EMAIL_DOMAINS)}"]

def _ssn_tokens(rng):
    return [f"{rng.randint(100, 899):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"]

def _mrn_tokens(rng):
    if rng.random() < 0.5:
        return [str(rng.randint(1000000, 9999999))]
    return [f"MRN-{rng.randint(100000, 999999)}"]

def _health_plan_tokens(rng):
    letters = "".join(rng.choice(ID_LETTERS) for _ in range(2))
    return [f"{letters}{rng.randint(100000000, 999999999)}"]

def _account_number_tokens(rng):
    return [str(rng.randint(10000000, 999999999999))]

def _license_number_tokens(rng):
    if rng.random() < 0.5:
        return [f"{rng.choice(ID_LETTERS)}{rng.randint(1000000, 9999999)}"]
    return [f"MD-{rng.randint(10000, 99999)}"]

def _vehicle_id_tokens(rng):
    if rng.random() < 0.5:
        chars = ID_LETTERS + "0123456789"
        return ["".join(rng.choice(chars) for _ in range(17))]
    plate_letters = "".join(rng.choice(ID_LETTERS) for _ in range(3))
    return [f"{plate_letters}-{rng.randint(1000, 9999)}"]

def _device_id_tokens(rng):
    return [f"SN-{rng.randint(1000000, 9999999)}{rng.choice(ID_LETTERS)}"]

def _url_tokens(rng):
    word = rng.choice(["patientportal", "myhealth", "telehealth", "clinicresults", "medrecords"])
    tld = rng.choice(["com", "org", "net"])
    return [f"https://www.{word}.{tld}"]

def _ip_address_tokens(rng):
    return [f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"]

def _biometric_id_tokens(rng):
    kind = rng.choice(["fingerprint", "retina", "voiceprint"])
    prefix = {"fingerprint": "FP", "retina": "RS", "voiceprint": "VP"}[kind]
    return [kind, "ID", f"{prefix}-{rng.randint(1000, 9999)}"]

def _other_id_tokens(rng):
    return [f"ID-{rng.randint(10000, 99999)}"]

def _profession_tokens(rng):
    return rng.choice(PROFESSIONS).split()

def _organization_tokens(rng):
    return [rng.choice(ORG_WORDS_1), rng.choice(ORG_WORDS_2)]

def _credit_card_tokens(rng):
    # Luhn-valid, not just random digits -- confirmed directly that Presidio's credit card
    # recognizer validates the Luhn checksum, so a plain random-16-digit generator was
    # silently unscoreable: Presidio correctly rejects non-Luhn-valid numbers as malformed,
    # which showed up as a suspicious 5.17% recall that had nothing to do with Presidio's
    # actual detection capability. Visa (4) / Mastercard (51-55) BIN prefixes for realism.
    prefix = rng.choice(["4", "51", "52", "53", "54", "55"])
    payload = [int(c) for c in prefix]
    while len(payload) < 15:
        payload.append(rng.randint(0, 9))
    total = 0
    for i, d in enumerate(reversed(payload)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check_digit = (10 - (total % 10)) % 10
    number = "".join(str(d) for d in payload + [check_digit])
    groups = [number[i:i + 4] for i in range(0, 16, 4)]
    sep = rng.choice(["-", " "])
    return [sep.join(groups)]

def _username_tokens(rng):
    first, last = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    style = rng.choice(["initiallast", "firstdot", "firstunderscore"])
    if style == "initiallast":
        return [f"{first[0]}{last}{rng.randint(1, 99)}"]
    if style == "firstdot":
        return [f"{first}.{last}{rng.randint(1, 99)}"]
    return [f"{first}_{last}_staff"]

def _passport_number_tokens(rng):
    prefix = rng.choice(ID_LETTERS) if rng.random() < 0.5 else "".join(rng.choice(ID_LETTERS) for _ in range(2))
    return [f"{prefix}{rng.randint(1000000, 9999999)}"]

def _crypto_wallet_tokens(rng):
    hexchars = "0123456789abcdef"
    return ["0x" + "".join(rng.choice(hexchars) for _ in range(40))]

def _age_tokens(rng):
    return [str(rng.randint(1, 99))]

def _age_compound_tokens(rng):
    # The "NN-year-old" adjective form is tokenized as a single whitespace-delimited word
    # (matching how it already appeared, untagged, in existing clinical-scenario templates),
    # so it's tagged as one B-AGE token rather than split across "NN" + "-year-old".
    return [f"{rng.randint(1, 99)}-year-old"]

PII_SLOT_BUILDERS = {
    "NAME": _phi_name_tokens, "DATE": _phi_date_tokens, "LOC": _phi_loc_tokens,
    "AGE": _age_tokens, "AGECOMPOUND": _age_compound_tokens,
    "PHONE": _phone_tokens, "FAX": _phone_tokens, "EMAIL": _email_tokens, "SSN": _ssn_tokens,
    "MRN": _mrn_tokens, "HEALTHPLAN": _health_plan_tokens, "ACCOUNT": _account_number_tokens,
    "LICENSE": _license_number_tokens, "VEHICLE": _vehicle_id_tokens, "DEVICE": _device_id_tokens,
    "URL": _url_tokens, "IP": _ip_address_tokens, "BIOMETRIC": _biometric_id_tokens,
    "OTHERID": _other_id_tokens, "PROFESSION": _profession_tokens, "ORG": _organization_tokens,
    "CREDITCARD": _credit_card_tokens, "USERNAME": _username_tokens,
    "PASSPORT": _passport_number_tokens, "CRYPTO": _crypto_wallet_tokens,
}
PII_SLOT_LABELS = {
    "NAME": "NAME", "DATE": "DATE", "LOC": "LOCATION", "AGE": "AGE", "AGECOMPOUND": "AGE",
    "PHONE": "PHONE", "FAX": "FAX", "EMAIL": "EMAIL", "SSN": "SSN",
    "MRN": "MEDICAL_RECORD_NUMBER", "HEALTHPLAN": "HEALTH_PLAN_ID",
    "ACCOUNT": "ACCOUNT_NUMBER", "LICENSE": "LICENSE_NUMBER", "VEHICLE": "VEHICLE_ID",
    "DEVICE": "DEVICE_ID", "URL": "URL", "IP": "IP_ADDRESS", "BIOMETRIC": "BIOMETRIC_ID",
    "OTHERID": "OTHER_ID", "PROFESSION": "PROFESSION", "ORG": "ORGANIZATION",
    "CREDITCARD": "CREDIT_CARD", "USERNAME": "USERNAME", "PASSPORT": "PASSPORT_NUMBER",
    "CRYPTO": "CRYPTO_WALLET",
}
PII_SENTENCE_TEMPLATES = [
    ["Patient", ":", "NAME", ",", "AGE", "years", "old", ".", "Contact", ":", "PHONE", ",",
     "Email", ":", "EMAIL", "."],
    ["Fax", "referrals", "to", "FAX", ".", "Social", "Security", "Number", "on", "file", ":",
     "SSN", "."],
    ["Medical", "Record", "Number", ":", "MRN", ".", "Health", "Plan", "ID", ":",
     "HEALTHPLAN", "."],
    ["Billing", "account", "number", ":", "ACCOUNT", ".", "Please", "contact", "billing",
     "with", "questions", "."],
    ["NAME", "holds", "professional", "license", "LICENSE", ".", "Employer", ":", "ORG", ",",
     "occupation", ":", "PROFESSION", "."],
    ["Vehicle", "on", "file", ":", "VEHICLE", ".", "Home", "monitoring", "device", "serial",
     "number", ":", "DEVICE", "."],
    ["Patient", "portal", "login", ":", "URL", ".", "Last", "recorded", "network", "address",
     ":", "IP", "."],
    ["Biometric", "identifier", "recorded", ":", "BIOMETRIC", "."],
    ["Reference", "number", ":", "OTHERID", "for", "future", "correspondence", "."],
    ["Payment", "was", "processed", "using", "card", "CREDITCARD", ".", "Portal", "username",
     ":", "USERNAME", "."],
    ["Passport", "number", "on", "file", ":", "PASSPORT", ".", "Cryptocurrency", "refund",
     "address", ":", "CRYPTO", "."],
    ["NAME", ",", "a", "AGECOMPOUND", "PROFESSION", ",", "was", "seen", "at", "LOC", "on",
     "DATE", "."],
    ["Please", "reach", "NAME", "at", "PHONE", "or", "EMAIL", "regarding", "the",
     "appointment", "on", "DATE", "."],
    ["Insurance", "verification", ":", "NAME", ",", "plan", "ID", "HEALTHPLAN", ",",
     "account", "ACCOUNT", "."],
    ["Employee", "of", "ORG", ",", "NAME", "was", "referred", "by", "employee", "ID",
     "OTHERID", "."],
]

def _build_pii_example(rng):
    tokens, labels = [], []
    for slot in rng.choice(PII_SENTENCE_TEMPLATES):
        if slot in PII_SLOT_BUILDERS:
            slot_tokens = PII_SLOT_BUILDERS[slot](rng)
            slot_labels = [f"B-{PII_SLOT_LABELS[slot]}"] + [f"I-{PII_SLOT_LABELS[slot]}"] * (len(slot_tokens) - 1)
        else:
            slot_tokens = [slot]
            slot_labels = ["O"]
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[lab] for lab in labels], "source": "synthetic_pii"}

def generate_pii_examples(n):
    if n <= 0:
        return []
    print(f"Generating {n} synthetic comprehensive-PII examples...")
    rng = random.Random(46)
    return [_build_pii_example(rng) for _ in range(n)]

# =====================================================================
# Source 8: synthetic VARIANT augmentation. Real variant-annotated open
# corpora are scarce (tmvar_v2 tops out around ~1-1.5k spans; tmvar_v3 would
# help but requires a raw NCBI FTP download this pipeline can't fetch), so
# VARIANT is the hardest category to balance from real data alone. Realistic
# HGVS/rsID/legacy mutation notation is synthesized in genomics-sentence
# templates to bring it up toward parity with the other categories.
# =====================================================================
AMINO_ACIDS_3 = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
                 "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
AMINO_ACIDS_1 = list("ARNDCQEGHILKMFPSTWYV")
DNA_BASES = ["A", "C", "G", "T"]
# Real, well-known named variants -- both with and without the "p." prefix, since both
# styles appear in real text (e.g. "p.Val600E" and bare "T790M"/"L858R").
NAMED_VARIANTS = [
    ["Delta"], ["Omicron"], ["Alpha"], ["Beta"], ["Gamma"],
    ["deltaF508"], ["p.Val600E"], ["p.R175H"], ["c.1521_1523delCTT"],
    ["T790M"], ["L858R"], ["N501Y"], ["E484K"], ["V600E"], ["G12D"], ["G12C"],
]

def _variant_tokens(rng):
    fmt = rng.choice(["hgvs_c", "hgvs_p_long", "hgvs_p_short", "rsid", "legacy"])
    if fmt == "hgvs_c":
        return [f"c.{rng.randint(1, 3000)}{rng.choice(DNA_BASES)}>{rng.choice(DNA_BASES)}"]
    if fmt == "hgvs_p_long":
        return [f"p.{rng.choice(AMINO_ACIDS_3)}{rng.randint(1, 900)}{rng.choice(AMINO_ACIDS_3)}"]
    if fmt == "hgvs_p_short":
        return [f"p.{rng.choice(AMINO_ACIDS_1)}{rng.randint(1, 900)}{rng.choice(AMINO_ACIDS_1)}"]
    if fmt == "rsid":
        return [f"rs{rng.randint(1000, 99999999)}"]
    return [f"{rng.choice(AMINO_ACIDS_1)}{rng.randint(1, 900)}{rng.choice(AMINO_ACIDS_1)}"]

def _any_variant_tokens(rng):
    return list(rng.choice(NAMED_VARIANTS)) if rng.random() < 0.35 else _variant_tokens(rng)

VARIANT_SENTENCE_TEMPLATES = [
    ["A", "novel", "VAR", "mutation", "was", "identified", "in", "the", "affected", "gene", "."],
    ["The", "VAR", "variant", "was", "significantly", "associated", "with", "increased", "disease", "risk", "."],
    ["Sequencing", "revealed", "a", "heterozygous", "VAR", "substitution", "in", "the", "proband", "."],
    ["Patients", "carrying", "the", "VAR", "polymorphism", "showed", "a", "reduced", "response", "to", "treatment", "."],
    ["We", "identified", "a", "de", "novo", "VAR", "mutation", "in", "the", "index", "patient", "."],
    ["The", "VAR", "allele", "was", "detected", "by", "Sanger", "sequencing", "in", "all", "affected", "relatives", "."],
    ["Functional", "analysis", "confirmed", "that", "VAR", "disrupts", "normal", "protein", "folding", "."],
    ["Genotyping", "identified", "the", "VAR", "single", "nucleotide", "polymorphism", "in", "affected", "individuals", "."],
    ["The", "pathogenic", "VAR", "change", "was", "confirmed", "by", "targeted", "resequencing", "."],
    ["A", "family", "history", "consistent", "with", "the", "VAR", "mutation", "was", "documented", "."],
    ["Genetic", "testing", "confirmed", "a", "VAR", "variant", "in", "the", "tumor", "sample", "."],
    ["PCR", "testing", "was", "positive", "for", "the", "VAR", "variant", "."],
    ["The", "VAR", "mutation", "is", "known", "to", "confer", "resistance", "to", "standard", "therapy", "."],
    ["Whole", "exome", "sequencing", "revealed", "a", "compound", "heterozygous", "VAR", "mutation", "."],
    ["Carriers", "of", "the", "VAR", "allele", "have", "an", "elevated", "lifetime", "cancer", "risk", "."],
]

def _build_synthetic_variant_example(rng):
    tokens, labels = [], []
    for slot in rng.choice(VARIANT_SENTENCE_TEMPLATES):
        if slot == "VAR":
            slot_tokens = _any_variant_tokens(rng)
            slot_labels = ["B-VARIANT"] + ["I-VARIANT"] * (len(slot_tokens) - 1)
        else:
            slot_tokens = [slot]
            slot_labels = ["O"]
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[lab] for lab in labels], "source": "synthetic_variant"}

def generate_synthetic_variant_examples(n):
    if n <= 0:
        return []
    print(f"Generating {n} synthetic VARIANT-augmentation examples...")
    rng = random.Random(43)
    return [_build_synthetic_variant_example(rng) for _ in range(n)]

# =====================================================================
# Source 8b: synthetic COMBINED clinical-scenario examples.
#
# Every other source teaches one narrow slice: bc5cdr/bionlp2004/ncbi/biored/
# tmvar_v2/linnaeus are dense PubMed-research-abstract prose that never
# mentions a patient name, date, or facility; synthetic_phi and
# synthetic_variant are short templates that only ever populate ONE category
# (NAME/DATE/LOCATION, or VARIANT) with nothing else around it. The model has
# therefore never seen "GENE mentioned inside an ordinary clinical-narrative
# sentence that also has a patient name/date/location in it" -- which is
# exactly the register real test queries use (confirmed directly: "BRAF"
# and other genes/diseases/chemicals get missed specifically in sentences
# shaped like clinical notes, even though the same terms are well represented
# in the PubMed-abstract-style training data). These templates deliberately
# co-occur DISEASE/CHEMICAL/GENE/CELL/SPECIES/VARIANT together with
# NAME/DATE/LOCATION in one natural sentence to close that gap.
# =====================================================================
DISEASE_NAMES_COMMON = [
    ["Malignant", "Melanoma"], ["Type", "2", "Diabetes"], ["Hypertension"],
    ["Rheumatoid", "Arthritis"], ["Cystic", "Fibrosis"], ["Asthma"],
    ["Breast", "Cancer"], ["Lung", "Cancer"], ["Alzheimer's", "Disease"],
    ["Parkinson's", "Disease"], ["Chronic", "Kidney", "Disease"], ["Pneumonia"],
    ["Multiple", "Sclerosis"], ["Crohn's", "Disease"], ["COVID-19"],
    ["Acute", "Myeloid", "Leukemia"], ["Coronary", "Artery", "Disease"],
    ["Major", "Depressive", "Disorder"], ["Osteoporosis"], ["Epilepsy"],
    ["Sepsis"], ["Lupus"], ["Non-small", "Cell", "Lung", "Cancer"],
    ["Acute", "Respiratory", "Distress", "Syndrome"], ["Colorectal", "Cancer"],
    ["Ovarian", "Cancer"], ["Pancreatic", "Cancer"], ["Prostate", "Cancer"],
    ["Congestive", "Heart", "Failure"], ["Atrial", "Fibrillation"], ["Psoriasis"],
    ["Endometriosis"], ["Hepatitis", "C"], ["Tuberculosis"], ["Meningitis"],
]
# Recently-added, rarer disease terms -- a single vocab-list occurrence wasn't enough
# repetition for the model to learn them reliably (confirmed on real test text: "malaria"
# and "pulmonary histoplasmosis" still missed/fragmented, "differentiation syndrome"
# truncated to just "differentiation"), same root cause as DRUG_NAMES_HARD/
# SPECIES_NAMES_HARD, and oversampled the same way in _disease_tokens.
DISEASE_NAMES_HARD = [
    ["Ductal", "Carcinoma"], ["Promyelocytic", "Leukemia"], ["Histoplasmosis"],
    ["Malaria"], ["Urinary", "Tract", "Infection"], ["Bronchitis"],
    ["Differentiation", "Syndrome"], ["QTc", "Prolongation"],
    # "Essential" composed as a generic DISEASE_MODIFIER wasn't enough to overcome its much
    # more common plain-English usage in the real bc5cdr/bionlp corpora ("essential for X
    # activation") -- confirmed directly: "Essential Hypertension" was missed entirely (both
    # words predicted O, "Essential" at 89.9% and "Hypertension" at 66.8%) even in isolation,
    # not just in context. Adding it as its own fixed compound name gives it direct,
    # unambiguous exposure instead of relying on the modifier composing correctly.
    ["Essential", "Hypertension"], ["Secondary", "Hypertension"],
]
# Real clinical dictation almost always prefixes a disease head noun with one of these --
# "relapsed acute promyelocytic leukemia", "invasive ductal carcinoma", "community-acquired
# pneumonia". The synthetic DISEASE_NAMES entries above are otherwise always fixed, complete
# phrases, so the model never saw a modifier prepended to a disease it needs to extend the
# B-/I-DISEASE span across -- it learns to catch the head noun but truncate the modifier off
# the front. Composing modifiers onto the span here (as part of the same entity) teaches
# leftward span extension instead of leaving it fixed-phrase-only.
DISEASE_MODIFIERS = [
    ["acute"], ["chronic"], ["relapsed"], ["recurrent"], ["invasive"], ["metastatic"],
    ["severe"], ["mild"], ["community-acquired"], ["hospital-acquired"], ["progressive"],
    ["localized"], ["advanced"],
    # Laterality, symptom character, and temporality/causality modifiers -- confirmed
    # missing directly on fresh real-note text: "bilateral lower extremity edema" ->
    # "lower extremity edema" (bilateral dropped), "crushing substernal chest pain
    # radiating to the left jaw" -> "chest pain" (three descriptive modifiers dropped),
    # "Essential Hypertension" -> "Hypertension" at 49% confidence (essentially a coin
    # flip). The original 13-entry list only covered severity/onset framing common in
    # oncology-style notes; real symptom descriptions in general medicine use a much
    # wider modifier vocabulary that was entirely absent from training.
    ["bilateral"], ["unilateral"], ["crushing"], ["substernal"], ["exertional"],
    ["positional"], ["intermittent"], ["persistent"], ["new-onset"], ["worsening"],
    ["uncontrolled"], ["poorly-controlled"], ["essential"], ["idiopathic"], ["secondary"],
    ["primary"], ["refractory"], ["stable"], ["unstable"], ["diffuse"],
]
DRUG_NAMES_COMMON = [
    ["Metformin"], ["Ivacaftor"], ["Methotrexate"], ["Ibuprofen"], ["Amoxicillin"],
    ["Dabrafenib"], ["Warfarin"], ["Insulin"], ["Prednisone"], ["Atorvastatin"],
    ["Lisinopril"], ["Omeprazole"], ["Albuterol"], ["Levothyroxine"], ["Pembrolizumab"],
    ["Vancomycin"], ["Piperacillin"], ["Erlotinib"], ["Doxorubicin"], ["Trastuzumab"],
    ["Rituximab"], ["Gemcitabine"], ["Paclitaxel"], ["Cisplatin"], ["Azithromycin"],
    ["Ciprofloxacin"], ["Losartan"], ["Amlodipine"], ["Furosemide"], ["Hydroxychloroquine"],
]
# Multi-word / hyphenated-combination names. Even after adding these once, a retrain still
# truncated them mid-word on real text ("Artemether-lumefan...", "Isavucon...") and still
# mislabeled "Amphotericin B" as GENE -- a single vocab-list entry only gets sampled a
# handful of times across ~30k training records, nowhere near enough repetition for the
# model to reliably continue a B-/I-CHEMICAL span across 6-8 rare subword pieces. DRUG_NAMES_HARD
# is mixed back into the pool at 4x the weight of a common entry (see _drug_tokens) to force
# more repeated exposure to exactly this failure mode.
DRUG_NAMES_HARD = [
    ["All-trans", "retinoic", "acid"], ["Isavuconazonium", "sulfate"],
    ["Amphotericin", "B"], ["Amphotericin", "B", "lipid", "complex"],
    ["Artemether-lumefantrine"], ["Trimethoprim-sulfamethoxazole"],
    ["Piperacillin-tazobactam"], ["Amoxicillin-clavulanate"],
]
# The weakest category across every held-out evaluation so far (0.75-0.85 precision/recall
# vs. 0.90+ for most other types), and unlike the other categories, it was never given a
# vocabulary-expansion pass this session. The real universe of HGNC-registered human gene
# symbols is ~43,000 -- 22 entries was never going to cover realistic clinical usage. This
# is still hand-curated (not a full gazetteer import), but broadens coverage across the
# categories a clinical NER tool actually encounters: oncogenes/tumor suppressors,
# pharmacogenes, cardiac/lipid genes, and monogenic-disease genes.
GENE_SYMBOLS = [
    ["BRAF"], ["TP53"], ["BRCA1"], ["BRCA2"], ["EGFR"], ["KRAS"], ["CFTR"],
    ["APOE"], ["MYC"], ["PTEN"], ["HER2"], ["ALK"], ["JAK2"], ["PIK3CA"],
    ["RB1"], ["APC"], ["VHL"], ["NF1"], ["MLH1"], ["MSH2"], ["RET"], ["ATM"],
    # Oncology: additional oncogenes/tumor suppressors and targeted-therapy biomarkers
    ["NRAS"], ["HRAS"], ["MET"], ["ROS1"], ["NTRK1"], ["NTRK2"], ["NTRK3"], ["ERBB2"],
    ["IDH1"], ["IDH2"], ["FLT3"], ["KIT"], ["PDGFRA"], ["ABL1"], ["BCR"], ["MYCN"],
    ["CDKN2A"], ["SMAD4"], ["STK11"], ["ARID1A"], ["CTNNB1"], ["CCND1"], ["CDK4"],
    ["CDK6"], ["MDM2"], ["AKT1"], ["MTOR"], ["ESR1"], ["PGR"], ["FGFR1"], ["FGFR2"],
    ["FGFR3"], ["BRIP1"], ["PALB2"], ["CHEK2"], ["RAD51C"], ["RAD51D"], ["TSC1"],
    ["TSC2"], ["WT1"], ["EWSR1"], ["SMARCB1"],
    # Pharmacogenes (drug-metabolism / dosing-relevant, common in clinical notes)
    ["CYP2D6"], ["CYP2C19"], ["CYP2C9"], ["CYP3A4"], ["CYP3A5"], ["TPMT"], ["DPYD"],
    ["UGT1A1"], ["VKORC1"], ["SLCO1B1"], ["NUDT15"], ["G6PD"], ["HLA-B"],
    # Cardiac / lipid / clotting
    ["LDLR"], ["PCSK9"], ["APOB"], ["MYH7"], ["MYBPC3"], ["SCN5A"], ["KCNQ1"],
    ["KCNH2"], ["TTN"], ["LMNA"], ["F5"], ["F2"], ["MTHFR"],
    # Monogenic / inherited disease
    ["HTT"], ["SOD1"], ["C9orf72"], ["PSEN1"], ["PSEN2"], ["APP"], ["MECP2"], ["FMR1"],
    ["SMN1"], ["DMD"], ["HBB"], ["HFE"], ["PAH"], ["GBA"], ["SERPINA1"],
]
CELL_TYPES = [
    ["lymphocyte"], ["T", "cell"], ["B", "cell"], ["neutrophil"], ["eosinophil"],
    ["bone", "marrow"], ["macrophage"], ["monocyte"], ["epithelial", "cell"], ["stem", "cell"],
    ["CD8+", "T", "cell"], ["CD4+", "T", "cell"], ["natural", "killer", "cell"],
    ["dendritic", "cell"], ["platelet"], ["fibroblast"], ["red", "blood", "cell"],
]
# Real notes routinely prefix a cell type with a descriptive modifier ("hypersegmented
# neutrophils", "reactive eosinophils", "immature myeloblasts") -- confirmed on real test
# text where "hypersegmented neutrophils" split into two disconnected CELL spans because
# no modifier had ever been composed onto a CELL span in training (same root cause as the
# DISEASE truncation fix below).
CELL_MODIFIERS = [
    ["hypersegmented"], ["reactive"], ["immature"], ["atypical"], ["activated"],
    ["elevated"], ["circulating"],
]
SPECIES_NAMES_COMMON = [
    ["Escherichia", "coli"], ["Streptococcus", "pneumoniae"], ["Staphylococcus", "aureus"],
    ["Mycobacterium", "tuberculosis"], ["Candida", "albicans"], ["Klebsiella", "pneumoniae"],
    ["Pseudomonas", "aeruginosa"], ["Salmonella", "enterica"], ["Clostridium", "difficile"],
    ["Helicobacter", "pylori"], ["Neisseria", "meningitidis"], ["Influenza", "A", "virus"],
]
# Fungal, parasitic and additional viral genera -- the list above was 100% bacteria (+1
# virus), so any fungal or parasitic pathogen was entirely unseen vocabulary. Confirmed on
# real test text: "Plasmodium vivax" missed entirely and "Histoplasma capsulatum" fragmented
# into two disconnected low-confidence spans. A retrain with a single occurrence of each
# fixed "Histoplasma capsulatum" but still only partially fixed "Plasmodium vivax" (span
# over-extended into "trophozoites"), so SPECIES_NAMES_HARD is oversampled 4x like
# DRUG_NAMES_HARD to give these rarer genera the repetition the common bacteria already had
# via real linnaeus/bc5cdr data.
SPECIES_NAMES_HARD = [
    ["Histoplasma", "capsulatum"], ["Aspergillus", "fumigatus"], ["Cryptococcus", "neoformans"],
    ["Coccidioides", "immitis"], ["Plasmodium", "vivax"], ["Plasmodium", "falciparum"],
    ["Toxoplasma", "gondii"], ["Giardia", "lamblia"], ["Trypanosoma", "cruzi"],
    ["Entamoeba", "histolytica"], ["SARS-CoV-2"], ["Hepatitis", "C", "virus"],
    ["Epstein-Barr", "virus"],
]
# Kept separate from SPECIES_NAMES_COMMON/HARD: "human"/"mouse" etc. are common nouns, not proper
# organism names, so they're only used standalone (never as a pathogen "infection by
# human" filler) -- mixed into a subset of templates deliberately, not the general pool.
GENERIC_SPECIES_WORDS = [["human"], ["mouse"], ["Homo", "sapiens"], ["murine"]]
def _disease_tokens(rng):
    pool = DISEASE_NAMES_COMMON + DISEASE_NAMES_HARD * 4
    tokens = list(rng.choice(pool))
    # Real dictation often stacks two or even three modifiers ("crushing substernal chest
    # pain", "relapsed acute promyelocytic leukemia") -- confirmed directly on fresh,
    # unrelated real clinical text where 1-3 descriptive modifiers were dropped from the
    # DISEASE span every time ("bilateral lower extremity edema" -> "lower extremity
    # edema", "Essential Hypertension" -> "Hypertension" at 49% confidence). Bumped from a
    # 40%/30% (single/second) composition rate to 55%/35%/15% (single/second/third) since
    # the original rate under-exposed the model to this pattern relative to how often real
    # notes actually use it.
    if rng.random() < 0.55:
        used = [rng.choice(DISEASE_MODIFIERS)]
        tokens = list(used[0]) + tokens
        if rng.random() < 0.35:
            mod2 = rng.choice([m for m in DISEASE_MODIFIERS if m not in used])
            used.append(mod2)
            tokens = list(mod2) + tokens
            if rng.random() < 0.15:
                mod3 = rng.choice([m for m in DISEASE_MODIFIERS if m not in used])
                tokens = list(mod3) + tokens
    return tokens

def _drug_tokens(rng):
    # DRUG_NAMES_HARD (multi-word/hyphenated) oversampled 4x -- see definition comment.
    pool = DRUG_NAMES_COMMON + DRUG_NAMES_HARD * 4
    return list(rng.choice(pool))

def _gene_tokens(rng):
    return list(rng.choice(GENE_SYMBOLS))

def _cell_tokens(rng):
    tokens = list(rng.choice(CELL_TYPES))
    if rng.random() < 0.35:
        tokens = list(rng.choice(CELL_MODIFIERS)) + tokens
    return tokens

def _species_tokens(rng):
    # SPECIES_NAMES_HARD (fungi/parasites/rarer viruses) oversampled 4x -- see definition
    # comment; GENERIC_SPECIES_WORDS ("human"/"mouse") mixed in at low frequency as before.
    pool = SPECIES_NAMES_COMMON + SPECIES_NAMES_HARD * 4
    if rng.random() < 0.3:
        pool = pool + GENERIC_SPECIES_WORDS
    return list(rng.choice(pool))

COMBINED_SLOT_BUILDERS = {
    "NAME": _phi_name_tokens, "DATE": _phi_date_tokens, "LOC": _phi_loc_tokens,
    "DISEASE": _disease_tokens, "CHEMICAL": _drug_tokens, "GENE": _gene_tokens,
    "CELL": _cell_tokens, "SPECIES": _species_tokens, "VARIANT": _any_variant_tokens,
    "AGECOMPOUND": _age_compound_tokens,
}
COMBINED_SLOT_LABELS = {
    "NAME": "NAME", "DATE": "DATE", "LOC": "LOCATION", "DISEASE": "DISEASE",
    "CHEMICAL": "CHEMICAL", "GENE": "GENE", "CELL": "CELL", "SPECIES": "SPECIES",
    "VARIANT": "VARIANT", "AGECOMPOUND": "AGE",
}

COMBINED_SENTENCE_TEMPLATES = [
    ["The", "biopsy", "confirmed", "an", "aggressive", "DISEASE", ".", "Genetic", "sequencing",
     "revealed", "a", "positive", "mutation", "in", "the", "GENE", "gene", ".", "The", "clinical",
     "trial", "began", "on", "DATE", "at", "LOC", "."],
    ["Patient", "is", "a", "AGECOMPOUND", "male", "presenting", "with", "a", "history", "of",
     "severe", "DISEASE", ".", "He", "was", "prescribed", "CHEMICAL", "starting", "on", "DATE",
     "at", "LOC", "."],
    ["Genomic", "analysis", "of", "SPECIES", "CELL", "cells", "demonstrated", "a", "pathogenic",
     "GENE", "mutation", ".", "The", "patient", ",", "NAME", ",", "was", "referred", "to", "LOC",
     "on", "DATE", "for", "targeted", "therapy", "."],
    ["Testing", "confirmed", "a", "VARIANT", "variant", "in", "the", "patient's", "DISEASE",
     "biopsy", ".", "We", "initiated", "treatment", "with", "CHEMICAL", "at", "LOC", "."],
    ["On", "DATE", ",", "NAME", "presented", "with", "severe", "DISEASE", ".", "Microscopic",
     "evaluation", "revealed", "elevated", "CELL", "count", "in", "bone", "marrow", "samples",
     "collected", "at", "LOC", "."],
    ["Culture", "isolates", "confirmed", "infection", "by", "SPECIES", ".", "The", "team",
     "administered", "CHEMICAL", "alongside", "CHEMICAL", "to", "control", "symptoms", "."],
    ["Patient", "NAME", "was", "seen", "at", "LOC", "on", "DATE", ".", "PCR", "testing", "was",
     "positive", "for", "the", "VARIANT", "variant", "causing", "acute", "DISEASE", "."],
    ["We", "analyzed", "the", "mutation", "of", "the", "tumor", "suppressor", "gene", "GENE",
     "in", "SPECIES", "CELL", "cells", ".", "In", "vitro", "studies", "showed", "that", "SPECIES",
     "exposure", "triggers", "a", "fast", "CELL", "division", "cycle", "."],
    ["NAME", "was", "diagnosed", "with", "DISEASE", "and", "prescribed", "CHEMICAL", "by", "the",
     "care", "team", "at", "LOC", "on", "DATE", "."],
    ["Sequencing", "of", "the", "GENE", "gene", "in", "SPECIES", "CELL", "cells", "revealed", "a",
     "VARIANT", "variant", "associated", "with", "DISEASE", "."],
    ["NAME", "reviewed", "the", "CELL", "biopsy", "and", "confirmed", "DISEASE", "with",
     "a", "VARIANT", "mutation", ".", "Treatment", "with", "CHEMICAL", "began", "at", "LOC",
     "on", "DATE", "."],
    ["Laboratory", "results", "showed", "elevated", "CELL", "levels", "consistent", "with",
     "DISEASE", ".", "The", "patient", "was", "started", "on", "CHEMICAL", "and", "followed",
     "up", "at", "LOC", "on", "DATE", "."],
    ["A", "GENE", "mutation", "was", "identified", "in", "the", "SPECIES", "genome", "during",
     "routine", "screening", "for", "DISEASE", "at", "LOC", "."],
    ["NAME", "contacted", "LOC", "on", "DATE", "reporting", "symptoms", "of", "DISEASE", ".",
     "Blood", "work", "revealed", "abnormal", "CELL", "counts", "and", "testing", "confirmed",
     "a", "VARIANT", "mutation", "."],
    ["Following", "exposure", "to", "SPECIES", ",", "the", "patient", "developed", "DISEASE",
     "and", "was", "treated", "with", "CHEMICAL", "at", "LOC", "beginning", "DATE", "."],
    ["NAME", "'s", "GENE", "test", "came", "back", "positive", "for", "the", "VARIANT",
     "mutation", "linked", "to", "DISEASE", ".", "Follow-up", "care", "was", "arranged", "at",
     "LOC", "for", "DATE", "."],
    ["A", "somatic", "GENE", "mutation", "was", "identified", "within", "exon", "NUM",
     ",", "consistent", "with", "DISEASE", ".", "Treatment", "with", "CHEMICAL", "began",
     "at", "LOC", "on", "DATE", "."],
    ["Pathology", "confirmed", "Grade", "NUM", "DISEASE", "following", "exposure", "to",
     "CHEMICAL", ".", "NAME", "was", "seen", "at", "LOC", "for", "follow-up", "."],
    # Negation / hedging / family-history phrasing. Note: this teaches the model to still
    # correctly find and type the entity SPAN when it appears in these real, extremely
    # common clinical-note contexts -- it does NOT teach assertion status (negated vs.
    # affirmed vs. family-history) since that's a different task (see NegEx-style assertion
    # classification) entirely out of scope for a plain span-tagging NER model. Without
    # these patterns, nothing in training ever put a DISEASE/CHEMICAL mention after "no
    # evidence of", "denied", "rule out", or "family history of" -- real notes are full of
    # exactly this phrasing.
    ["No", "evidence", "of", "DISEASE", "was", "found", "on", "imaging", ".", "Follow-up",
     "scheduled", "at", "LOC", "on", "DATE", "."],
    ["NAME", "denied", "any", "history", "of", "DISEASE", ".", "Continue", "routine",
     "monitoring", "at", "LOC", "."],
    ["Findings", "are", "suspicious", "for", "DISEASE", ";", "CHEMICAL", "was", "held",
     "pending", "further", "workup", "."],
    ["Rule", "out", "DISEASE", "prior", "to", "initiating", "CHEMICAL", ".", "NAME", "will",
     "follow", "up", "on", "DATE", "."],
    ["NAME", "reports", "a", "family", "history", "of", "DISEASE", "in", "a",
     "first-degree", "relative", "."],
    ["The", "mother", "of", "NAME", "was", "diagnosed", "with", "DISEASE", ";", "genetic",
     "counseling", "regarding", "GENE", "was", "recommended", "."],
    # Comma-separated medication list ("Current Medications: X 1000mg BID, Y 20mg daily, Z
    # 40mg daily") -- confirmed directly on real SOAP-note text that this exact format
    # causes the FIRST TWO drugs in a three-item list to be missed entirely (tagged O at
    # 97%+ confidence), with only the LAST item caught. Every other CHEMICAL template
    # narrates one drug at a time in prose ("was prescribed X"), never several in a row
    # each immediately followed by a dose+frequency token, so this exact adjacency pattern
    # had zero prior training exposure.
    ["Current", "Medications", ":", "CHEMICAL", "DOSE", ",", "CHEMICAL", "DOSE", ",",
     "CHEMICAL", "DOSE", "."],
    ["Discharge", "Medications", ":", "CHEMICAL", "DOSE", "and", "CHEMICAL", "DOSE", "."],
    ["NAME", "was", "started", "on", "CHEMICAL", "DOSE", ",", "CHEMICAL", "DOSE", ",", "and",
     "CHEMICAL", "DOSE", "at", "LOC", "."],
    # Two DISEASE mentions joined by "and" in the same clause ("Shortness of breath and
    # bilateral lower extremity edema") -- every other DISEASE template has exactly one
    # disease per sentence, so a second disease immediately following "and" (rather than
    # after a full-stop) was never modeled; confirmed the first disease correctly extends
    # but the SECOND one loses its lead-in modifier in exactly this construction.
    ["Chief", "Complaint", ":", "DISEASE", "and", "DISEASE", "."],
]

def _build_combined_scenario_example(rng):
    tokens, labels = [], []
    for slot in rng.choice(COMBINED_SENTENCE_TEMPLATES):
        if slot == "NUM":
            # A bare small integer that must stay O even directly adjacent to a DISEASE/
            # GENE/VARIANT mention ("exon 9", "Grade 3 anemia"). No prior template ever put
            # a number next to an entity without it BEING the entity, so the model had zero
            # negative signal here -- confirmed on real test text where a bare "9" after
            # "exon" got tagged B-VARIANT and a bare "3" after "Grade" got tagged B-DISEASE.
            slot_tokens = [str(rng.randint(1, 30))]
            slot_labels = ["O"]
        elif slot == "DOSE":
            # A dosage+frequency pair that must stay O directly after a CHEMICAL mention --
            # confirmed missing on real test text: a "Drug DOSEmg FREQ, Drug DOSEmg FREQ,
            # Drug DOSEmg FREQ" medication list (e.g. "Metformin 1000mg BID, Lisinopril
            # 20mg daily, Atorvastatin 40mg daily") caused the FIRST TWO drugs in the list
            # to be missed entirely (tagged O at 97%+ confidence) while only the THIRD was
            # caught -- every prior CHEMICAL template narrates one drug at a time in prose
            # ("was prescribed X"), never a comma-separated dosage list, so the model had
            # no exposure to "drug name immediately followed by a dose token" at all.
            dose_num = rng.choice([5, 10, 20, 25, 40, 50, 75, 81, 100, 250, 325, 500, 650, 1000])
            freq = rng.choice(["daily", "BID", "TID", "QID", "weekly", "nightly"])
            slot_tokens = [f"{dose_num}mg", freq]
            slot_labels = ["O", "O"]
        elif slot in COMBINED_SLOT_BUILDERS:
            slot_tokens = COMBINED_SLOT_BUILDERS[slot](rng)
            slot_labels = [f"B-{COMBINED_SLOT_LABELS[slot]}"] + [f"I-{COMBINED_SLOT_LABELS[slot]}"] * (len(slot_tokens) - 1)
        else:
            slot_tokens = [slot]
            slot_labels = ["O"]
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[lab] for lab in labels], "source": "synthetic_combined"}

def generate_combined_scenario_examples(n):
    if n <= 0:
        return []
    print(f"Generating {n} synthetic combined-clinical-scenario examples...")
    rng = random.Random(44)
    return [_build_combined_scenario_example(rng) for _ in range(n)]

# =====================================================================
# Vocabulary-holdout generalization probe.
#
# Every other test-split record is, structurally, drawn from the SAME finite
# vocabulary lists as train -- train and test differ only in which combination of
# terms/templates landed in which hash bucket, not in whether the vocabulary itself was
# ever seen. That means held-out-set F1 measures "does the model handle a new sentence
# built from familiar words," not "does it generalize to genuinely unseen clinical terms" --
# a real methodological gap for any claim about generalization.
#
# These four lists are used ONLY here, deliberately excluded from GENE_SYMBOLS/
# DISEASE_NAMES_*/DRUG_NAMES_*/SPECIES_NAMES_* above, and every example built from them is
# force-routed to the test split regardless of its hash bucket (see split_for() in main())
# -- so the model never sees these specific terms during training, and test-set
# performance on them is a genuine unseen-vocabulary measurement.
# =====================================================================
GENE_SYMBOLS_HELDOUT = [
    ["JAK1"], ["JAK3"], ["KDR"], ["MAP2K1"], ["GNAS"], ["SF3B1"], ["TERT"], ["ARID2"],
    ["BAP1"], ["CDH1"], ["POLE"], ["POLD1"], ["ATRX"], ["DAXX"], ["MSH6"], ["PMS2"],
    ["BARD1"], ["RAD50"], ["NBN"], ["XRCC2"],
]
DISEASE_NAMES_HELDOUT = [
    ["Osteomyelitis"], ["Endocarditis"], ["Pyelonephritis"], ["Cellulitis"], ["Pericarditis"],
]
DRUG_NAMES_HELDOUT = [
    ["Rivaroxaban"], ["Empagliflozin"], ["Semaglutide"], ["Tofacitinib"], ["Ustekinumab"],
    ["Nivolumab"],
]
SPECIES_NAMES_HELDOUT = [
    ["Legionella", "pneumophila"], ["Bordetella", "pertussis"], ["Vibrio", "cholerae"],
    ["Leishmania", "donovani"], ["Cytomegalovirus"],
]

HELDOUT_SLOT_BUILDERS = dict(COMBINED_SLOT_BUILDERS)
HELDOUT_SLOT_BUILDERS.update({
    "DISEASE": lambda rng: list(rng.choice(DISEASE_NAMES_HELDOUT)),
    "CHEMICAL": lambda rng: list(rng.choice(DRUG_NAMES_HELDOUT)),
    "GENE": lambda rng: list(rng.choice(GENE_SYMBOLS_HELDOUT)),
    "SPECIES": lambda rng: list(rng.choice(SPECIES_NAMES_HELDOUT)),
})

def _build_vocab_holdout_example(rng):
    tokens, labels = [], []
    for slot in rng.choice(COMBINED_SENTENCE_TEMPLATES):
        if slot == "NUM":
            slot_tokens = [str(rng.randint(1, 30))]
            slot_labels = ["O"]
        elif slot == "DOSE":
            dose_num = rng.choice([5, 10, 20, 25, 40, 50, 75, 81, 100, 250, 325, 500, 650, 1000])
            freq = rng.choice(["daily", "BID", "TID", "QID", "weekly", "nightly"])
            slot_tokens = [f"{dose_num}mg", freq]
            slot_labels = ["O", "O"]
        elif slot in HELDOUT_SLOT_BUILDERS:
            slot_tokens = HELDOUT_SLOT_BUILDERS[slot](rng)
            slot_labels = [f"B-{COMBINED_SLOT_LABELS[slot]}"] + [f"I-{COMBINED_SLOT_LABELS[slot]}"] * (len(slot_tokens) - 1)
        else:
            slot_tokens = [slot]
            slot_labels = ["O"]
        tokens.extend(slot_tokens)
        labels.extend(slot_labels)
    return {"tokens": tokens, "ner_tags": [master_label2id[lab] for lab in labels], "source": "synthetic_vocab_holdout"}

def generate_vocab_holdout_examples(n):
    if n <= 0:
        return []
    print(f"Generating {n} vocabulary-holdout generalization-probe examples (test-split only)...")
    rng = random.Random(45)
    return [_build_vocab_holdout_example(rng) for _ in range(n)]

# =====================================================================
# Source 9: i2b2.jsonl (local file) -- n2c2 2018 Track 2 medication/ADE
# relation-extraction data, reformatted as instruction/context/response
# rows. This is real clinical-note text (not PubMed abstracts), which
# directly helps the domain-mismatch problem. Two things are extracted:
#   1. The tagged SUBJECT/OBJECT entity pair per row -> CHEMICAL (drug,
#      always the object) and DISEASE (for Reason-Drug/ADE-Drug subjects;
#      the other relation types' subjects are dosage attributes like
#      strength/route/frequency, not in our schema, so left as "O").
#   2. MIMIC-style de-identification placeholders like [**Hospital1 18**]
#      or [**2116-1-31**] -- the real PHI value is already redacted before
#      release, but the bracket content still encodes which category it
#      was. Each placeholder is replaced with a realistic synthetic value
#      of the matching category (NAME/DATE/LOCATION), tagged accordingly;
#      brackets outside our schema (phone, age, ID, ...) get a neutral
#      untagged filler so the sentence still reads naturally.
# =====================================================================
I2B2_RELATION_SUBJECT_TYPE = {
    "Reason-Drug": "DISEASE",
    "ADE-Drug": "DISEASE",
}
BRACKET_RE = re.compile(r"\[\*\*(.*?)\*\*\]")
I2B2_TAG_RE = re.compile(r"<<(SUBJECT|OBJECT)>>(.*?)<</\1>>")
NEUTRAL_FILLERS = ["12345", "555-0142", "42", "Acme Corp"]

# The single largest bracket bucket (~1500 occurrences) has no descriptive text at all,
# just bare digit/dash/slash content -- but inspection of the actual raw values shows
# these are overwhelmingly MM-DD or YYYY-M-D date shorthand (e.g. "3-22", "2182-4-25"),
# not IDs or ages, which almost always carry a descriptive label like "Age over #" or
# "Numeric Identifier #" instead of appearing bare. A bare number with a "-" or "/"
# separator and 2-3 numeric parts is classified as DATE; a lone number with no separator
# stays unmapped (genuinely ambiguous -- could be an age, an ID, ...).
_BARE_DATE_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?$")

def _bracket_category(content):
    if _BARE_DATE_RE.match(content.strip()):
        return "DATE"
    normalized = re.sub(r"[\d\-/]+", "#", content).strip()
    if not normalized or normalized == "#":
        return None
    if "Name" in normalized or "Initial" in normalized or normalized.startswith("Doctor"):
        return "NAME"
    if "Month" in normalized or "Date" in normalized or "Year" in normalized:
        return "DATE"
    if "Hospital" in normalized or "Location" in normalized or "State" in normalized or "Country" in normalized:
        return "LOCATION"
    return None  # Telephone/Fax, Age, Numeric Identifier, Company, Pager, Unit Number, ...

def _synthetic_replacement_tokens(category, rng):
    if category == "NAME":
        return [rng.choice(FIRST_NAMES + LAST_NAMES)]
    if category == "DATE":
        return _phi_date_tokens(rng)
    if category == "LOCATION":
        if rng.random() < 0.5:
            return rng.choice(HOSPITALS).split()
        return [rng.choice(CITIES) + ",", rng.choice(STATES)]
    return None

def fetch_i2b2_examples():
    path = os.path.join(BASE_DIR, "i2b2.jsonl")
    if not os.path.exists(path):
        print("i2b2.jsonl not found locally, skipping real clinical-note source.")
        return []
    rng = random.Random(123)
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                resp = json.loads(rec["response"])
            except Exception:
                continue
            relation = resp.get("Relation", "")
            subject_type = I2B2_RELATION_SUBJECT_TYPE.get(relation)  # None -> subject left "O"
            context = rec.get("context", "")

            # Keep only the annotated prefix through the closing OBJECT tag (or SUBJECT
            # if no OBJECT tag) -- everything after that is a redundant plain-text repeat
            # of the same content baked into how this field was constructed.
            m_obj_close = re.search(r"<</OBJECT>>", context)
            m_subj_close = re.search(r"<</SUBJECT>>", context)
            cut = m_obj_close.end() if m_obj_close else (m_subj_close.end() if m_subj_close else None)
            if cut is None:
                continue
            context = context[:cut]

            # Single unified pass: collect every "replacement region" (de-id brackets to
            # substitute, SUBJECT/OBJECT tags to strip) in the ORIGINAL text, sorted by
            # position, then build the output string once so offsets never need
            # re-mapping across multiple passes.
            regions = []  # (start, end, replacement_text, entity_type_or_None)
            for m in BRACKET_RE.finditer(context):
                cat = _bracket_category(m.group(1))
                if cat is None:
                    regions.append((m.start(), m.end(), rng.choice(NEUTRAL_FILLERS), None))
                else:
                    repl = " ".join(_synthetic_replacement_tokens(cat, rng))
                    regions.append((m.start(), m.end(), repl, cat))
            for m in I2B2_TAG_RE.finditer(context):
                tag_kind, text_val = m.group(1), m.group(2)
                if tag_kind == "OBJECT":
                    # The object tag is often re-inserted right after the subject rather
                    # than at its natural position in the prose, causing the drug name to
                    # appear twice (e.g. "Topiramate 25mg PO BID Topiramate"). If the
                    # object text already appears earlier in the untagged prefix, tag
                    # that natural occurrence instead and drop the inserted duplicate.
                    natural_pos = context[:m.start()].rfind(text_val)
                    if natural_pos != -1:
                        regions.append((natural_pos, natural_pos + len(text_val), text_val, "CHEMICAL"))
                        regions.append((m.start(), m.end(), "", None))
                        continue
                ent_type = "CHEMICAL" if tag_kind == "OBJECT" else subject_type
                regions.append((m.start(), m.end(), text_val, ent_type))
            regions.sort(key=lambda r: r[0])

            out, pos, cursor = [], 0, 0
            entity_char_spans = []
            for start, end, repl, ent_type in regions:
                if start < cursor:
                    continue  # skip any overlapping region (shouldn't happen)
                out.append(context[cursor:start])
                pos += len(context[cursor:start])
                span_start = pos
                out.append(repl)
                pos += len(repl)
                if ent_type:
                    entity_char_spans.append((span_start, pos, ent_type))
                cursor = end
            out.append(context[cursor:])
            final_text = "".join(out)

            for chunk_start, chunk_end in _split_sentences_with_offsets(final_text.replace("\n", " ")):
                chunk_text = final_text.replace("\n", " ")[chunk_start:chunk_end]
                local_entities = [
                    {"offsets": [[s - chunk_start, e - chunk_start]], "type": t}
                    for s, e, t in entity_char_spans if chunk_start <= s and e <= chunk_end
                ]
                tokens_with_offsets = _tokenize_with_offsets(chunk_text)
                tokens = [t for t, _, _ in tokens_with_offsets]
                tags = [master_label2id["O"]] * len(tokens)
                for ent in local_entities:
                    est, eend = ent["offsets"][0]
                    first = True
                    for i, (_, ts, te) in enumerate(tokens_with_offsets):
                        if max(ts, est) < min(te, eend):
                            tags[i] = master_label2id[f"{'B' if first else 'I'}-{ent['type']}"]
                            first = False
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": tags, "source": "i2b2_meds"})
                if MAX_EXAMPLES > 0 and len(examples) >= MAX_EXAMPLES:
                    print(f"Loaded i2b2.jsonl: {len(examples)} examples")
                    return examples
    print(f"Loaded i2b2.jsonl: {len(examples)} examples")
    return examples

# Some sources (BioRED in particular) have I-X immediately after O in their own raw
# annotations -- an upstream artifact around tokens split by punctuation inside a single
# entity mention (e.g. a chemical formula broken up by parentheses). That's invalid BIO,
# so any orphaned I-X (not preceded by B-X or I-X of the same type) is promoted to B-X.
# This never discards entity information, it only corrects the boundary marker.
def repair_bio_tags(tag_ids):
    fixed = list(tag_ids)
    changed = False
    prev_label = "O"
    for i, tid in enumerate(fixed):
        label = master_labels[tid] if 0 <= tid < len(master_labels) else "O"
        if label.startswith("I-"):
            ent_type = label[2:]
            if prev_label not in (f"B-{ent_type}", f"I-{ent_type}"):
                tid = master_label2id[f"B-{ent_type}"]
                fixed[i] = tid
                changed = True
                label = f"B-{ent_type}"
        prev_label = label
    return fixed, changed

# =====================================================================
# Multi-label greedy balancing. Real-source volumes are wildly uneven
# (GENE/DISEASE/CHEMICAL naturally in the tens of thousands, VARIANT
# capped around ~1-2k even with synthetic augmentation), so exact parity
# isn't achievable, but this brings every category to the same target
# ceiling wherever supply allows, keeping an example only while at least
# one of the entity types it contains is still under target -- so a
# sentence carrying both an over-target GENE and an under-target VARIANT
# still gets kept for the VARIANT's sake.
# =====================================================================
def balance_examples(all_examples, target_per_type, no_entity_fraction=0.12, seed=42):
    rng = random.Random(seed)
    shuffled = all_examples[:]
    rng.shuffle(shuffled)

    counts = {t: 0 for t in ENTITY_TYPES}
    kept = []
    no_entity_examples = []

    for ex in shuffled:
        types_present = set()
        for t in ex["ner_tags"]:
            label = master_labels[t]
            if label.startswith("B-"):
                types_present.add(label[2:])
        if not types_present:
            no_entity_examples.append(ex)
            continue
        if any(counts[t] < target_per_type for t in types_present):
            kept.append(ex)
            for t in types_present:
                counts[t] += sum(1 for tag in ex["ner_tags"] if master_labels[tag] == f"B-{t}")

    no_entity_cap = int(len(kept) * no_entity_fraction)
    kept.extend(no_entity_examples[:no_entity_cap])
    return kept, counts

# =====================================================================
# Load, align, merge, filter, balance, and split
# =====================================================================
def main():
    print("Starting dataset preparation (pure stdlib, no HF `datasets` install required)...")

    print("Loading dataset tner/bc5cdr ...")
    bc5cdr_examples = fetch_tner_dataset("tner/bc5cdr", "bc5cdr")

    print("Loading dataset tner/bionlp2004 ...")
    bionlp_examples = fetch_tner_dataset("tner/bionlp2004", "bionlp2004")

    print("Loading dataset bigbio/blurb (ncbi_disease) ...")
    ncbi_rows = fetch_rows_via_datasets_server("bigbio/blurb", "ncbi_disease", "train")
    ncbi_examples = [align_ncbi_row(r) for r in ncbi_rows]

    print("Loading dataset disi-unibo-nlp/biored ...")
    biored_rows = fetch_rows_via_datasets_server("disi-unibo-nlp/biored", "default", "train")
    biored_examples = [align_biored_row(r) for r in biored_rows]

    print("Loading dataset bigbio/tmvar_v2 (genetic variant mentions) ...")
    tmvar_examples = fetch_tmvar_v2_examples()

    print("Loading dataset bigbio/linnaeus (species mentions) ...")
    linnaeus_examples = fetch_linnaeus_examples()

    print("Loading i2b2.jsonl (real clinical-note medication/ADE data) ...")
    i2b2_examples = fetch_i2b2_examples()

    SYNTHETIC_PHI_EXAMPLES = int(os.environ.get("PHI_SYNTHETIC_EXAMPLES", "12000") or 0)
    synthetic_phi_examples = generate_synthetic_phi_examples(
        SYNTHETIC_PHI_EXAMPLES if MAX_EXAMPLES <= 0 else min(SYNTHETIC_PHI_EXAMPLES, MAX_EXAMPLES)
    )
    # VARIANT has the thinnest real supply of any category (tmvar_v2 tops out around
    # 1-1.5k spans), and was still the smallest final category after the last rebuild --
    # boosted well past the others here so it can comfortably hit a higher balance target.
    SYNTHETIC_VARIANT_EXAMPLES = int(os.environ.get("PHI_SYNTHETIC_VARIANT_EXAMPLES", "14000") or 0)
    synthetic_variant_examples = generate_synthetic_variant_examples(
        SYNTHETIC_VARIANT_EXAMPLES if MAX_EXAMPLES <= 0 else min(SYNTHETIC_VARIANT_EXAMPLES, MAX_EXAMPLES)
    )
    SYNTHETIC_COMBINED_EXAMPLES = int(os.environ.get("PHI_SYNTHETIC_COMBINED_EXAMPLES", "16000") or 0)
    synthetic_combined_examples = generate_combined_scenario_examples(
        SYNTHETIC_COMBINED_EXAMPLES if MAX_EXAMPLES <= 0 else min(SYNTHETIC_COMBINED_EXAMPLES, MAX_EXAMPLES)
    )
    VOCAB_HOLDOUT_EXAMPLES = int(os.environ.get("PHI_VOCAB_HOLDOUT_EXAMPLES", "1200") or 0)
    vocab_holdout_examples = generate_vocab_holdout_examples(
        VOCAB_HOLDOUT_EXAMPLES if MAX_EXAMPLES <= 0 else min(VOCAB_HOLDOUT_EXAMPLES, MAX_EXAMPLES)
    )
    # No real corpus source has any exposure to phone/SSN/MRN/IP/credit-card/etc. at all --
    # this is the ONLY source for 20 of the 29 categories. 15 templates share these slots
    # unevenly (some categories appear in only 1-2 templates), so a large count is needed
    # for even the rarest categories (BIOMETRIC_ID, CREDIT_CARD, PASSPORT_NUMBER) to get
    # meaningful exposure. These are fixed-format numeric/alphanumeric patterns, though --
    # a much easier recognition task than free-text clinical entities -- so they don't need
    # to hit the same 8000-span balance target as DISEASE/CHEMICAL to be well-learned.
    SYNTHETIC_PII_EXAMPLES = int(os.environ.get("PHI_SYNTHETIC_PII_EXAMPLES", "20000") or 0)
    synthetic_pii_examples = generate_pii_examples(
        SYNTHETIC_PII_EXAMPLES if MAX_EXAMPLES <= 0 else min(SYNTHETIC_PII_EXAMPLES, MAX_EXAMPLES)
    )

    all_examples = (
        bc5cdr_examples + bionlp_examples + ncbi_examples + biored_examples
        + tmvar_examples + linnaeus_examples + i2b2_examples
        + synthetic_phi_examples + synthetic_variant_examples + synthetic_combined_examples
        + vocab_holdout_examples + synthetic_pii_examples
    )
    if not all_examples:
        print("No datasets were loaded. Nothing to merge.")
        sys.exit(0)

    print("\nMerging final dataset...")
    pre_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if len(ex["tokens"]) > 0 and len(ex["tokens"]) == len(ex["ner_tags"])]
    dropped = pre_filter_size - len(all_examples)
    if dropped:
        print(f"Dropped {dropped} empty or malformed records.")

    # A handful of source sentences (mostly from BC5CDR's own sentence-splitter
    # mis-breaking on decimal points in stats notation, e.g. "P = 0.815") are pure
    # numeric/punctuation debris with no real content word in them, e.g.
    # ["9", "/", "7", "."], ["815", ";", "P", "=", "0", "."], or ["0", "and", "31", "."]
    # (only a stopword, no actual content). These carry no NER signal and are noise
    # for fine-tuning, so drop any record with no non-stopword alphabetic token. Short
    # but real phrases (e.g. ["Massive", "bleeding"]) are untouched by this check.
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "his", "her", "their", "our",
        "your", "my", "he", "she", "they", "we", "you", "i", "not", "no", "so", "than",
        "then", "also", "into", "out", "up", "down", "over", "under", "after", "before",
        "between", "during", "while", "about", "above", "below", "per", "vs",
    }
    def _has_content_word(tokens):
        for t in tokens:
            if re.search(r"[A-Za-z]{2,}", t) and t.lower() not in STOPWORDS:
                return True
        return False

    pre_word_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if _has_content_word(ex["tokens"])]
    no_word_dropped = pre_word_filter_size - len(all_examples)
    if no_word_dropped:
        print(f"Dropped {no_word_dropped} records with no real content word (numeric/punctuation/stopword-only debris).")

    # Records with fewer than 4 tokens give a NER model essentially no surrounding
    # context to learn from.
    pre_length_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if len(ex["tokens"]) >= 4]
    too_short_dropped = pre_length_filter_size - len(all_examples)
    if too_short_dropped:
        print(f"Dropped {too_short_dropped} records with fewer than 4 tokens (no usable context).")

    # BC5CDR's own upstream sentence-splitter sometimes glues a truncated tail from one
    # "sentence" onto a structured-abstract section header from the next, e.g. (confirmed
    # verbatim in BC5CDR's own test.json): ["001", ").", "CONCLUSIONS", ":", "Precurarization",
    # "with", "0", "."].
    _ABSTRACT_HEADERS = {
        "BACKGROUND", "METHODS", "METHOD", "RESULTS", "RESULT", "CONCLUSIONS", "CONCLUSION",
        "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "PURPOSE", "DESIGN", "SETTING",
        "PARTICIPANTS", "INTERVENTION", "INTERVENTIONS", "MEASUREMENTS", "MEASURES",
        "FINDINGS", "DISCUSSION", "INTRODUCTION", "SUMMARY", "MATERIALS", "IMPORTANCE", "CONTEXT",
    }
    def _looks_like_split_artifact(tokens):
        for i in range(len(tokens) - 1):
            if tokens[i].upper() == tokens[i] and tokens[i] in _ABSTRACT_HEADERS and tokens[i + 1] == ":":
                return True
        if len(tokens) >= 2 and tokens[0].isdigit() and re.fullmatch(r"[).,;%:]+", tokens[1]):
            return True
        return False

    pre_artifact_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if not _looks_like_split_artifact(ex["tokens"])]
    artifact_dropped = pre_artifact_filter_size - len(all_examples)
    if artifact_dropped:
        print(f"Dropped {artifact_dropped} records with a structured-abstract-header/leading-fragment artifact.")

    repaired = 0
    for ex in all_examples:
        fixed_tags, changed = repair_bio_tags(ex["ner_tags"])
        if changed:
            ex["ner_tags"] = fixed_tags
            repaired += 1
    if repaired:
        print(f"Repaired {repaired} records with orphaned I- tags (promoted to B-).")

    # A handful of rows have corrupted char-offset alignment inherited from the source
    # (a bare quote mark tagged as an entity, elsewhere in the same row a plain word
    # tagged as an entity). The corruption isn't isolated to one token, so the whole row
    # is dropped rather than patched -- any entity span with no alphanumeric character
    # at all is treated as a signal the row's alignment is broken.
    def _has_only_punct_entity(tokens, tags):
        cur_start = None
        for i, t in enumerate(tags + [0]):
            label = master_labels[t] if 0 <= t < len(master_labels) else "O"
            is_b = label.startswith("B-")
            is_o = label == "O"
            if (is_b or is_o) and cur_start is not None:
                span = "".join(tokens[cur_start:i])
                if not re.search(r"[A-Za-z0-9]", span):
                    return True
                cur_start = None
            if is_b:
                cur_start = i
        return False

    pre_corrupt_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if not _has_only_punct_entity(ex["tokens"], ex["ner_tags"])]
    corrupt_dropped = pre_corrupt_filter_size - len(all_examples)
    if corrupt_dropped:
        print(f"Dropped {corrupt_dropped} records with a punctuation-only entity span (corrupted source alignment).")

    # linnaeus's full-text articles occasionally embed raw DNA/primer sequences quoted
    # verbatim and, in a few papers, garbled MathML-to-text extraction artifacts
    # (hundreds of characters long) that leaked into the plain-text dump.
    _DNA_LIKE_RE = re.compile(r"^[ACGTNacgtn]{12,}$")
    _MATHML_PREFIXES = ("feaafiart", "vr0dc8me")
    def _has_garbage_token(tokens):
        for t in tokens:
            if _DNA_LIKE_RE.match(t):
                return True
            if t.startswith(_MATHML_PREFIXES):
                return True
            if len(t) > 40 and re.fullmatch(r"[A-Za-z0-9]+", t) and sum(c.isupper() for c in t) > 5 and sum(c.isdigit() for c in t) > 3:
                return True
        return False

    pre_garbage_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if not _has_garbage_token(ex["tokens"])]
    garbage_dropped = pre_garbage_filter_size - len(all_examples)
    if garbage_dropped:
        print(f"Dropped {garbage_dropped} records containing a raw DNA-sequence literal or MathML-extraction artifact token.")

    # bionlp2004's own original annotations occasionally tag a bare amino-acid-position
    # range as B-protein instead of the actual protein name next to it (confirmed against
    # the raw source). A standalone number is never a real GENE/DISEASE/CHEMICAL/CELL/
    # SPECIES mention, so neutralize just that span to "O". VARIANT and DATE are excluded --
    # numeric-looking variant notation (positions, rsIDs) and ISO/slash-format dates
    # (e.g. "2002-09-18", "09/11/1991") are both legitimately all-digits-and-punctuation.
    # Schema-v2 identifier categories are legitimately digits-and-punctuation too (that's
    # the whole point of most of them) -- confirmed this filter was silently clearing over
    # 12,000 spans on the first run with the new schema, wiping out AGE ("45"), SSN
    # ("482-19-4920"), ACCOUNT_NUMBER, plain-format MEDICAL_RECORD_NUMBER, IP_ADDRESS
    # ("192.168.1.45"), and dash/dot-formatted PHONE/FAX/CREDIT_CARD before they ever
    # reached the balancer -- exactly the categories this schema expansion exists for.
    _PURELY_NUMERIC_RE = re.compile(r"^[\d.,\-/]+$")
    _NUMERIC_EXEMPT_TYPES = {
        "VARIANT", "DATE", "AGE", "PHONE", "FAX", "SSN", "MEDICAL_RECORD_NUMBER",
        "ACCOUNT_NUMBER", "IP_ADDRESS", "CREDIT_CARD",
    }
    numeric_entities_cleared = 0
    for ex in all_examples:
        tags = ex["ner_tags"]
        cur_start, cur_type = None, None
        spans_to_clear = []
        for i, t in enumerate(tags + [0]):
            label = master_labels[t] if 0 <= t < len(master_labels) else "O"
            if (label == "O" or label.startswith("B-")) and cur_start is not None:
                if cur_type not in _NUMERIC_EXEMPT_TYPES and _PURELY_NUMERIC_RE.match("".join(ex["tokens"][cur_start:i])):
                    spans_to_clear.append((cur_start, i))
                cur_start = None
            if label.startswith("B-"):
                cur_start, cur_type = i, label[2:]
        for s, e in spans_to_clear:
            for i in range(s, e):
                tags[i] = master_label2id["O"]
            numeric_entities_cleared += 1
    if numeric_entities_cleared:
        print(f"Cleared {numeric_entities_cleared} purely-numeric entity spans.")

    # Sentences ending mid-clause on a bare function word, or mid-word on a trailing
    # hyphen, or opening an unclosed parenthetical within the last few tokens (e.g.
    # "...( Fig .", "...( p < 0 .") are truncated fragments, not complete sentences --
    # confirmed by direct inspection against multiple sources. A 4+ run of an identical
    # short non-hyphen token (e.g. "NA NA NA NA NA") is a data-table dump, not prose.
    _ENDING_FUNCTION_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "his", "her", "their", "our",
        "your", "my", "he", "she", "they", "we", "you", "i", "not", "no", "so", "than",
        "then", "also", "into", "out", "up", "down", "over", "under", "after", "before",
        "between", "during", "while", "about", "above", "below", "per", "vs", "which",
        "who", "whom", "whose", "when", "where", "why", "how", "because", "although",
        "though", "since", "until", "unless", "without", "within", "among", "across",
        "through", "toward", "towards", "onto", "upon", "via",
    }
    def _is_dangling_end(tokens):
        return tokens[-1].lower().strip(".,;:!?") in _ENDING_FUNCTION_WORDS
    def _ends_mid_word(tokens):
        return tokens[-1].endswith("-")
    def _has_table_dump_pattern(tokens):
        run_tok, run_len = None, 0
        for t in tokens:
            run_len = run_len + 1 if t == run_tok else 1
            run_tok = t
            if run_len >= 4 and len(t) <= 3 and t != "-":
                return True
        return False
    def _opens_unclosed_paren_near_end(tokens, window=5):
        unclosed = []
        for i, t in enumerate(tokens):
            if t in ("(", "["):
                unclosed.append(i)
            elif t in (")", "]") and unclosed:
                unclosed.pop()
        return bool(unclosed) and unclosed[-1] >= len(tokens) - window
    def _is_truncated(tokens):
        return (_is_dangling_end(tokens) or _ends_mid_word(tokens)
                or _has_table_dump_pattern(tokens) or _opens_unclosed_paren_near_end(tokens))

    pre_truncation_filter_size = len(all_examples)
    all_examples = [ex for ex in all_examples if not _is_truncated(ex["tokens"])]
    truncation_dropped = pre_truncation_filter_size - len(all_examples)
    if truncation_dropped:
        print(f"Dropped {truncation_dropped} truncated/incomplete-fragment records.")

    # Some source corpora share underlying PubMed abstracts, and each source only
    # annotates its own entity types, so the exact same sentence can show up multiple
    # times with different -- sometimes contradictory -- label sets. Identical input
    # mapped to different labels is actively harmful for fine-tuning, so dedupe by
    # tokens ALONE, keeping the single most complete (most non-"O" tags) annotation.
    best_by_tokens = {}
    for ex in all_examples:
        key = tuple(ex["tokens"])
        non_o_count = sum(1 for t in ex["ner_tags"] if t != 0)
        if key not in best_by_tokens or non_o_count > best_by_tokens[key][1]:
            best_by_tokens[key] = (ex, non_o_count)
    deduped = [ex for ex, _ in best_by_tokens.values()]
    duplicates = len(all_examples) - len(deduped)
    if duplicates:
        print(f"Dropped {duplicates} duplicate/conflicting-label records (kept the most complete annotation per unique sentence).")
    all_examples = deduped

    print(f"\nPre-balance dataset size: {len(all_examples)} records.")
    print("Pre-balance entity counts:")
    pre_balance_counts = {}
    for ex in all_examples:
        for t in ex["ner_tags"]:
            label = master_labels[t]
            if label.startswith("B-"):
                pre_balance_counts[label[2:]] = pre_balance_counts.get(label[2:], 0) + 1
    for ent_type, cnt in sorted(pre_balance_counts.items(), key=lambda x: -x[1]):
        print(f"  {ent_type:10s}: {cnt}")

    TARGET_PER_TYPE = int(os.environ.get("PHI_TARGET_PER_TYPE", "8000") or 8000)
    if MAX_EXAMPLES <= 0:
        all_examples, balanced_counts = balance_examples(all_examples, TARGET_PER_TYPE)
        print(f"\nBalanced to target {TARGET_PER_TYPE} spans/type (capped by real supply for some categories).")

    random.Random(42).shuffle(all_examples)
    print(f"\nSuccess! Final Master Dataset Size: {len(all_examples)} records.")

    # A deterministic hash-of-tokens split (rather than each source's own train/valid/test
    # boundaries) so identical sentences that appear under different source splits always
    # land in the same partition -- otherwise deduping across sources could leak the same
    # sentence into both train and test.
    # synthetic_vocab_holdout records are always force-routed to test regardless of hash
    # bucket -- these use gene/disease/drug/species terms deliberately excluded from every
    # other generator, so the ONLY way they end up in train is if this override is skipped.
    # That's what makes their test-split score a genuine unseen-vocabulary measurement
    # rather than "a new combination of already-seen words."
    def split_for(ex):
        if ex.get("source") == "synthetic_vocab_holdout":
            return "test"
        digest = hashlib.md5(" ".join(ex["tokens"]).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "validation"
        return "test"

    by_split = {"train": [], "validation": [], "test": []}
    for ex in all_examples:
        by_split[split_for(ex)].append(ex)

    for split_name, path in SPLIT_PATHS.items():
        with open(path, "w", encoding="utf-8") as f:
            for ex in by_split[split_name]:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_name}: {len(by_split[split_name])} records -> {path}")

    print("\nRecords per source:")
    source_counts = {}
    for ex in all_examples:
        source_counts[ex["source"]] = source_counts.get(ex["source"], 0) + 1
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:16s}: {cnt:6d} ({100*cnt/len(all_examples):5.1f}%)")

    print("\nFinal entity spans per label (B- tag counts):")
    entity_counts = {}
    for ex in all_examples:
        for t in ex["ner_tags"]:
            label = master_labels[t]
            if label.startswith("B-"):
                entity_counts[label[2:]] = entity_counts.get(label[2:], 0) + 1
    for ent_type, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {ent_type:10s}: {cnt}")

    labels_path = os.path.join(BASE_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": master_labels,
            "label2id": master_label2id,
            "id2label": {str(i): l for i, l in enumerate(master_labels)},
        }, f, indent=2)
    print(f"\nLabel schema saved to: {labels_path}")

if __name__ == "__main__":
    main()
