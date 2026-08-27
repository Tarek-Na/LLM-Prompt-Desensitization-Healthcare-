"""
Builds merged_clinical_phi_v16.{train,validation,test}.jsonl on top of v15 (the 7-category
schema with SPECIES and CELL dropped). Adds exactly one thing: CADEC's real 127-record
TRAIN split as new DISEASE training data. Nothing else in v15 is touched, so this is
additive only -- every existing NAME/LOCATION/GENE/CHEMICAL/VARIANT/DATE record and tag is
carried over unchanged, which is the whole point: improve DISEASE without moving anything
else.

Why CADEC specifically: this project's existing real DISEASE sources (NCBI-disease,
BC5CDR) are both formal PubMed-abstract corpora. DISEASE's actual weak spot, confirmed by
re-evaluating against CADEC's held-out 22-record test split, is informal patient-forum
language ("excrutiating pain", "swollen knees", written in first person) -- a register nothing
else in the training data represents. CADEC's TRAIN split (127 records, disjoint from its own
validation/test splits used for eval) is real, human-annotated, exactly-domain-matched text
that was never used in training before now. This mirrors the same move already proven for
GENE and CHEMICAL in v12/v13 -- add the benchmark's own real train split, never its eval
split -- just applied to the one category that still needs it.

No subsampling or capping here, unlike BC2GM/BC4CHEMD in v13: 127 records is small enough
relative to the ~76k record dataset that it cannot reproduce the class-imbalance regression
that capping was built to prevent. Checked directly below (see the printed entity-count
ratios) rather than assumed.
"""
import os
import json
import random
import hashlib
import urllib.request
import urllib.parse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
VERSION = "v16"
REQUEST_TIMEOUT = 60

BASE_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, "merged_clinical_phi_v15.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, "merged_clinical_phi_v15.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, "merged_clinical_phi_v15.test.jsonl"),
}
OUTPUT_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.test.jsonl"),
}
LABELS_PATH = os.path.join(REPO_ROOT, f"labels_{VERSION}.json")

# Same 7-category schema as v15 -- unchanged.
master_labels = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-CHEMICAL", "I-CHEMICAL",
    "B-GENE", "I-GENE",
    "B-VARIANT", "I-VARIANT",
    "B-NAME", "I-NAME",
    "B-DATE", "I-DATE",
    "B-LOCATION", "I-LOCATION",
]
master_label2id = {label: i for i, label in enumerate(master_labels)}


# =====================================================================
# HTTP helpers (stdlib only), identical pattern to build_lora_dataset_v13.py.
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
    rows, offset = [], 0
    while offset < total_rows:
        length = min(page_size, total_rows - offset)
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
        time.sleep(0.3)
    print(f"Loaded {dataset}/{config}/{split}: {len(rows)} rows")
    return rows


# =====================================================================
# CADEC train -> real DISEASE positives, informal patient-forum register.
# =====================================================================
def fetch_cadec_train_examples():
    rows = fetch_rows_via_datasets_server("JoelMba/CADEC", "default", "train")
    # CADEC's own ClassLabel order, confirmed via the datasets-server API rather than
    # assumed: 0=B-Disorders, 1=I-Disorders, 2=O.
    names = ["B-Disorders", "I-Disorders", "O"]
    examples = []
    for row in rows:
        tokens = row.get("tokens", [])
        if not tokens:
            continue
        tags = []
        for t in row.get("ner_tags", []):
            native = names[t] if 0 <= t < len(names) else "O"
            if native == "B-Disorders":
                tags.append(master_label2id["B-DISEASE"])
            elif native == "I-Disorders":
                tags.append(master_label2id["I-DISEASE"])
            else:
                tags.append(master_label2id["O"])
        examples.append({"tokens": tokens, "ner_tags": tags, "source": "cadec_real"})
    return examples


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


def entity_counts(records):
    counts = {}
    for rec in records:
        for t in rec["ner_tags"]:
            label = master_labels[t]
            if label.startswith("B-"):
                counts[label[2:]] = counts.get(label[2:], 0) + 1
    return counts


def main():
    rng = random.Random(46)
    print(f"Building merged_clinical_phi_{VERSION} -- v15 records unchanged, plus CADEC's "
          f"real 127-record train split added as new DISEASE data (informal patient-forum "
          f"register, the diagnosed gap in DISEASE's existing formal-PubMed training data). "
          f"CADEC's own validation/test splits are never touched, so the eval benchmark "
          f"stays genuinely held out.\n")

    base_by_split = {}
    for split_name, path in BASE_SPLIT_PATHS.items():
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        base_by_split[split_name] = rows
        print(f"{split_name:12s}: loaded {len(rows)} records from v15, unchanged.")

    print("\nFetching CADEC train split...")
    cadec_examples = fetch_cadec_train_examples()
    pre_filter = len(cadec_examples)
    cadec_examples = [ex for ex in cadec_examples if len(ex["tokens"]) >= 2]
    if pre_filter != len(cadec_examples):
        print(f"Dropped {pre_filter - len(cadec_examples)} records shorter than 2 tokens.")

    by_split = {"train": [], "validation": [], "test": []}
    for ex in cadec_examples:
        by_split[split_for(ex["tokens"])].append(ex)
    print(f"CADEC new examples: {len(cadec_examples)} total -- "
          f"{len(by_split['train'])} train / {len(by_split['validation'])} validation / "
          f"{len(by_split['test'])} test.")

    print("\nWriting output files...")
    total_by_split = {}
    for split_name in ("train", "validation", "test"):
        before_counts = entity_counts(base_by_split[split_name])
        merged = base_by_split[split_name] + by_split[split_name]
        rng.shuffle(merged)
        total_by_split[split_name] = merged
        after_counts = entity_counts(merged)
        with open(OUTPUT_SPLIT_PATHS[split_name], "w", encoding="utf-8") as f:
            for ex in merged:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name:12s}: {len(merged):6d} records -> "
              f"{os.path.basename(OUTPUT_SPLIT_PATHS[split_name])}")
        print(f"{'':14s}DISEASE spans: {before_counts.get('DISEASE', 0)} -> "
              f"{after_counts.get('DISEASE', 0)} "
              f"(+{after_counts.get('DISEASE', 0) - before_counts.get('DISEASE', 0)})")
        for other in ("NAME", "DATE", "LOCATION", "GENE", "CHEMICAL", "VARIANT"):
            b, a = before_counts.get(other, 0), after_counts.get(other, 0)
            assert b == a, (
                f"{other} span count changed in {split_name} ({b} -> {a}) -- this build is "
                f"supposed to be additive-only for DISEASE, something is wrong."
            )
        print(f"{'':14s}NAME/DATE/LOCATION/GENE/CHEMICAL/VARIANT span counts: unchanged, "
              f"confirmed.")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "labels": master_labels,
            "label2id": master_label2id,
            "id2label": {str(i): l for i, l in enumerate(master_labels)},
        }, f, indent=2)
    print(f"\nLabel schema (unchanged, 7-category) written to {os.path.basename(LABELS_PATH)}")

    print("\nChecking for cross-split leakage (identical token sequence in more than one "
          "split)...")
    seq_sets = {name: {" ".join(r["tokens"]) for r in recs}
                for name, recs in total_by_split.items()}
    any_leak = False
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = seq_sets[a] & seq_sets[b]
        if overlap:
            any_leak = True
            print(f"  LEAK: {len(overlap)} identical records shared between {a} and {b}!")
        else:
            print(f"  {a} / {b}: no overlap.")
    if any_leak:
        raise SystemExit(
            "Refusing to keep output: cross-split leakage detected above -- investigate "
            "before training on this data."
        )

    train_disease = entity_counts(total_by_split["train"])
    disease_count = train_disease.get("DISEASE", 0)
    if disease_count:
        print(f"\nRatios in final train split (for comparison against v11's healthy "
              f"GENE:SPECIES ~1.8:1 / v12's broken ~3.8:1, the same class-imbalance check "
              f"this project already learned to run before trusting a data addition):")
        for other in ("NAME", "DATE", "LOCATION", "GENE", "CHEMICAL", "VARIANT"):
            other_count = train_disease.get(other, 0)
            print(f"  {other}:DISEASE = {other_count / disease_count:.2f}:1")


if __name__ == "__main__":
    main()
