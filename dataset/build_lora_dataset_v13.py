"""
Builds merged_clinical_phi_v13.{train,validation,test}.jsonl on top of v11 (not v12 --
v12's full-corpus ingestion is being redone here with caps, not layered further).

v12 added the FULL BC2GM/BC4CHEMD train splits as real GENE/CHEMICAL data, and it worked
spectacularly for those two categories (GENE relaxed F1 27.83%->85.91%, CHEMICAL
46.12%->89.61%). But it wrecked SPECIES, the deliberately-untouched zero-shot check:
64.53%->22.71% relaxed F1, with the confusion matrix showing the cause directly -- 625
spurious GENE predictions and 385 spurious CHEMICAL predictions on Species-800 text that
weren't there before. This is class-imbalance interference: v12 pushed CHEMICAL:SPECIES
from 1.6:1 (v11, where SPECIES was healthy) to 5.3:1, and GENE:SPECIES from 1.8:1 to 3.8:1,
skewing the model's prior on ambiguous biological text toward the now-dominant classes.

Fix: same real BC2GM/BC4CHEMD data, but capped by subsampling so the final GENE/CHEMICAL
counts land around 2.5-3x SPECIES's count instead of 3.8-5.3x -- enough real signal to
still meaningfully help GENE/CHEMICAL, without repeating the imbalance that broke SPECIES.
This is an estimate based on the v11/v12 data points, not a guarantee; it needs a retrain +
re-eval (including Species-800, still left untouched here) to confirm it lands right.
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
VERSION = "v13"
REQUEST_TIMEOUT = 60

BASE_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, "merged_clinical_phi_v11.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, "merged_clinical_phi_v11.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, "merged_clinical_phi_v11.test.jsonl"),
}
OUTPUT_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.test.jsonl"),
}
LABELS_PATH = os.path.join(REPO_ROOT, f"labels_{VERSION}.json")

DROP_SOURCES = {"bc2gm_negative", "bc4chemd_negative"}

# Subsample fractions of the full BC2GM/BC4CHEMD train splits, chosen to land GENE and
# CHEMICAL around 2.5-3x SPECIES's 6,236-span count (v11's healthy ratio was ~1.6-1.8x,
# v12's broken ratio was ~3.8-5.3x) -- an estimate to verify by retraining, not a guarantee.
BC2GM_KEEP_FRACTION = float(os.environ.get("PHI_BC2GM_KEEP_FRACTION", "0.5"))
BC4CHEMD_KEEP_FRACTION = float(os.environ.get("PHI_BC4CHEMD_KEEP_FRACTION", "0.3"))

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
# HTTP helpers (stdlib only), identical pattern to build_dataset.py/Calldataset.py.
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
# BC2GM train -> real GENE positives, subsampled.
# =====================================================================
def fetch_bc2gm_train_examples():
    rows = fetch_rows_via_datasets_server("spyysalo/bc2gm_corpus", "bc2gm_corpus", "train")
    names = ["O", "B-GENE", "I-GENE"]
    examples = []
    for row in rows:
        tokens = row.get("tokens", [])
        if not tokens:
            continue
        tags = [master_label2id[names[t]] if 0 <= t < len(names) else master_label2id["O"]
                for t in row.get("ner_tags", [])]
        examples.append({"tokens": tokens, "ner_tags": tags, "source": "bc2gm_real"})
    return examples

# =====================================================================
# BC4CHEMD train -> real CHEMICAL positives, subsampled.
# =====================================================================
def fetch_bc4chemd_train_examples():
    rows = fetch_rows_via_datasets_server("disi-unibo-nlp/bc4chemd", "default", "train")
    examples = []
    for row in rows:
        tokens = row.get("tokens", [])
        if not tokens:
            continue
        tags = []
        for t in row.get("ner_tags", []):
            if t in ("B-CHEMICAL", "I-CHEMICAL"):
                tags.append(master_label2id[t])
            else:
                tags.append(master_label2id["O"])
        examples.append({"tokens": tokens, "ner_tags": tags, "source": "bc4chemd_real"})
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

def main():
    rng = random.Random(45)
    print(f"Building merged_clinical_phi_v13 -- same real BC2GM/BC4CHEMD data as v12, but "
          f"capped ({BC2GM_KEEP_FRACTION:.0%} of BC2GM, {BC4CHEMD_KEEP_FRACTION:.0%} of "
          f"BC4CHEMD) to avoid the GENE/CHEMICAL:SPECIES imbalance that broke SPECIES "
          f"recall in v12. Species-800 stays untouched.\n")

    base_by_split = {}
    for split_name, path in BASE_SPLIT_PATHS.items():
        kept, dropped = 0, 0
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("source") in DROP_SOURCES:
                    dropped += 1
                    continue
                rows.append(rec)
                kept += 1
        base_by_split[split_name] = rows
        print(f"{split_name:12s}: loaded {kept} records from v11 "
              f"(dropped {dropped} superseded bc2gm_negative/bc4chemd_negative records).")

    print()
    print("Fetching BC2GM train split...")
    bc2gm_examples = fetch_bc2gm_train_examples()
    print("Fetching BC4CHEMD train split...")
    bc4chemd_examples = fetch_bc4chemd_train_examples()

    pre_bc2gm = len(bc2gm_examples)
    pre_bc4chemd = len(bc4chemd_examples)
    if 0 < BC2GM_KEEP_FRACTION < 1:
        bc2gm_examples = rng.sample(bc2gm_examples, int(pre_bc2gm * BC2GM_KEEP_FRACTION))
    if 0 < BC4CHEMD_KEEP_FRACTION < 1:
        bc4chemd_examples = rng.sample(bc4chemd_examples, int(pre_bc4chemd * BC4CHEMD_KEEP_FRACTION))
    print(f"Subsampled BC2GM: {pre_bc2gm} -> {len(bc2gm_examples)} records "
          f"({BC2GM_KEEP_FRACTION:.0%}).")
    print(f"Subsampled BC4CHEMD: {pre_bc4chemd} -> {len(bc4chemd_examples)} records "
          f"({BC4CHEMD_KEEP_FRACTION:.0%}).")

    new_examples = bc2gm_examples + bc4chemd_examples
    pre_filter = len(new_examples)
    new_examples = [ex for ex in new_examples if len(ex["tokens"]) >= 2]
    if pre_filter != len(new_examples):
        print(f"Dropped {pre_filter - len(new_examples)} new records shorter than 2 tokens.")

    by_split = {"train": [], "validation": [], "test": []}
    for ex in new_examples:
        by_split[split_for(ex["tokens"])].append(ex)
    print(f"\nNew real examples: {len(new_examples)} total "
          f"({len(bc2gm_examples)} BC2GM + {len(bc4chemd_examples)} BC4CHEMD) -- "
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

    species_count = entity_counts.get("SPECIES", 0)
    if species_count:
        print(f"\nGENE:SPECIES ratio = {entity_counts.get('GENE', 0) / species_count:.2f}:1 "
              f"(v11 was 1.83:1, v12 was 3.79:1)")
        print(f"CHEMICAL:SPECIES ratio = {entity_counts.get('CHEMICAL', 0) / species_count:.2f}:1 "
              f"(v11 was 1.59:1, v12 was 5.30:1)")

    all_o_count = sum(1 for ex in total_by_split["train"] if all(t == 0 for t in ex["ner_tags"]))
    print(f"\nFully-negative (all-O) records in train split: {all_o_count} "
          f"({100*all_o_count/len(total_by_split['train']):.1f}% of train)")

if __name__ == "__main__":
    main()
