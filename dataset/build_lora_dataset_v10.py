"""
Builds merged_clinical_phi_v10.{train,validation,test}.jsonl, a new version of the LoRA
model's training data targeting two specific weaknesses found by scoring the v9-trained
model against four real, independent benchmarks (BC2GM, BC4CHEMD, Species-800, CoNLL-2003):

1. NAME and LOCATION were trained almost entirely on synthetic, closed-vocabulary
   templates (39 first names, 35 last names, 12 sentence templates, ~20 cities) with only
   a tiny sliver of real i2b2 text. That produced a model that scores perfectly on its own
   held-out test set (drawn from the same closed vocabulary) but collapses on real CoNLL-
   2003 text (relaxed F1 30% for NAME, 10% for LOCATION). This script adds CoNLL-2003's
   TRAIN split (never its validation/test, which stay held out for lora_eval_conll2003.py)
   as a genuine, diverse real-text NAME/LOCATION source.

2. Every category showed heavy spurious firing on real text with no matching entities at
   all (e.g. ~1.4 bogus non-GENE predictions per sentence on BC4CHEMD, CHEMICAL/DATE/
   SPECIES/DISEASE predictions on plain CoNLL-2003 newswire that contains none of those).
   This script mines genuine all-O (no-entity) sentences from the TRAIN splits of four
   real, diverse corpora as true-negative examples, so the model gets direct signal that
   not every sentence contains one of its trained categories.

Deliberately dependency-free like build_dataset.py/Calldataset.py: only the Python
standard library is used, data is fetched via the HF datasets-server rows API.

This script does NOT touch build_dataset.py or benchmark_clinical_phi.jsonl -- the 30-
category benchmark already used to score the 7 SOTA models stays exactly as published.
It reads the CURRENT merged_clinical_phi.*.jsonl (which does carry the full 30-category
schema now that build_dataset.py has grown past the LoRA model's original 9-category
scope) and remaps anything outside the 9-category schema to O, so the output is directly
compatible with LoRa-Code.py's existing master_labels list -- no code changes needed
there beyond pointing at the new file names.
"""
import os
import json
import random
import re
import time
import hashlib
import urllib.request
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
REQUEST_TIMEOUT = 60
VERSION = "v10"

EXISTING_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, "merged_clinical_phi.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, "merged_clinical_phi.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, "merged_clinical_phi.test.jsonl"),
}
OUTPUT_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.test.jsonl"),
}
LABELS_PATH = os.path.join(REPO_ROOT, f"labels_{VERSION}.json")

# Negative-example volume per source, capped so the added negatives stay a meaningful
# minority of the final dataset rather than swamping the positive-example signal.
NEGATIVES_PER_SOURCE = int(os.environ.get("PHI_NEGATIVES_PER_SOURCE", "1500"))

# =====================================================================
# 9-category schema, exactly matching LoRa-Code.py's hardcoded master_labels list --
# this is the compatibility contract the whole script exists to preserve.
# =====================================================================
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
IN_SCHEMA_TYPES = {"DISEASE", "CHEMICAL", "GENE", "CELL", "SPECIES", "VARIANT",
                    "NAME", "DATE", "LOCATION"}

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

def fetch_rows_via_datasets_server(dataset, config, split, page_size=100, max_rows=0):
    try:
        total_rows = get_split_row_count(dataset, config, split)
    except Exception as e:
        print(f"Warning: could not get row count for {dataset}/{config}: {e}")
        return []
    if total_rows == 0:
        print(f"Warning: {dataset}/{config}/{split} reports 0 rows.")
        return []

    limit = total_rows if max_rows <= 0 else min(total_rows, max_rows)
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
        time.sleep(0.3)
    print(f"Loaded {dataset}/{config}/{split}: {len(rows)} rows")
    return rows

# =====================================================================
# Step 1: load the existing (now 30-category) merged_clinical_phi.*.jsonl and remap
# anything outside the 9-category schema to O. This is what makes the current file --
# already the LoRA model's real, working training data for GENE/CHEMICAL/DISEASE/CELL/
# SPECIES/VARIANT and even most of NAME/DATE/LOCATION -- safe to reuse without a crash.
# =====================================================================
def load_and_remap_existing(path):
    labels_v9 = json.load(open(os.path.join(REPO_ROOT, "labels_v9.json"), encoding="utf-8"))["labels"]
    out = []
    remapped_tokens = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new_tags = []
            for t in rec["ner_tags"]:
                label = labels_v9[t] if 0 <= t < len(labels_v9) else "O"
                ent_type = label[2:] if label != "O" else None
                if ent_type is not None and ent_type in IN_SCHEMA_TYPES:
                    new_tags.append(master_label2id[label])
                else:
                    if ent_type is not None:
                        remapped_tokens += 1
                    new_tags.append(master_label2id["O"])
            out.append({"tokens": rec["tokens"], "ner_tags": new_tags, "source": rec.get("source", "?")})
    print(f"Loaded {len(out)} existing records from {os.path.basename(path)}, "
          f"remapped {remapped_tokens} out-of-schema PII tokens (SSN/EMAIL/PHONE/etc.) to O.")
    return out

# =====================================================================
# Step 2: real CoNLL-2003 train split -- PER->NAME, LOC->LOCATION, ORG/MISC dropped to O.
# Verified directly against this mirror's own data (not assumed): index 3 labels "EU" and
# index 7 labels "German"/"British" in the famous first training sentence, confirming
# B-ORG and B-MISC at those positions match the standard 9-class scheme used here.
# =====================================================================
CONLL_LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                      "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

def _conll_type_map(native_type):
    if native_type == "PER":
        return "NAME"
    if native_type == "LOC":
        return "LOCATION"
    return None

def fetch_conll2003_examples():
    rows = fetch_rows_via_datasets_server("lhoestq/conll2003", "default", "train")
    examples, n_with_entity, n_all_o = [], 0, 0
    for row in rows:
        tokens = row.get("tokens", [])
        if not tokens:
            continue
        tags = []
        has_entity = False
        for t in row.get("ner_tags", []):
            native = CONLL_LABEL_NAMES[t] if 0 <= t < len(CONLL_LABEL_NAMES) else "O"
            if native == "O":
                tags.append(master_label2id["O"])
                continue
            prefix, native_type = native[:2], native[2:]
            mapped = _conll_type_map(native_type)
            if mapped is None:
                tags.append(master_label2id["O"])
                continue
            has_entity = True
            tags.append(master_label2id[f"{prefix}{mapped}"])
        if has_entity:
            n_with_entity += 1
        else:
            n_all_o += 1
        examples.append({"tokens": tokens, "ner_tags": tags, "source": "conll2003_real"})
    print(f"CoNLL-2003 train: {n_with_entity} records with a real NAME/LOCATION mention, "
          f"{n_all_o} naturally entity-free records (both kept -- the entity-free ones "
          f"double as real-newswire negative examples).")
    return examples

# =====================================================================
# Step 3: hard-negative mining -- genuine all-O sentences from real, diverse train splits
# of the exact corpora used to evaluate the model (BC2GM, BC4CHEMD, Species-800), so the
# model sees real biomedical text that legitimately contains none of its trained
# categories. Capped per source via NEGATIVES_PER_SOURCE so negatives stay a minority.
# =====================================================================
def _mine_all_o_negatives(rows, token_field, tag_getter, source_name, cap, rng):
    negatives = []
    for row in rows:
        tokens = row.get(token_field, [])
        if not tokens:
            continue
        tags = tag_getter(row)
        if any(t != "O" for t in tags):
            continue
        negatives.append({"tokens": tokens, "ner_tags": [master_label2id["O"]] * len(tokens),
                           "source": f"{source_name}_negative"})
    if len(negatives) > cap:
        negatives = rng.sample(negatives, cap)
    print(f"{source_name}: mined {len(negatives)} all-O negative examples (capped at {cap}).")
    return negatives

def fetch_bc2gm_negatives(rng, cap):
    rows = fetch_rows_via_datasets_server("spyysalo/bc2gm_corpus", "bc2gm_corpus", "train")
    names = ["O", "B-GENE", "I-GENE"]
    return _mine_all_o_negatives(
        rows, "tokens", lambda r: [names[t] if 0 <= t < len(names) else "O" for t in r["ner_tags"]],
        "bc2gm", cap, rng)

def fetch_bc4chemd_negatives(rng, cap):
    rows = fetch_rows_via_datasets_server("disi-unibo-nlp/bc4chemd", "default", "train")
    return _mine_all_o_negatives(rows, "tokens", lambda r: r["ner_tags"], "bc4chemd", cap, rng)

def fetch_species800_negatives(rng, cap):
    rows = fetch_rows_via_datasets_server("marcov/species_800_promptsource", "default", "train")
    names = ["O", "B", "I"]
    seen_ids = set()
    deduped = []
    for r in rows:
        rid = r.get("id")
        if rid is not None:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
        deduped.append(r)
    return _mine_all_o_negatives(
        deduped, "tokens", lambda r: [names[t] if 0 <= t < len(names) else "O" for t in r["ner_tags"]],
        "species800", cap, rng)

# =====================================================================
# Deterministic hash-of-tokens split, same convention as build_dataset.py/Calldataset.py,
# so a re-run is reproducible and new examples land in a stable split.
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
    rng = random.Random(42)
    print("Building merged_clinical_phi_v10 -- targeted fix for NAME/LOCATION real-text "
          "generalization and cross-category spurious firing.\n")

    # Step 1: existing 9-category-compatible base (each existing split kept in its own file).
    base_by_split = {}
    for split_name, path in EXISTING_SPLIT_PATHS.items():
        base_by_split[split_name] = load_and_remap_existing(path)

    # Step 2: new real CoNLL-2003 NAME/LOCATION examples (train split only -- validation/
    # test of this same corpus are what lora_eval_conll2003.py scores against, so those
    # stay untouched and there is zero leakage between this new training data and that
    # eval).
    print()
    conll_examples = fetch_conll2003_examples()

    # Step 3: hard negatives from the exact three other corpora used for evaluation
    # (again, train splits only -- their own test splits stay held out for evaluation).
    print()
    negatives = []
    negatives += fetch_bc2gm_negatives(rng, NEGATIVES_PER_SOURCE)
    negatives += fetch_bc4chemd_negatives(rng, NEGATIVES_PER_SOURCE)
    negatives += fetch_species800_negatives(rng, NEGATIVES_PER_SOURCE)

    new_examples = conll_examples + negatives
    pre_filter = len(new_examples)
    new_examples = [ex for ex in new_examples if len(ex["tokens"]) >= 2]
    if pre_filter != len(new_examples):
        print(f"Dropped {pre_filter - len(new_examples)} new records shorter than 2 tokens.")

    by_split = {"train": [], "validation": [], "test": []}
    for ex in new_examples:
        by_split[split_for(ex["tokens"])].append(ex)

    print("\nWriting output files...")
    total_by_split = {}
    for split_name in ("train", "validation", "test"):
        merged = base_by_split[split_name] + by_split[split_name]
        rng.shuffle(merged)
        total_by_split[split_name] = merged
        with open(OUTPUT_SPLIT_PATHS[split_name], "w", encoding="utf-8") as f:
            for ex in merged:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name:12s}: {len(base_by_split[split_name]):6d} existing + "
              f"{len(by_split[split_name]):6d} new = {len(merged):6d} records "
              f"-> {os.path.basename(OUTPUT_SPLIT_PATHS[split_name])}")

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
