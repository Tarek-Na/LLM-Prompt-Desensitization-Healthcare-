"""
Builds merged_clinical_phi_v15.{train,validation,test}.jsonl on top of v13 (not v14 --
v14's synthetic gene-vs-species contrastive fix for SPECIES barely moved the needle and
made GENE's spurious firing on Species-800 slightly worse, so v13 stayed the recommended
checkpoint and v14 is not built on further).

This version drops SPECIES and CELL from the schema entirely, 9 categories down to 7.
Both were the weakest categories on real external benchmarks (SPECIES 37.02% relaxed F1
on Species-800, CELL 42.04% on CellFinder) with diagnosed, not-easily-fixable causes: v12
showed adding more real GENE/CHEMICAL data actively regresses SPECIES via class-imbalance
interference, and CELL-line names are arbitrary lab-assigned codes with no shared grammar
to generalize from. Rather than keep chasing these with more data, we're removing them so
the model isn't spending capacity and label-imbalance budget on two categories it can't
reliably do, which should let it commit fully to the 7 categories that already work.

Every v13 record that carries a B-SPECIES/I-SPECIES/B-CELL/I-CELL tag is KEPT, not dropped
-- only those specific tags are relabeled to O. Checked directly against the v13 files
before writing this: SPECIES and CELL tags overwhelmingly co-occur in the same sentence
with other, still-relevant entity types (58.7% of SPECIES-tagged records and 82.3% of
CELL-tagged records also contain a GENE/CHEMICAL/DISEASE/etc. span), so dropping whole
records would throw away real GENE/CHEMICAL/DISEASE training signal along with the tags
we actually want gone. The remainder (pure SPECIES-only or CELL-only sentences, mostly from
LINNAEUS and BioNLP2004) still has value once relabeled to all-O: they're real biological-
text negatives that teach the model not to fire GENE/CHEMICAL on species and cell-line
vocabulary, the same kind of hard-negative mining already used elsewhere in this pipeline.
"""
import os
import json
import hashlib
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
VERSION = "v15"

BASE_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, "merged_clinical_phi_v13.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, "merged_clinical_phi_v13.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, "merged_clinical_phi_v13.test.jsonl"),
}
OUTPUT_SPLIT_PATHS = {
    "train": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.train.jsonl"),
    "validation": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.validation.jsonl"),
    "test": os.path.join(REPO_ROOT, f"merged_clinical_phi_{VERSION}.test.jsonl"),
}
LABELS_PATH = os.path.join(REPO_ROOT, f"labels_{VERSION}.json")

# v13's schema, needed to decode the label ids already baked into the v13 files.
OLD_LABELS = [
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

# The new 7-category schema. SPECIES and CELL are gone, everything else keeps its name so
# any tag not found here (i.e. every B-SPECIES/I-SPECIES/B-CELL/I-CELL) falls back to O.
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
O_ID = master_label2id["O"]

# old_id -> new_id, by label name, not position (the two schemas don't share positions).
REMAP = [master_label2id.get(name, O_ID) for name in OLD_LABELS]


def remap_tags(old_tags):
    return [REMAP[t] for t in old_tags]


def entity_counts(records, labels):
    counts = collections.Counter()
    for rec in records:
        for t in rec["ner_tags"]:
            label = labels[t]
            if label.startswith("B-"):
                counts[label[2:]] += 1
    return counts


def main():
    print(f"Building merged_clinical_phi_{VERSION} -- same v13 records, SPECIES and CELL "
          f"dropped from the schema and relabeled to O wherever they occurred. No records "
          f"are removed for containing those tags; only exact-duplicate records within a "
          f"split are dropped as a light overfitting guard.\n")

    total_by_split = {}
    for split_name, path in BASE_SPLIT_PATHS.items():
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)

        before_all_o = sum(1 for r in records if all(t == 0 for t in r["ner_tags"]))
        old_counts = entity_counts(records, OLD_LABELS)

        seen = set()
        deduped = []
        for rec in records:
            key = " ".join(rec["tokens"])
            if key in seen:
                continue
            seen.add(key)
            new_tags = remap_tags(rec["ner_tags"])
            deduped.append({
                "tokens": rec["tokens"],
                "ner_tags": new_tags,
                "source": rec.get("source", "UNKNOWN"),
            })

        after_all_o = sum(1 for r in deduped if all(t == 0 for t in r["ner_tags"]))
        new_counts = entity_counts(deduped, master_labels)
        dropped_dupes = len(records) - len(deduped)

        total_by_split[split_name] = deduped
        print(f"{split_name:12s}: {len(records):6d} v13 records -> "
              f"{dropped_dupes} exact within-split duplicates dropped -> "
              f"{len(deduped):6d} final records")
        print(f"{'':14s}all-O share: {100*before_all_o/len(records):.1f}% (v13, 9-cat) -> "
              f"{100*after_all_o/len(deduped):.1f}% ({VERSION}, 7-cat)")
        print(f"{'':14s}dropped from schema -- SPECIES: {old_counts.get('SPECIES', 0)} spans, "
              f"CELL: {old_counts.get('CELL', 0)} spans")
        print(f"{'':14s}kept: " + ", ".join(f"{k}={new_counts.get(k, 0)}" for k in
              ("NAME", "DATE", "LOCATION", "GENE", "CHEMICAL", "DISEASE", "VARIANT")))
        print()

    print("Checking for cross-split leakage (identical token sequence in more than one "
          "split)...")
    seq_sets = {name: {" ".join(r["tokens"]) for r in recs}
                for name, recs in total_by_split.items()}
    leak_pairs = [
        ("train", "validation"), ("train", "test"), ("validation", "test"),
    ]
    any_leak = False
    for a, b in leak_pairs:
        overlap = seq_sets[a] & seq_sets[b]
        if overlap:
            any_leak = True
            print(f"  LEAK: {len(overlap)} identical records shared between {a} and {b}!")
        else:
            print(f"  {a} / {b}: no overlap.")
    if any_leak:
        raise SystemExit(
            "Refusing to write output: cross-split leakage detected above. This should be "
            "impossible coming from v13's already hash-partitioned files, so something "
            "upstream changed -- investigate before training on this data."
        )

    print("\nWriting output files...")
    for split_name, recs in total_by_split.items():
        with open(OUTPUT_SPLIT_PATHS[split_name], "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"  {split_name:12s}: {len(recs):6d} records -> "
              f"{os.path.basename(OUTPUT_SPLIT_PATHS[split_name])}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "labels": master_labels,
            "label2id": master_label2id,
            "id2label": {str(i): l for i, l in enumerate(master_labels)},
        }, f, indent=2)
    print(f"\nLabel schema (7-category, SPECIES and CELL removed) written to "
          f"{os.path.basename(LABELS_PATH)}")


if __name__ == "__main__":
    main()
