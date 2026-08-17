"""
Converts merged_clinical_phi.{train,validation,test}.jsonl (our internal tokens+BIO format)
into a model-agnostic benchmark format: raw text + character-offset gold entity spans.

This is the actual deliverable for testing SOTA PII/PHI models. Almost every model
(transformer-based or spaCy-based) takes raw text as input and does its own internal
tokenization, so shipping our own token boundaries would be useless to them. Any model can
be run on the `text` field and scored against `entities` regardless of its own tokenizer.

Output format, one JSON object per line:
    {"text": "...", "entities": [{"type": "NAME", "start": 8, "end": 16, "text": "John Doe"}, ...], "source": "synthetic_pii"}
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from sota_eval_common import detokenize_with_offsets, extract_gold_spans

master_labels = json.load(open("labels.json", encoding="utf-8"))["labels"]

SOURCE_SPLITS = [
    ("train", "merged_clinical_phi.train.jsonl"),
    ("validation", "merged_clinical_phi.validation.jsonl"),
    ("test", "merged_clinical_phi.test.jsonl"),
]

token_offset_checks = 0
token_offset_mismatches = 0
mismatch_examples = []


def convert(path):
    global token_offset_checks, token_offset_mismatches
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tokens, tags = rec["tokens"], rec["ner_tags"]
            text, offsets = detokenize_with_offsets(tokens)

            # Correctness check on the actual risk: does every token's own offset slice
            # back out to that exact token? If this holds for every token in every record,
            # the offset bookkeeping is provably correct, and every entity span built from
            # consecutive token offsets (start of first token, end of last token) is
            # therefore correct too. Checking the entity span's own text against
            # text[start:end] would be circular, since extract_gold_spans derives "text"
            # from that exact slice by construction.
            for tok, (s, e) in zip(tokens, offsets):
                token_offset_checks += 1
                if text[s:e] != tok:
                    token_offset_mismatches += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append((tok, text[s:e], text))

            spans = extract_gold_spans(tokens, tags, master_labels, text, offsets)
            records.append({
                "text": text,
                "entities": [{"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]}
                             for s in spans],
                "source": rec.get("source", "unknown"),
            })
    return records


all_records = []
for split_name, path in SOURCE_SPLITS:
    print(f"Converting {path} ...")
    records = convert(path)
    out_path = f"benchmark_clinical_phi.{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> {out_path} ({len(records)} records)")
    all_records.extend(records)

combined_path = "benchmark_clinical_phi.jsonl"
with open(combined_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"\nCombined: {combined_path} ({len(all_records)} records)")

total_entities = sum(len(r["entities"]) for r in all_records)
print(f"\nToken-offset correctness check: {token_offset_checks} tokens checked, "
      f"{token_offset_mismatches} mismatches.")
if token_offset_mismatches:
    print("WARNING: offset bookkeeping is broken. Do NOT use this export until fixed.")
    for tok, got, text in mismatch_examples:
        print(f"  expected {tok!r}, got {got!r}  in: {text[:100]!r}")
else:
    print("All token offsets verified correct. Every entity span's character "
          "boundaries are provably exact.")
print(f"Total entity spans across combined benchmark: {total_entities}")
