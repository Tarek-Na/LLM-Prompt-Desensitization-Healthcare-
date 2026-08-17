"""
Diagnostic: why does d4data/biomedical-ner-all's DATE row have such a large strict-vs-relaxed
gap (32.60% strict F1 vs 76.86% relaxed F1)? That's almost exactly the shape Apollo's DATE
row had before its SentencePiece leading-space bug was found and fixed, so this checks the
same thing for d4data (DistilBERT/WordPiece) rather than assuming WordPiece makes it immune.

For every gold DATE span in a sample of real records, this finds predictions that overlap it
(relaxed match) but don't exact-match it (strict miss), and prints the gold text, the
predicted text, and the start/end character deltas between them. A consistent delta pattern
(e.g. always start_delta=-1) points at an offset bug like Apollo's; a scattered/inconsistent
pattern with predicted text that's a genuinely different (shorter/longer/reworded) span
points at real model behavior instead.

Dependencies: transformers, torch.
"""
import json
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
SAMPLE_SIZE = int(os.environ.get("SOTA_EVAL_SAMPLE_SIZE", "3000") or 3000)
MAX_EXAMPLES_PRINTED = 40
SEED = 0

MACCROBAT_TO_SCHEMA = {
    "Age": "AGE", "Date": "DATE", "Disease_disorder": "DISEASE", "Occupation": "PROFESSION",
}


def _normalize(name):
    return name.replace("_", "").replace(" ", "").replace("-", "").lower()


_MACCROBAT_TO_SCHEMA_NORM = {_normalize(k): v for k, v in MACCROBAT_TO_SCHEMA.items()}

MODEL_NAME = "d4data/biomedical-ner-all"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label
print(f"Ready. Running on {device}.\n")


def predict_spans(text):
    # Identical merge logic to sota_eval_biomedical_ner_all.py. This has to match the real
    # scoring script exactly, or any pattern found here wouldn't explain the real numbers.
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512,
                     return_tensors="pt")
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()
    labels = [id2label[i] for i in pred_ids]

    entities = []
    cur = None
    for i, label in enumerate(labels):
        cs, ce = offset_mapping[i]
        if cs == ce:
            continue
        if label == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        etype = label.split("-", 1)[1] if "-" in label else label
        schema_type = _MACCROBAT_TO_SCHEMA_NORM.get(_normalize(etype), f"D4DATA_{etype}")
        if cur and cur["type"] == schema_type:
            cur["end"] = ce
        else:
            if cur:
                entities.append(cur)
            while cs < ce and text[cs].isspace():
                cs += 1
            cur = {"start": cs, "end": ce, "type": schema_type}
    if cur:
        entities.append(cur)
    return [{"type": e["type"], "start": e["start"], "end": e["end"],
              "text": text[e["start"]:e["end"]]} for e in entities]


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


records = list(iter_records(DATASET_PATH))
random.seed(SEED)
random.shuffle(records)
records = records[:SAMPLE_SIZE]
print(f"Sampled {len(records)} records.\n")

exact = 0
overlap_not_exact = []   # (gold, pred) pairs
missed_entirely = 0
delta_counts = {}

for rec in records:
    text = rec["text"]
    if not text.strip():
        continue
    gold_dates = [g for g in rec["entities"] if g["type"] == "DATE"]
    if not gold_dates:
        continue
    preds = predict_spans(text)
    pred_dates = [p for p in preds if p["type"] == "DATE"]

    for g in gold_dates:
        exact_match = any(p["start"] == g["start"] and p["end"] == g["end"] for p in pred_dates)
        if exact_match:
            exact += 1
            continue
        overlapping = [p for p in pred_dates if overlaps(g, p)]
        if not overlapping:
            missed_entirely += 1
            continue
        for p in overlapping:
            sd = p["start"] - g["start"]
            ed = p["end"] - g["end"]
            delta_counts[(sd, ed)] = delta_counts.get((sd, ed), 0) + 1
            if len(overlap_not_exact) < MAX_EXAMPLES_PRINTED:
                overlap_not_exact.append((g, p, sd, ed, text))

print(f"Gold DATE spans: exact match={exact}, overlap-but-not-exact={len(delta_counts) and sum(delta_counts.values())}, "
      f"missed entirely (no overlap at all)={missed_entirely}\n")

print("=== Most common (start_delta, end_delta) patterns among overlap-but-not-exact pairs ===")
for (sd, ed), c in sorted(delta_counts.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  start_delta={sd:+d}  end_delta={ed:+d}   count={c}")

print(f"\n=== Up to {MAX_EXAMPLES_PRINTED} concrete examples (gold vs predicted) ===")
for g, p, sd, ed, text in overlap_not_exact:
    ctx_start = max(0, g["start"] - 20)
    ctx_end = min(len(text), g["end"] + 20)
    print(f"  gold={g['text']!r} (start={g['start']} end={g['end']})")
    print(f"  pred={p['text']!r} (start={p['start']} end={p['end']})  start_delta={sd:+d} end_delta={ed:+d}")
    print(f"  context: ...{text[ctx_start:ctx_end]!r}...")
    print()
