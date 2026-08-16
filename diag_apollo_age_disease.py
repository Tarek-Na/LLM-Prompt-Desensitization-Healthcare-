"""
Diagnostic: systematically checks Apollo's residual AGE/DISEASE strict-vs-relaxed gaps
(post SentencePiece-offset fix). Previously this was only spot-checked on a single AGE
example (gold "19" vs Apollo's "19 years old"). This pulls every overlap-but-not-exact
AGE/DISEASE pair from a real sample and classifies each one as:
    extension   -- predicted span is gold span plus extra text on one or both sides
    truncation  -- predicted span is a strict substring of gold
    shift       -- predicted span overlaps but isn't a clean superset/subset (real boundary
                   disagreement, not just "model included more/less context")
so the AGE finding can be confirmed (or not) as the dominant pattern rather than a one-off.

Self-contained for Colab:
    !pip install transformers torch
"""
import json
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
SAMPLE_SIZE = int(os.environ.get("SOTA_EVAL_SAMPLE_SIZE", "3000") or 3000)
MAX_EXAMPLES_PRINTED = 25
SEED = 0

MACCROBAT_TO_SCHEMA = {
    "Age": "AGE", "Date": "DATE", "Disease_disorder": "DISEASE", "Occupation": "PROFESSION",
}


def _normalize(name):
    return name.replace("_", "").replace(" ", "").replace("-", "").lower()


_MACCROBAT_TO_SCHEMA_NORM = {_normalize(k): v for k, v in MACCROBAT_TO_SCHEMA.items()}

MODEL_NAME = "Clinical-AI-Apollo/Medical-NER"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label
print(f"Ready. Running on {device}.\n")


def predict_spans(text):
    # Identical merge logic to sota_eval_clinical_ai_apollo.py, including the SentencePiece
    # leading-space trim -- has to match the real scoring script exactly.
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
        schema_type = _MACCROBAT_TO_SCHEMA_NORM.get(_normalize(etype), f"APOLLO_{etype}")
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


def classify(g, p):
    if p["start"] <= g["start"] and p["end"] >= g["end"]:
        return "extension"
    if p["start"] >= g["start"] and p["end"] <= g["end"]:
        return "truncation"
    return "shift"


records = list(iter_records(DATASET_PATH))
random.seed(SEED)
random.shuffle(records)
records = records[:SAMPLE_SIZE]
print(f"Sampled {len(records)} records.\n")

for target_type in ("AGE", "DISEASE"):
    exact = 0
    missed_entirely = 0
    mismatches = []
    category_counts = {"extension": 0, "truncation": 0, "shift": 0}

    for rec in records:
        text = rec["text"]
        if not text.strip():
            continue
        gold_spans = [g for g in rec["entities"] if g["type"] == target_type]
        if not gold_spans:
            continue
        preds = predict_spans(text)
        pred_spans = [p for p in preds if p["type"] == target_type]

        for g in gold_spans:
            if any(p["start"] == g["start"] and p["end"] == g["end"] for p in pred_spans):
                exact += 1
                continue
            overlapping = [p for p in pred_spans if overlaps(g, p)]
            if not overlapping:
                missed_entirely += 1
                continue
            for p in overlapping:
                cat = classify(g, p)
                category_counts[cat] += 1
                mismatches.append((g, p, cat))

    total_mismatches = sum(category_counts.values())
    print(f"=== {target_type} ===")
    print(f"exact={exact}  missed_entirely={missed_entirely}  overlap_not_exact={total_mismatches}")
    for cat, c in category_counts.items():
        pct = 100 * c / total_mismatches if total_mismatches else 0
        print(f"  {cat:<11} {c}  ({pct:.1f}% of mismatches)")
    print(f"\n-- up to {MAX_EXAMPLES_PRINTED} examples --")
    for g, p, cat in mismatches[:MAX_EXAMPLES_PRINTED]:
        print(f"  [{cat}] gold={g['text']!r} (start={g['start']} end={g['end']})  "
              f"pred={p['text']!r} (start={p['start']} end={p['end']})")
    print()
