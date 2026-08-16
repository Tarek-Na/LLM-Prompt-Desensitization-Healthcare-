"""
Diagnostic: why does obi/deid_roberta_i2b2 get 0/3,204 EXACT EMAIL matches (strict) despite
2,893/3,204 overlapping matches (relaxed)? Dumps gold EMAIL spans next to whatever obi
actually predicted for them, character-for-character, so the boundary difference is visible
directly instead of guessed at.

Dependencies: transformers, torch.
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
MAX_EXAMPLES = int(os.environ.get("SOTA_EVAL_MAX_RECORDS", "25") or 25)

OBI_TO_SCHEMA = {
    "PATIENT": "NAME", "STAFF": "NAME", "PATORG": "ORGANIZATION", "HOSP": "LOCATION",
    "LOC": "LOCATION", "AGE": "AGE", "DATE": "DATE", "EMAIL": "EMAIL", "PHONE": "PHONE",
    "ID": "OTHER_ID",
}

MODEL_NAME = "obi/deid_roberta_i2b2"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label
print(f"Ready. Running on {device}.\n")


def predict_spans(text):
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
        schema_type = OBI_TO_SCHEMA.get(label.split("-", 1)[1], f"OBI_{label}")
        if cur and cur["type"] == schema_type:
            cur["end"] = ce
        else:
            if cur:
                entities.append(cur)
            cur = {"start": cs, "end": ce, "type": schema_type}
    if cur:
        entities.append(cur)
    return [{"type": e["type"], "start": e["start"], "end": e["end"],
              "text": text[e["start"]:e["end"]]} for e in entities]


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


shown = 0
for rec in iter_records(DATASET_PATH):
    if shown >= MAX_EXAMPLES:
        break
    text = rec["text"]
    gold_emails = [g for g in rec["entities"] if g["type"] == "EMAIL"]
    if not gold_emails:
        continue

    preds = predict_spans(text)
    pred_emails = [p for p in preds if p["type"] == "EMAIL"]

    for g in gold_emails:
        overlapping = [p for p in pred_emails if overlaps(p, g)]
        exact = any(p["start"] == g["start"] and p["end"] == g["end"] for p in overlapping)
        shown += 1
        print(f"--- example {shown} {'[EXACT MATCH]' if exact else '[BOUNDARY MISMATCH]' if overlapping else '[NO OVERLAP AT ALL]'} ---")
        print(f"  gold : [{g['start']:>4}:{g['end']:>4}] {g['text']!r}")
        if overlapping:
            for p in overlapping:
                print(f"  pred : [{p['start']:>4}:{p['end']:>4}] {p['text']!r}")
        else:
            print("  pred : (no EMAIL-typed prediction overlapped this gold span)")
        # Show a bit of surrounding context so we can see what's right before/after the
        # email in the sentence -- useful if the mismatch is a punctuation/whitespace issue.
        ctx_start = max(0, g["start"] - 15)
        ctx_end = min(len(text), g["end"] + 15)
        print(f"  ctx  : ...{text[ctx_start:g['start']]}[{text[g['start']:g['end']]}]{text[g['end']:ctx_end]}...")
        print()
        if shown >= MAX_EXAMPLES:
            break

print(f"\nShowed {shown} EMAIL examples.")
