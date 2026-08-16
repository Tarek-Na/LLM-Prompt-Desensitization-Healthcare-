"""
Diagnostic: why does Clinical-AI-Apollo/Medical-NER get near-zero STRICT matches on DATE
(23/11318) and AGE (0/3288) while RELAXED recall is 96%/90%? Dumps gold spans next to
predicted spans character-for-character, plus the raw offset_mapping for the first few
tokens of each entity, to see whether this is a fixed offset error or something else.

Dependencies: transformers, torch.
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
MAX_EXAMPLES = int(os.environ.get("SOTA_EVAL_MAX_RECORDS", "20") or 20)

MACCROBAT_TO_SCHEMA = {"Age": "AGE", "Date": "DATE", "Disease_disorder": "DISEASE", "Occupation": "PROFESSION"}


def _normalize(name):
    return name.replace("_", "").replace(" ", "").replace("-", "").lower()


_NORM = {_normalize(k): v for k, v in MACCROBAT_TO_SCHEMA.items()}

MODEL_NAME = "Clinical-AI-Apollo/Medical-NER"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label
print(f"Ready. Running on {device}. Tokenizer class: {type(tokenizer).__name__}\n")


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
    raw_tokens = []  # (label, cs, ce, token_str) for debugging
    input_ids = enc["input_ids"][0].tolist()
    for i, label in enumerate(labels):
        cs, ce = offset_mapping[i]
        tok_str = tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
        raw_tokens.append((label, cs, ce, tok_str))
        if cs == ce:
            continue
        if label == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        etype = label.split("-", 1)[1] if "-" in label else label
        schema_type = _NORM.get(_normalize(etype), f"APOLLO_{etype}")
        if cur and cur["type"] == schema_type:
            cur["end"] = ce
        else:
            if cur:
                entities.append(cur)
            cur = {"start": cs, "end": ce, "type": schema_type}
    if cur:
        entities.append(cur)
    spans = [{"type": e["type"], "start": e["start"], "end": e["end"],
              "text": text[e["start"]:e["end"]]} for e in entities]
    return spans, raw_tokens


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


shown = 0
raw_dumped = False
for rec in iter_records(DATASET_PATH):
    if shown >= MAX_EXAMPLES:
        break
    text = rec["text"]
    gold_targets = [g for g in rec["entities"] if g["type"] in ("DATE", "AGE")]
    if not gold_targets:
        continue

    preds, raw_tokens = predict_spans(text)
    pred_targets = [p for p in preds if p["type"] in ("DATE", "AGE")]

    if not raw_dumped:
        print("=== RAW TOKEN DUMP (first record with a DATE/AGE gold entity) ===")
        print(f"text: {text!r}")
        for label, cs, ce, tok in raw_tokens[:40]:
            marker = f"[{cs}:{ce}]={text[cs:ce]!r}" if cs != ce else "(special)"
            print(f"  {label:<20} {marker:<30} tok={tok!r}")
        print()
        raw_dumped = True

    for g in gold_targets:
        overlapping = [p for p in pred_targets if p["type"] == g["type"] and overlaps(p, g)]
        exact = any(p["start"] == g["start"] and p["end"] == g["end"] for p in overlapping)
        shown += 1
        print(f"--- example {shown} [{g['type']}] {'EXACT' if exact else ('OVERLAP-ONLY' if overlapping else 'NO OVERLAP')} ---")
        print(f"  gold : [{g['start']:>4}:{g['end']:>4}] {g['text']!r}")
        for p in overlapping:
            delta_s = p['start'] - g['start']
            delta_e = p['end'] - g['end']
            print(f"  pred : [{p['start']:>4}:{p['end']:>4}] {p['text']!r}  (start_delta={delta_s:+d} end_delta={delta_e:+d})")
        if not overlapping:
            print("  pred : (none)")
        print()
        if shown >= MAX_EXAMPLES:
            break

print(f"\nShowed {shown} DATE/AGE examples.")
