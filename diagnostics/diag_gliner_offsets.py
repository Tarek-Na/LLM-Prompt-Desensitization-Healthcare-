"""
Diagnostic: GLiNER-biomed showed 0% STRICT match on URL (tp=0/1322) and BIOMETRIC_ID
(tp=0/1309) while RELAXED recall was 100% on both in the full 61,288-record run.

FIRST ATTEMPT at this diagnostic prompted GLiNER with only ["url", "biometric identifier"]
and got a contradictory result: BIOMETRIC_ID's gap turned out to be real (GLiNER correctly
skips the descriptive prefix, e.g. gold "voiceprint ID VP-8629" vs GLiNER's "VP-8629",
a genuine annotation-scope difference, not a bug), but URL scored a perfect 12/12 exact
matches in that test, flatly contradicting the full run's tp=0/1322. The two runs weren't
comparable: the real benchmark script prompts GLiNER with ALL 30 schema labels
simultaneously, not just 2, and GLiNER's predicted boundaries can plausibly shift depending
on what else is in the label set. This version uses the EXACT SAME 30-label list as
sota_eval_gliner_biomed.py so the comparison is actually apples-to-apples.

Dependencies: gliner.
"""
import json
import os
from gliner import GLiNER

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
MAX_EXAMPLES = int(os.environ.get("SOTA_EVAL_MAX_RECORDS", "20") or 20)
SCORE_THRESHOLD = float(os.environ.get("SOTA_EVAL_SCORE_THRESHOLD", "0.5"))

# Identical to GLINER_LABEL_TO_SCHEMA in sota_eval_gliner_biomed.py. Must match exactly,
# since the whole point is reproducing the real script's exact prompt conditions.
GLINER_LABEL_TO_SCHEMA = {
    "person name": "NAME", "date": "DATE", "location": "LOCATION", "age": "AGE",
    "phone number": "PHONE", "fax number": "FAX", "email address": "EMAIL",
    "social security number": "SSN", "medical record number": "MEDICAL_RECORD_NUMBER",
    "health plan number": "HEALTH_PLAN_ID", "account number": "ACCOUNT_NUMBER",
    "license number": "LICENSE_NUMBER", "vehicle identifier": "VEHICLE_ID",
    "device identifier": "DEVICE_ID", "url": "URL", "ip address": "IP_ADDRESS",
    "biometric identifier": "BIOMETRIC_ID", "identification number": "OTHER_ID",
    "profession": "PROFESSION", "organization": "ORGANIZATION",
    "credit card number": "CREDIT_CARD", "username": "USERNAME",
    "passport number": "PASSPORT_NUMBER",
    "cryptocurrency wallet address": "CRYPTO_WALLET", "disease": "DISEASE",
    "chemical": "CHEMICAL", "gene": "GENE", "cell line": "CELL", "species": "SPECIES",
    "genetic variant": "VARIANT",
}
GLINER_LABELS = list(GLINER_LABEL_TO_SCHEMA.keys())

MODEL_NAME = "Ihor/gliner-biomed-base-v1.0"
print(f"Loading {MODEL_NAME}...")
model = GLiNER.from_pretrained(MODEL_NAME)
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Ready. Running on {device}.\n")
except Exception as e:
    print(f"Ready. (device placement skipped: {e})\n")


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
    gold_targets = [g for g in rec["entities"] if g["type"] in ("URL", "BIOMETRIC_ID")]
    if not gold_targets:
        continue

    ents = model.predict_entities(text, GLINER_LABELS, threshold=SCORE_THRESHOLD)
    preds = [
        {"type": GLINER_LABEL_TO_SCHEMA.get(e["label"].lower(), f"GLINER_{e['label']}"),
         "start": e["start"], "end": e["end"], "text": e["text"], "score": e.get("score")}
        for e in ents
    ]

    for g in gold_targets:
        overlapping = [p for p in preds if p["type"] == g["type"] and overlaps(p, g)]
        exact = any(p["start"] == g["start"] and p["end"] == g["end"] for p in overlapping)
        shown += 1
        print(f"--- example {shown} [{g['type']}] {'EXACT' if exact else ('OVERLAP-ONLY' if overlapping else 'NO OVERLAP')} ---")
        print(f"  gold : [{g['start']:>4}:{g['end']:>4}] {g['text']!r}")
        for p in overlapping:
            delta_s = p['start'] - g['start']
            delta_e = p['end'] - g['end']
            print(f"  pred : [{p['start']:>4}:{p['end']:>4}] {p['text']!r}  score={p['score']:.3f}  (start_delta={delta_s:+d} end_delta={delta_e:+d})")
        if not overlapping:
            print("  pred : (none)")
        print()
        if shown >= MAX_EXAMPLES:
            break

print(f"\nShowed {shown} URL/BIOMETRIC_ID examples.")
