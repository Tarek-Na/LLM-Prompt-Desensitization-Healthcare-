"""
Validation harness for merged_clinical_phi.jsonl

Checks (stdlib only):
  1. JSON well-formedness per line
  2. tokens/ner_tags length match
  3. tag ids within valid label range
  4. BIO well-formedness (no I-X immediately after O or after a different type,
     i.e. every I-X must be preceded by B-X or I-X)
  5. label distribution (counts of B- tags per entity type = entity counts)
  6. sample sentences per entity type for manual eyeballing
  7. flags empty token lists, tokens/tags mismatch
"""
import json
import sys
import random
from collections import Counter, defaultdict

MASTER_LABELS = [
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
    "B-AGE", "I-AGE",
    "B-PHONE", "I-PHONE",
    "B-FAX", "I-FAX",
    "B-EMAIL", "I-EMAIL",
    "B-SSN", "I-SSN",
    "B-MEDICAL_RECORD_NUMBER", "I-MEDICAL_RECORD_NUMBER",
    "B-HEALTH_PLAN_ID", "I-HEALTH_PLAN_ID",
    "B-ACCOUNT_NUMBER", "I-ACCOUNT_NUMBER",
    "B-LICENSE_NUMBER", "I-LICENSE_NUMBER",
    "B-VEHICLE_ID", "I-VEHICLE_ID",
    "B-DEVICE_ID", "I-DEVICE_ID",
    "B-URL", "I-URL",
    "B-IP_ADDRESS", "I-IP_ADDRESS",
    "B-BIOMETRIC_ID", "I-BIOMETRIC_ID",
    "B-OTHER_ID", "I-OTHER_ID",
    "B-PROFESSION", "I-PROFESSION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-USERNAME", "I-USERNAME",
    "B-PASSPORT_NUMBER", "I-PASSPORT_NUMBER",
    "B-CRYPTO_WALLET", "I-CRYPTO_WALLET",
]
NUM_LABELS = len(MASTER_LABELS)

PATH = sys.argv[1] if len(sys.argv) > 1 else "merged_clinical_phi.jsonl"

total = 0
bad_json = 0
len_mismatch = 0
bad_tag_id = 0
bio_violations = 0
empty_tokens = 0

label_token_counts = Counter()
entity_counts = Counter()
examples_per_type = defaultdict(list)
bio_violation_examples = []

random.seed(42)

with open(PATH, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            bad_json += 1
            if bad_json <= 5:
                print(f"[BAD JSON] line {line_no}: {e}")
            continue

        tokens = rec.get("tokens", [])
        tags = rec.get("ner_tags", [])

        if not tokens:
            empty_tokens += 1

        if len(tokens) != len(tags):
            len_mismatch += 1
            if len_mismatch <= 5:
                print(f"[LEN MISMATCH] line {line_no}: {len(tokens)} tokens vs {len(tags)} tags")
            continue

        prev_label = "O"
        for tok, tag_id in zip(tokens, tags):
            if not isinstance(tag_id, int) or tag_id < 0 or tag_id >= NUM_LABELS:
                bad_tag_id += 1
                prev_label = "O"
                continue

            label = MASTER_LABELS[tag_id]
            label_token_counts[label] += 1

            if label.startswith("I-"):
                ent_type = label[2:]
                prev_ok = prev_label == f"B-{ent_type}" or prev_label == f"I-{ent_type}"
                if not prev_ok:
                    bio_violations += 1
                    if len(bio_violation_examples) < 10:
                        bio_violation_examples.append((line_no, tokens, tags))

            if label.startswith("B-"):
                ent_type = label[2:]
                entity_counts[ent_type] += 1
                if len(examples_per_type[ent_type]) < 30:
                    examples_per_type[ent_type].append((tokens, tags))

            prev_label = label

print("=" * 70)
print(f"Total records:            {total}")
print(f"Bad JSON lines:           {bad_json}")
print(f"Token/tag length mismatch:{len_mismatch}")
print(f"Out-of-range tag ids:     {bad_tag_id}")
print(f"Empty token lists:        {empty_tokens}")
print(f"BIO sequence violations:  {bio_violations}")
print("=" * 70)

print("\nEntity span counts (B- starts) by type:")
for ent_type, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
    print(f"  {ent_type:10s}: {cnt}")

print("\nRaw label token frequency:")
for label in MASTER_LABELS:
    print(f"  {label:14s}: {label_token_counts.get(label, 0)}")

if bio_violation_examples:
    print("\nSample BIO violations:")
    for line_no, tokens, tags in bio_violation_examples[:5]:
        labels = [MASTER_LABELS[t] if 0 <= t < NUM_LABELS else "UNK" for t in tags]
        print(f"  line {line_no}:")
        print(f"    tokens: {tokens}")
        print(f"    labels: {labels}")

print("\nSample sentences per entity type (up to 3 each):")
for ent_type in ["DISEASE", "CHEMICAL", "GENE", "CELL", "SPECIES", "VARIANT", "NAME", "DATE", "LOCATION"]:
    samples = examples_per_type.get(ent_type, [])
    print(f"\n-- {ent_type} ({len(samples)} sampled, {entity_counts.get(ent_type,0)} total) --")
    for tokens, tags in random.sample(samples, min(3, len(samples))):
        labels = [MASTER_LABELS[t] if 0 <= t < NUM_LABELS else "UNK" for t in tags]
        spans = []
        cur, cur_type = None, None
        for i, lab in enumerate(labels):
            if lab == "O":
                if cur is not None:
                    spans.append((" ".join(tokens[cur:i]), cur_type))
                    cur = None
            elif lab.startswith("B-"):
                if cur is not None:
                    spans.append((" ".join(tokens[cur:i]), cur_type))
                cur, cur_type = i, lab[2:]
            elif lab.startswith("I-") and cur is None:
                cur, cur_type = i, lab[2:]
        if cur is not None:
            spans.append((" ".join(tokens[cur:]), cur_type))
        print(f"   sent: {' '.join(tokens)[:150]}")
        print(f"   spans: {spans}")
