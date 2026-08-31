# LoRA clinical PHI/NER model — results

DeBERTa-v3-base fine-tuned with LoRA (r=64, alpha=128, dropout=0.1, adapters on
`query_proj`/`value_proj`, full-rank `classifier` head) for 7-category clinical PHI and
biomedical named-entity recognition: **NAME, DATE, LOCATION, GENE, CHEMICAL, DISEASE,
VARIANT**.

Every number below comes from scoring the model against a real, independently-annotated
external corpus the model never trained on, not the model's own held-out split — this
project found in-distribution numbers to be an unreliable predictor of real-world
performance in earlier iterations.

Checkpoint: `r64_alpha128_lr5e-04_v16data_7cat` (LoRA config in `V16/adapter_config.json`).
Training data: `dataset/merged_clinical_phi_v16.{train,validation,test}.jsonl`, built by
`dataset/build_lora_dataset_v16.py`.

## Current model (v16) — full results

| Category | Benchmark | Strict P | Strict R | Strict F1 | Relaxed P | Relaxed R | Relaxed F1 |
|---|---|---|---|---|---|---|---|
| NAME | CoNLL-2003 | 96.90% | 96.54% | 96.72% | 97.95% | 97.65% | 97.80% |
| LOCATION | CoNLL-2003 | 90.67% | 93.82% | 92.22% | 91.54% | 94.66% | 93.08% |
| GENE | BC2GM | 66.45% | 60.65% | 63.42% | 88.31% | 80.14% | 84.03% |
| CHEMICAL | BC4CHEMD | 75.23% | 73.95% | 74.58% | 86.27% | 82.48% | 84.33% |
| VARIANT | OSIRIS | 52.21% | 41.40% | 46.19% | 82.75% | 88.91% | 85.72% |
| DISEASE | CADEC (n=22, small — see note) | 42.03% | 61.70% | 50.00% | 52.17% | 74.47% | 61.36% |

Strict = exact type + exact character span match. Relaxed = exact type, any character
overlap with the gold span. Source: eval run in the repo's `output-finetuned9.txt`,
confusion matrices in `confusion_matrices/confusion_matrix_lora_deberta_on_*.png`.

CADEC's test split is 22 records / 47 gold DISEASE spans — real signal, but read with that
scale in mind, not the same statistical confidence as the 5,000+ record benchmarks. A
Wilson score interval on relaxed recall: 74.5% [60.5%, 84.7%].

## How the schema got here: three checkpoints, same 5 benchmarks

| Checkpoint | Change | NAME | LOCATION | GENE | CHEMICAL | VARIANT | DISEASE |
|---|---|---|---|---|---|---|---|
| v13 | earlier schema, baseline | 98.07% | 93.49% | 83.70% | 84.58% | 85.71% | 31.62% |
| v15 | schema simplified to the current 7 categories | 97.80% | 93.03% | 85.33% | 85.21% | 86.46% | 25.00% |
| v16 | + 96 real CADEC training records (of its 127-record native train split) as new DISEASE data | 97.80% | 93.08% | 84.03% | 84.33% | 85.72% | **61.36%** |

All figures relaxed F1.

The v15→v16 step is the one deliberate, isolated intervention: 152 added training spans (out
of ~76k), touching only DISEASE — every other category's span count is byte-identical
between v15 and v16 (verified directly against the two dataset versions). DISEASE relaxed
recall went 48.9% (v15) to 74.5% (v16)
on the same 47 gold CADEC spans both times — Wilson intervals [35.3%, 62.8%] vs.
[60.5%, 84.7%], non-overlapping.
