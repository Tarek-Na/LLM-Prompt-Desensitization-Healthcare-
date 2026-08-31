# Clinical PHI/NER LoRA model

DeBERTa-v3-base fine-tuned with LoRA for 7-category clinical PHI and biomedical named-entity
recognition: **NAME, DATE, LOCATION, GENE, CHEMICAL, DISEASE, VARIANT**.

Every reported number comes from scoring the model against a real, independently-annotated
external corpus it never trained on, not its own held-out split — in-distribution numbers
turned out to be an unreliable predictor of real-world performance across earlier iterations
of this project, so nothing here is reported without an external check.

## Results

| Category | Benchmark | Relaxed F1 |
|---|---|---|
| NAME | CoNLL-2003 | 97.80% |
| LOCATION | CoNLL-2003 | 93.08% |
| GENE | BC2GM | 84.03% |
| CHEMICAL | BC4CHEMD | 84.33% |
| VARIANT | OSIRIS | 85.72% |
| DISEASE | CADEC (n=22 — see note below) | 61.36% |

Relaxed = exact type, any character overlap with the gold span. Full STRICT/RELAXED tables
and confidence intervals on the small benchmarks are in
**[LORA_RESULTS_SUMMARY.md](LORA_RESULTS_SUMMARY.md)**.

## Training data

`dataset/merged_clinical_phi_v16.{train,validation,test}.jsonl` — built by
`dataset/build_lora_dataset_v16.py`. Composition of the train split by source:

| Source | Category taught | Records | Kind |
|---|---|---|---|
| synthetic_pii | NAME, DATE, LOCATION | 16,022 | synthetic |
| CoNLL-2003 | NAME, LOCATION | 11,183 | real (Reuters newswire) |
| BC4CHEMD | CHEMICAL | 7,324 | real (PubMed, 30% of train split) |
| synthetic_combined | multi-category | 6,463 | synthetic |
| BioNLP 2004 | GENE | 5,333 | real (GENIA shared task) |
| BC2GM | GENE | 5,041 | real (PubMed, 50% of train split) |
| LINNAEUS | negatives only | 4,982 | real (biological-text negatives) |
| synthetic_phi_numeric_contrast | NAME, DATE, LOCATION vs. phone/SSN/IP | 4,783 | synthetic |
| synthetic_variant | VARIANT | 4,254 | synthetic |
| synthetic_phi | NAME, DATE, LOCATION | 3,715 | synthetic |
| Synthetic (PHI, clinical-narrative style) | DISEASE, CHEMICAL, DATE, NAME, LOCATION | 3,328 | synthetic |
| BC5CDR | CHEMICAL, DISEASE | 1,996 | real |
| BioRED | GENE, CHEMICAL, DISEASE, VARIANT | 1,198 | real (multi-entity) |
| NCBI-Disease | DISEASE | 1,044 | real |
| (biomedical negative mining) | negatives only | 408 | real (biological-text negatives) |
| tmVar v2 | VARIANT | 301 | real |
| CADEC | DISEASE | 96 | real (informal patient-forum text) |

CADEC was added specifically to fix DISEASE's remaining domain gap: its existing sources
(BC5CDR, NCBI-Disease) are formal PubMed-abstract text, and CADEC is informal, first-person
patient-forum language — the register DISEASE mentions actually needed. The 96 records above
are this project's training split only: CADEC's native train split is 127 records, and after
this project's own deterministic train/validation/test partitioning, 96 landed in training,
18 in validation, and 13 in this project's own test split (disjoint from the 22-record
official CADEC test split used for evaluation above). See `dataset/build_lora_dataset_v16.py`
for the exact diff and its leakage/balance checks.

## Repo layout

```
training/
  LoRa-Code-7cat.py       training script (LoRA r=64, alpha=128, on query/value projections)
  LoRa-Score.py           same-distribution test-set scorer
  LoRa-Raw.py             live free-text inference loop

evaluation/
  lora_eval_{conll2003,bc2gm,bc4chemd,cadec,osiris}.py
                          one self-contained eval script per real external benchmark,
                          STRICT/RELAXED/COLLAPSED scoring + confusion matrix PNG

dataset/
  build_lora_dataset_v16.py              builds the training data, documents what went in
  labels_v16.json                        7-category label schema
  merged_clinical_phi_v16.*.jsonl        the training/validation/test data itself

confusion_matrices/
  confusion_matrix_lora_deberta_on_*.png   gold-type vs. predicted-type confusion matrix,
                                             one per benchmark, v16 checkpoint

LORA_RESULTS_SUMMARY.md   full results, ablation history, confidence intervals
```

## Running an eval script

Each `evaluation/lora_eval_*.py` script is self-contained: point `BASE_MODEL_PATH` at a local
copy of DeBERTa-v3-base and `ADAPTER_PATH` at the trained LoRA adapter, then run it. Each run
scores the full benchmark split, prints STRICT/RELAXED/COLLAPSED per-type tables and a
confusion matrix, and writes `confusion_matrix_lora_deberta_on_<benchmark>.png` next to the
dataset.

`LORA_EVAL_MAX_RECORDS` (env var) caps how many records to score, for a quick test run.

## Known limitations

- The "Synthetic (PHI, clinical-narrative style)" source (3,328 records) is treated as
  synthetic because its provenance could not be confirmed as any specific, licensed corpus;
  its clinical/medication narrative reads as realistic rather than clearly template-generated,
  and a fraction of its records show a substitution artifact (the wrong token replaced during
  synthetic value insertion, e.g. a name standing in for the sig abbreviation "Sig"). If it is
  in fact derived from real, unattributed clinical records, its licensing status is unknown.
- CADEC's test split is 22 records / 47 gold DISEASE spans — real signal, but read with that
  scale in mind, not the same statistical confidence as the 5,000+ record benchmarks. A
  Wilson score interval on v16's relaxed recall: 74.5% [60.5%, 84.7%].
- DATE has no benchmark anywhere with gold date-span annotations; it's validated separately
  via a targeted stress test (see `LORA_RESULTS_SUMMARY.md`) rather than against an external
  corpus, and isn't included in the results table above for that reason.
- No significance testing beyond the Wilson intervals noted for the small benchmarks.
