# LLM-Prompt-Desensitization-Healthcare

A 30-category healthcare PII/PHI + clinical-entity NER benchmark, and a scored comparison
of 7 SOTA PII/PHI and biomedical NER models against it. The dataset is the contribution
here — this repo does not train a model, it measures how well existing models handle
healthcare text de-identification and clinical entity extraction.

## Entity schema (30 types)

`ACCOUNT_NUMBER, AGE, BIOMETRIC_ID, CELL, CHEMICAL, CREDIT_CARD, CRYPTO_WALLET, DATE,
DEVICE_ID, DISEASE, EMAIL, FAX, GENE, HEALTH_PLAN_ID, IP_ADDRESS, LICENSE_NUMBER,
LOCATION, MEDICAL_RECORD_NUMBER, NAME, ORGANIZATION, OTHER_ID, PASSPORT_NUMBER, PHONE,
PROFESSION, SPECIES, SSN, URL, USERNAME, VARIANT, VEHICLE_ID`

Covers both classic PHI/PII (names, dates, contact info, identifiers) and biomedical
entities (diseases, genes, chemicals, species, cell lines, genetic variants) in the same
schema, so both general-purpose de-identification tools and biomedical NER models can be
scored on the same data.

## Results

Full writeup, per-model bugs found and fixed, real findings vs. scoring artifacts, and the
evidence behind every number: **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)**.

| model | in-scope types | P | R | F1 |
|---|---|---|---|---|
| Microsoft Presidio (threshold=0.4) | 15 | 45.47% | 58.86% | **51.31%** |
| obi/deid_roberta_i2b2 | 8 | 23.64% | 51.46% | **32.40%** |
| StanfordAIMI/stanford-deidentifier-base | 6 | 49.49% | 91.99% | **64.35%** |
| spaCy en_core_web_sm | 4 | 21.64% | 51.29% | **30.43%** |
| Ihor/gliner-biomed-base-v1.0 (zero-shot, 30 categories) | 30 | 45.03% | 70.07% | **54.83%** |
| Clinical-AI-Apollo/Medical-NER | 4 | 44.05% | 50.76% | **47.16%** |
| d4data/biomedical-ner-all | 4 | 29.38% | 26.17% | **27.69%** |

"In-scope" P/R/F1 restricts each model's aggregate to only the entity types its own label
scheme can ever predict — scoring a PII tool against biomedical categories it was never
built to detect (or vice versa) isn't a fair comparison. STRICT matching (exact type +
exact span boundary) throughout; see RESULTS_SUMMARY.md for RELAXED and COLLAPSED axes and
the full per-category confusion matrices.

## Repo layout

```
build_dataset.py, export_benchmark.py, validate_dataset.py   dataset generation pipeline
labels.json                                                   the 30-category schema
benchmark_clinical_phi*.jsonl                                 the benchmark itself (raw
                                                                text + character-offset
                                                                gold entity spans)

sota_eval_common.py                                            shared scoring/confusion-
                                                                matrix infrastructure
sota_eval_{presidio,obi_i2b2,stanford_deid,spacy,
           gliner_biomed,clinical_ai_apollo,
           biomedical_ner_all}.py                              one self-contained,
                                                                Colab-ready eval script
                                                                per model

diag_*.py                                                       root-cause diagnostics for
                                                                real findings (SentencePiece
                                                                offset bug, DATE
                                                                fragmentation, AGE/DISEASE
                                                                annotation-scope gaps, ...)

confusion_matrix_*.png                                          gold-type vs. predicted-
                                                                type confusion matrix per
                                                                model, full 61,288-record
                                                                runs, auto-generated

modelsinfo.txt, diag.txt                                        raw console output backing
                                                                every number and finding in
                                                                RESULTS_SUMMARY.md
```

## Running an evaluation

Each `sota_eval_*.py` script is self-contained (stdlib + the one model's own pip package,
no local imports) and meant to be run in Google Colab, one model at a time:

```python
!pip install <model's package>   # see the top of each script
```
Then run the script (or paste it into a cell). Set `DATASET_PATH` to wherever
`benchmark_clinical_phi.jsonl` lives (defaults to a Google Drive path), and it scores the
full dataset, prints STRICT/RELAXED/COLLAPSED per-type tables plus confusion matrices, and
writes `confusion_matrix_<model>.png` next to the dataset automatically.

`SOTA_EVAL_MAX_RECORDS` (env var) caps how many records to score, for a quick local test.

## Known limitations

- The benchmark is synthetic (template-generated), not real clinical text — obi and
  Stanford's underlying models were trained on real i2b2 notes, so there's an unmeasured
  domain-shift gap in both directions.
- "In-scope" P/R/F1 is a fairness metric for comparing models, not a practical
  "how much PHI did this tool actually catch" number — that would need full-schema recall
  reported separately.
- Presidio's threshold (0.4) and GLiNER's threshold (0.5) are single operating points, not
  tuned via a precision/recall sweep.
- No confidence intervals or significance testing on any reported F1.
- A biomedical specialist ensemble (Disease/Gene/Chemical/Species detectors) and an
  `openai/privacy-filter` evaluation were built during this project but are excluded from
  the reported results by decision, not by data problem.
