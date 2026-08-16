# SOTA PII/PHI + Biomedical NER Benchmark — Consolidated Results

All runs on the full 61,288-record `benchmark_clinical_phi.jsonl`. "In-scope" P/R/F1
restricts each model's aggregate to only the entity types its own schema mapping can ever
predict (see conversation for full reasoning) — comparing against the full 30-category
schema unfairly penalizes models never designed to detect categories like DISEASE/GENE for
general PII tools, or PHONE/SSN for biomedical NER tools. Numbers below are STRICT (exact
type + exact span boundary) unless noted.

This benchmark covers 7 models. The biomedical specialist ensemble
(`sota_eval_medical_ner.py`) and openai/privacy-filter (`sota_eval_openai_privacy_filter.py`)
are excluded from the paper's scope by decision, not by data problem -- both scripts still
exist and work, just aren't part of the reported results.

**Provenance note (2026-08-14):** every number in both tables below was regenerated from a
single file (`modelsinfo.txt`, one execution per model, STRICT/RELAXED/COLLAPSED tables +
confusion-matrix JSON all from the same run) and cross-checked three ways: the confusion
matrix's row sums against each type's STRICT `tp+fn`, that `tp+fn` against a gold-total
invariant table built from all 7 models simultaneously (any two models disagreeing on the
same type's total would mean a parsing or data bug -- none did), and the in-scope P/R/F1
recomputed directly from summed tp/fp/fn rather than hand-read off the console. This caught a
real error: **spaCy's in-scope precision/F1 in this table were previously wrong (15.73%/24.07%,
should have been 21.64%/30.43%)** -- a transcription mistake made earlier in this project, not
a different run or a code bug (the underlying confusion matrix was already correct and
unchanged). Presidio and obi shifted by smaller amounts for the same reason. All 7 confusion
matrix PNGs were regenerated from this same file, so the images and the table numbers are now
guaranteed to come from the same execution.

**Second incident, same day:** a separate local test (checking that the JSON-dump code change
worked) wrote its own throwaway 300-record output directly into this project directory instead
of an isolated folder, silently overwriting the real 61,288-record Presidio PNG with a test
artifact -- caught only because the image was re-opened and its title read "(300 records)"
instead of "(61,288 records)". All 7 PNGs were regenerated a second time, directly from
`modelsinfo.txt`, with every row re-verified against the gold-total invariant table
immediately after rendering (not just once at transcription time). This is the second
near-miss in this project caused by an artifact silently diverging from its source without
an automatic check catching it immediately -- both are now closed, but it's a real pattern:
manual/incidental file writes are a recurring risk, and confusion matrices are the one
artifact type here with a strong-enough built-in invariant to catch it after the fact.

## General PII/PHI de-identification models

| model | in-scope types | in-scope P | in-scope R | in-scope F1 | status |
|---|---|---|---|---|---|
| Microsoft Presidio (threshold=0.4) | 15 | 45.47% | 58.86% | **51.31%** | clean |
| obi/deid_roberta_i2b2 | 8 | 23.64% | 51.46% | **32.40%** | clean, 2 real bugs fixed |
| StanfordAIMI/stanford-deidentifier-base | 6 | 49.49% | 91.99% | **64.35%** | clean, 1 real bug fixed |
| spaCy en_core_web_sm | 4 | 21.64% | 51.29% | **30.43%** | clean, crash bug fixed |

**Bugs found and fixed in this group:**
- Presidio: no confidence threshold on `analyzer.analyze()` flooded LICENSE_NUMBER with 0.01-0.30-confidence noise → added `score_threshold=0.4`.
- Dataset-wide: CREDIT_CARD numbers weren't Luhn-valid, causing Presidio's checksum validator to reject them → fixed generator, verified 0/1,388 Luhn failures post-fix.
- obi: trusted the model's own BILOU (B/I/L/U) tags literally, but they're internally unreliable (confirmed via real token trace — repeated `L-PHONE`, split `U-PATIENT`) → merge on same-schema-type adjacency instead, ignoring the BILOU prefix.
- obi + Stanford: schema-collapse-before-merge bug — native labels were compared for merging *before* mapping to schema type, so a model flip between two natively-different-but-schema-same labels (e.g. PATIENT→NAME, HCW→NAME) incorrectly fragmented one entity into two.
- spaCy: `ScoreAccumulator.report()` method was entirely missing (dropped in an earlier edit) — would have crashed on first real run. Never caught because spaCy hadn't been run since the edit that dropped it.
- All 4 + common module: RELAXED scorer double-counted recall — one `tp` bucket was shared between precision and recall numerators, so a model that fragments predictions (esp. obi) could get one gold entity "recalled" once per overlapping fragment, pushing reported recall above the true number of gold entities. Confirmed on obi's DATE row: relaxed recall total was reporting 15,345 "recalled" entities against a true gold total of 11,318. Fixed: precision and recall now use separate numerators (`tp_pred` vs `tp_gold`) everywhere.

**Known real findings (not bugs):**
- obi essentially cannot delineate full EMAIL address spans — reliably tags only the first BPE subword fragment then stops (confirmed via real token trace: `jane.kim14@mercygeneral.org` → predicted `jane` only). Genuine model limitation from synthetic/OOV email formats, not a bug in scoring.
- Presidio's LOCATION and NAME show large strict-vs-relaxed gaps (LOCATION: 2.12%→78.80%) — Presidio finds *something* overlapping almost every gold location but rarely on the exact boundary. Real Presidio behavior.

## Biomedical / clinical NER models

| model | scope | P | R | F1 | status |
|---|---|---|---|---|---|
| Ihor/gliner-biomed-base-v1.0 (zero-shot, all 30 categories) | 30 | 45.03% | 70.07% | **54.83%** | clean, GPU/batching bug fixed |
| Clinical-AI-Apollo/Medical-NER (DeBERTa-v2, MACCROBAT) | 4 (Age/Date/Disease/Occupation) | 44.05% | 50.76% | **47.16%** | clean post-fix |
| d4data/biomedical-ner-all (DistilBERT, same MACCROBAT) | 4 (same) | 29.38% | 26.17% | **27.69%** | clean, DATE gap explained |

**Bugs found and fixed in this group:**
- GLiNER: script never placed the model on GPU (`GLiNER.from_pretrained()` doesn't auto-detect CUDA) and called `predict_entities()` one record at a time instead of batching — <10% progress after 36 minutes on a T4. Fixed: explicit `.to(device)` + `batch_predict_entities()`, full run now 14.3 min.
- Clinical-AI-Apollo: **confirmed, root-caused bug** — DeBERTa-v2's SentencePiece tokenizer maps a word-initial token's offset to *include* its own leading space character, so every entity's predicted start was silently one character too early. STRICT scores were near-zero before the fix (DATE R=0.20%, AGE tp=0) and jumped to sane values after (DATE R=83.13%, AGE R=49.91%). Same defensive trim applied to d4data (harmless no-op there, confirmed by unchanged clean invariants).

**Known real findings (not bugs):**
- GLiNER's URL and BIOMETRIC_ID strict scores (both near-zero) are both genuine, confirmed via a diagnostic that reproduces the real script's exact 30-label prompt (an earlier version of this diagnostic used only 2 labels and got a misleadingly clean result — not a valid comparison, since GLiNER's predicted boundaries shift depending on the full label set). **URL**: GLiNER consistently splits every URL into two disconnected fragments — `"https"` and `"www.domain.tld"` — treating `"://"` as a gap, and in most examples the second fragment truncates to just `"www"`, dropping the domain entirely. This exactly explains the full run's relaxed `tp_pred=2644` vs `tp_gold=1322` (~2 fragments per gold URL). **BIOMETRIC_ID**: GLiNER tags only the code (`"VP-8629"`), not our gold spans' descriptive prefix (`"voiceprint ID VP-8629"`) — an annotation-scope difference, same class as Apollo's AGE finding.
- **d4data's huge DATE strict-vs-relaxed gap (32.60% vs 76.86%) is confirmed genuine fragmentation, not an offset bug**, via a diagnostic over 3,000 sampled records. No single `(start_delta, end_delta)` pair dominates (the top pattern is only ~30% of mismatches, versus a single pattern explaining ~85%+ of Apollo's original SentencePiece bug), which rules out a systematic offset. Instead d4data's Date detector breaks multi-component dates into separate disconnected sub-entities rather than one continuous span, for both numeric and spelled-out formats: `"1970-03-11"` → `"1970"` + `"03"` as two separate predictions (the day isn't captured at all); `"2023-10-18"` → only `"10"` (year and day both missing); `"November 27, 2022"` → `"November 27"` + `"2022"` as two separate predictions. The model fails to bridge across the punctuation/connector tokens (hyphens, slashes, commas) between date components, so the merge logic correctly reports what the model actually tagged: several disconnected fragments instead of one span. Same class of finding as obi's fragmentation and GLiNER's URL-splitting, just for d4data + DATE.
- Apollo never once produces an exact-boundary match for PROFESSION (`tp=0` of 2,649) — confirmed genuine via confusion-matrix arithmetic: summing every `PROFESSION -> X` row exactly reproduces the STRICT `fn` count, meaning Apollo consistently calls those spans something else (history, description, sex, disease) rather than missing them silently.
- Apollo vs d4data (architecture-controlled comparison, same task/labels): Apollo (DeBERTa-v2, larger) wins on DATE (75.72% vs 32.60%) and DISEASE (28.73% vs 14.97%); d4data (DistilBERT, lighter) wins on AGE (61.43% vs 39.56%) and is the only one with real PROFESSION signal (21.98% vs 0%). Apollo wins in aggregate only because of DATE's outsized weight.
- **Apollo's AGE/DISEASE strict-vs-relaxed gaps are confirmed genuine annotation-scope differences**, via a systematic diagnostic over 3,000 sampled records (not a single spot-check). AGE: 74.6% of mismatches are Apollo *extending* the span to include the unit (gold `"28"` → pred `"28 years old"`, consistent across every extension example); a further 25.4% are Apollo *dropping the leading digit* of hyphenated compound ages (gold `"2-year-old"`/`"95-year-old"`/`"12-year-old"` → pred `"-year-old"` every time), a distinct model limitation on that specific format. DISEASE is the mirror image: 98.6% of mismatches are Apollo *dropping* descriptive/severity modifiers and keeping only the core disease name (gold `"acute Essential Hypertension"` → pred `"Essential Hypertension"`, `"metastatic Bronchitis"` → pred `"Bronchitis"`, `"bilateral Pancreatic Cancer"` → pred `"Pancreatic Cancer"`, and so on through nearly every example) — our gold spans bundle severity/laterality/chronicity qualifiers into the DISEASE span, MACCROBAT training doesn't. A smaller secondary pattern shows genuine mid-word subword-tagging failures on long/rare compound terms (`"Kniest dysplasia"` → `"est dysplasia"`; `"Promyelocytic Leukemia"` → disconnected `"yel"` + `"Leukemia"`; `"non-alcoholic fatty liver disease"` splitting at the hyphen into `"non"` + `"alcoholic fatty liver disease"`) — real model behavior on morphologically complex vocabulary, not a scoring bug.

## Confusion matrices (gold type vs. predicted type, STRICT axis, in-scope rows)

| model | image | source |
|---|---|---|
| Microsoft Presidio | `confusion_matrix_microsoft_presidio.png` | full 61,288 records, all row totals verified exactly against gold-total invariants — **done** |
| obi/deid_roberta_i2b2 | `confusion_matrix_obi_deid_roberta_i2b2.png` | full 61,288 records, all row totals verified — **done** |
| StanfordAIMI/stanford-deidentifier-base | `confusion_matrix_stanfordaimi_stanford_deidentifier_base.png` | full 61,288 records, all row totals verified — **done** |
| spaCy en_core_web_sm | `confusion_matrix_spacy_en_core_web_sm.png` | full 61,288 records, all row totals verified — **done**. Notable real finding: only 251/10,644 gold LOCATION entities land on `LOCATION` at the exact boundary; 3,125 land on `ORGANIZATION` instead (~12x the correct-type count) -- spaCy is frequently right on the span, wrong on the label, likely a facility/hospital-name ORG-vs-GPE disagreement, not a bug |
| Ihor/gliner-biomed-base-v1.0 | `confusion_matrix_ihor_gliner_biomed_base_v1_0.png` | full 61,288 records, all 30 rows verified exactly against gold-total invariants — **done** |
| Clinical-AI-Apollo/Medical-NER | `confusion_matrix_clinical_ai_apollo_medical_ner.png` | full 61,288 records, all row totals verified — **done** |
| d4data/biomedical-ner-all | `confusion_matrix_d4data_biomedical_ner_all.png` | full 61,288 records, all row totals verified — **done** |

All 7 in-scope models now have verified confusion matrices, cross-checked row-by-row against
the dataset's fixed per-type gold totals (e.g. DATE=11,318, NAME=16,066 hold across every
model regardless of which one is being scored) — every row's cells sum exactly to that type's
true count, confirming the double-counting fix holds at full scale everywhere it's been
tested.

All prior confusion-matrix images and their backing JSON were deleted from this directory
before regenerating -- everything either predated the 3-way strict/relaxed/collapsed
confusion code or the double-counting fix below, and stale images risked someone grabbing a
wrong number by mistake. All 7 rerun so far (Presidio, obi, Stanford, spaCy, GLiNER, Apollo,
d4data) came back clean.

**Confusion matrix generation is now automatic.** All 7 in-scope scripts embed their own
`render_confusion_matrix_png()` and call it at the end of `report()`, writing
`confusion_matrix_<model>.png` straight to the same Google Drive folder as the dataset — no
manual copy/paste out of console output required anymore. This exists because
hand-transcribing confusion tables out of Colab output (which also truncates past ~50,000
characters) produced real, silent data-quality problems earlier in this project.

**Bug found and fixed in the confusion-matrix generator itself** (`_update_confusion_exact`,
affects the STRICT and COLLAPSED confusion matrices in all scripts): the original version
paired a gold span with *every* prediction sharing its exact boundary instead of just one.
Presidio's digit-heavy fields (ACCOUNT_NUMBER, CREDIT_CARD, IP_ADDRESS) routinely trigger 2-3
regex recognizers on the identical span, so one gold entity got tallied multiple times --
inflating the confusion matrix's row total past the true gold count (ACCOUNT_NUMBER showed
n=3,143 against a true gold total of 2,686) while silently disagreeing with the STRICT
P/R/F1 table printed right above it in the same report. Fixed by enforcing the same 1:1
gold/prediction pairing the STRICT/COLLAPSED tables already use -- verified on a real
full-61,288-record Presidio rerun, where every row now sums exactly to its STRICT-table gold
total (ACCOUNT_NUMBER 2,282+320+81+3=2,686; DATE 11,099+3+4+62+150=11,318).
**This bug never affected any P/R/F1 number in this document** -- it only corrupted the
confusion-matrix visualizations, and only for models/categories where predictions of
different types land on the exact same character span. Re-verified on all 7 regenerated
images: every row across all of them (including all 30 rows of GLiNER's full-schema matrix)
sums exactly to that type's known gold total.

## Open items before treating results as final

None. All findings below are resolved as genuine model behavior, not scoring bugs, and every
in-scope model's confusion matrix has been regenerated and verified against gold-total
invariants.

**Code audit (2026-08-13):** re-read the core scoring/merge/confusion logic in all 7
in-scope scripts looking for anything else like the double-counting bug. Found nothing new
-- and the fact that every row of every regenerated confusion matrix (including all 30 of
GLiNER's) lands exactly on the correct gold total is itself strong evidence the matching
logic is sound, since a hidden bug in that path would very likely have shown up as a
mismatch somewhere across 7 models and hundreds of rows.

Resolved: GLiNER's URL and BIOMETRIC_ID rows were flagged as a possible repeat of the Apollo
bug — a diagnostic using the real script's exact 30-label prompt (see findings above) showed
both are genuine GLiNER behavior, not a bug. No code change needed or possible (GLiNER
computes its own span boundaries internally).

Resolved: Apollo's AGE/DISEASE strict-vs-relaxed gaps are confirmed genuine annotation-scope
differences via `diag_apollo_age_disease.py` over 3,000 sampled records (see findings above)
-- not a bug, no code change possible or needed.

Resolved: d4data's DATE strict-vs-relaxed gap is confirmed genuine fragmentation via
`diag_d4data_date.py` over 3,000 sampled records (see findings above) -- not a bug, no code
change possible or needed (the model itself doesn't bridge across date-component
punctuation).
