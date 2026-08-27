"""
Scores the LoRA-fine-tuned DeBERTa NER model against OSIRIS, testing VARIANT recognition
against real, independently-annotated MEDLINE abstracts describing human genetic variation
-- never used anywhere in this project's training data (unlike tmvar_v2, this project's
existing real VARIANT source, which is a training source; other tmVar revisions were
avoided as candidates precisely because they're close relatives of the same underlying
corpus and risk sentence overlap).

Loaded via the HF datasets-server rows API (bigbio/osiris, config osiris_bigbio_kb) rather
than the `datasets` library's own load_dataset -- confirmed directly that this repo still
carries a script-based loader that datasets 4.0+ refuses to run at all (same failure mode
fixed earlier in this project for lhoestq/conll2003 and others). The rows API sidesteps
this entirely and is the same technique dataset/build_dataset.py already uses for its own
tmvar_v2/linnaeus sources.

This is a full-text/character-offset "kb" schema, not pre-tokenized BIO, so this script
chunks each document into sentence-sized pieces the same way dataset/build_dataset.py
already does for tmvar_v2/linnaeus, then maps entities onto each chunk by character offset.
Its native scheme was checked directly via the HF datasets-server API: OSIRIS annotates
"variant" and "gene" mentions; only "variant" maps to this benchmark's VARIANT, "gene" is
dropped from gold entirely (not folded into GENE, since this comparison is scoped to
VARIANT only).

OSIRIS has no separate test split (105 documents, all under "train"), but since none of it
was ever used in this project's training data, the entire split is a legitimate held-out
eval set -- there is nothing to leak. CC-BY licensed (Furlong et al., BMC Bioinformatics
2008). Note the corpus is small -- real signal, but read the numbers with that scale in
mind, not with the same statistical confidence as the 5,000+ record benchmarks used for
the other categories.

Dependencies: transformers, torch, peft, matplotlib. (No `datasets` library needed for
loading this one -- see the rows-API helpers below.)
"""
import os
import re
import json
import time
import urllib.request
import urllib.parse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

BASE_MODEL_PATH = "/content/drive/MyDrive/MyProjectFolder/local_deberta_model_base"
ADAPTER_PATH = "/content/drive/MyDrive/MyProjectFolder/v16"
OUT_DIR = os.environ.get("LORA_EVAL_OUT_DIR", "/content/drive/MyDrive/MyProjectFolder")
MAX_RECORDS = int(os.environ.get("LORA_EVAL_MAX_RECORDS", "0") or 0)
BATCH_SIZE = int(os.environ.get("LORA_EVAL_BATCH_SIZE", "16") or 16)

MODEL_NAME = "LoRA-DeBERTa (r64_alpha128) on OSIRIS"
DATASET_REPO = "bigbio/osiris"
DATASET_CONFIG = "osiris_bigbio_kb"
IN_SCOPE_TYPES = ["VARIANT"]

master_labels = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-CHEMICAL", "I-CHEMICAL",
    "B-GENE", "I-GENE",
    "B-VARIANT", "I-VARIANT",
    "B-NAME", "I-NAME",
    "B-DATE", "I-DATE",
    "B-LOCATION", "I-LOCATION",
]
label2id = {label: i for i, label in enumerate(master_labels)}
id2label = {i: label for i, label in enumerate(master_labels)}


def map_native_type(native_type):
    """OSIRIS annotates "variant" and "gene". Only "variant" maps to this benchmark's
    VARIANT; "gene" is dropped from gold entirely, not folded into GENE, since this
    comparison is scoped to VARIANT only."""
    return "VARIANT" if native_type.lower() == "variant" else None


# ==========================================
# 2. LOAD MODEL (mirrors LoRa-Score.py / LoRa-Raw.py)
# ==========================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
print("Loading base DeBERTa model...")
base_model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL_PATH, num_labels=len(master_labels), id2label=id2label, label2id=label2id,
    local_files_only=True,
    # local_deberta_base still carries a classifier head sized for the old 19-label
    # (9-category) schema; harmless to reinit since PeftModel.from_pretrained below
    # replaces it with the adapter's own trained 15-label classifier anyway.
    ignore_mismatched_sizes=True,
)
print(f"Attaching adapters from: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("Merging weights...")
merged_model = model.merge_and_unload()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    merged_model = merged_model.half().to(device)
else:
    merged_model = merged_model.to(device)
merged_model.eval()
print(f"Ready. Running on {device}.\n")


def _trim_span_whitespace(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def predict_spans_batch(texts):
    enc = tokenizer(texts, return_tensors="pt", return_offsets_mapping=True,
                     truncation=True, max_length=192, padding=True)
    offset_mapping = enc.pop("offset_mapping").tolist()
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = merged_model(**inputs).logits
    pred_ids = logits.argmax(dim=-1).tolist()
    results = []
    for bi, text in enumerate(texts):
        entities = []
        cur = None
        for idx in range(len(pred_ids[bi])):
            cs, ce = offset_mapping[bi][idx]
            if cs == ce:
                continue
            label = id2label[pred_ids[bi][idx]]
            if label == "O":
                if cur:
                    entities.append(cur)
                    cur = None
                continue
            prefix, ent_type = label[:2], label[2:]
            if prefix == "B-" or cur is None or cur["type"] != ent_type:
                if cur:
                    entities.append(cur)
                cur = {"start": cs, "end": ce, "type": ent_type}
            else:
                cur["end"] = ce
        if cur:
            entities.append(cur)
        trimmed = []
        for e in entities:
            s, en = _trim_span_whitespace(text, e["start"], e["end"])
            if s < en:
                trimmed.append({"type": e["type"], "start": s, "end": en, "text": text[s:en]})
        results.append(trimmed)
    return results


# ==========================================
# 3. DOCUMENT -> SENTENCE-CHUNK CONVERSION (same technique as
# dataset/build_dataset.py's _passage_to_sentence_examples)
# ==========================================
_SENT_BOUNDARY_RE = re.compile(r"[.!?]\s+(?=[A-Z0-9])")
_SENT_ABBREVIATIONS = {
    "fig", "figs", "st", "dr", "mr", "mrs", "ms", "vs", "eg", "ie", "etc", "al",
    "no", "vol", "cf", "approx", "sp", "spp", "var", "cv", "ca", "resp", "prof",
    "inc", "co", "corp", "jr", "sr", "ed", "eds", "pp", "viz", "e.g", "i.e",
}

def _split_sentences_with_offsets(text):
    spans = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        split_at = m.start() + 1
        word_match = re.search(r"(\S+)$", text[start:m.start() + 1])
        prev_word = word_match.group(1).rstrip(".!?") if word_match else ""
        if not prev_word:
            continue
        if prev_word.replace(".", "").isdigit():
            continue
        if prev_word.lower() in _SENT_ABBREVIATIONS or (len(prev_word) <= 2 and prev_word.isalpha()):
            continue
        spans.append((start, split_at))
        start = split_at
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def document_to_chunks(text, p_start, entities):
    """Splits one document's full text into sentence-sized chunks, maps entities onto
    each chunk by character offset, and yields (chunk_text, gold_spans) pairs. Only
    entities fully contained within a chunk are kept."""
    chunks = []
    line_start = 0
    for line in text.split("\n"):
        for rel_start, rel_end in _split_sentences_with_offsets(line):
            chunk_start = line_start + rel_start
            chunk_end = line_start + rel_end
            chunk_text = text[chunk_start:chunk_end]
            gold = []
            for ent in entities:
                mapped = map_native_type(ent["type"])
                if mapped is None:
                    continue
                for est, eend in ent["offsets"]:
                    gs, ge = est - p_start, eend - p_start
                    if chunk_start <= gs and ge <= chunk_end:
                        ls, le = gs - chunk_start, ge - chunk_start
                        gold.append({"type": mapped, "start": ls, "end": le,
                                      "text": chunk_text[ls:le]})
            if chunk_text.strip():
                chunks.append((chunk_text, gold))
        line_start += len(line) + 1
    return chunks


# ==========================================
# 4. SCORING INFRASTRUCTURE (identical to sota_eval_common.py / the sota_eval_*.py scripts)
# ==========================================
def score_predictions_strict(gold_spans, pred_spans):
    matched_gold = [False] * len(gold_spans)
    tp, fp = [], []
    for p in pred_spans:
        found = False
        for i, g in enumerate(gold_spans):
            if not matched_gold[i] and p["type"] == g["type"] and p["start"] == g["start"] and p["end"] == g["end"]:
                matched_gold[i] = True
                tp.append(g)
                found = True
                break
        if not found:
            fp.append(p)
    fn = [g for i, g in enumerate(gold_spans) if not matched_gold[i]]
    return tp, fp, fn


def score_predictions_relaxed(gold_spans, pred_spans):
    def overlaps(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]
    gold_found = [False] * len(gold_spans)
    pred_tp, fp = [], []
    for p in pred_spans:
        match = False
        for i, g in enumerate(gold_spans):
            if p["type"] == g["type"] and overlaps(p, g):
                match = True
                gold_found[i] = True
        (pred_tp if match else fp).append(p)
    gold_tp = [g for i, g in enumerate(gold_spans) if gold_found[i]]
    fn = [g for i, g in enumerate(gold_spans) if not gold_found[i]]
    return pred_tp, fp, fn, gold_tp


COLLAPSED_GROUPS = {
    "SSN": "IDENTIFIER", "MEDICAL_RECORD_NUMBER": "IDENTIFIER", "HEALTH_PLAN_ID": "IDENTIFIER",
    "ACCOUNT_NUMBER": "IDENTIFIER", "LICENSE_NUMBER": "IDENTIFIER", "VEHICLE_ID": "IDENTIFIER",
    "DEVICE_ID": "IDENTIFIER", "PASSPORT_NUMBER": "IDENTIFIER", "BIOMETRIC_ID": "IDENTIFIER",
    "CREDIT_CARD": "IDENTIFIER", "CRYPTO_WALLET": "IDENTIFIER", "OTHER_ID": "IDENTIFIER",
    "PHONE": "CONTACT", "FAX": "CONTACT", "EMAIL": "CONTACT", "URL": "CONTACT",
    "IP_ADDRESS": "CONTACT",
}


def _collapsed_type(schema_type):
    return COLLAPSED_GROUPS.get(schema_type, schema_type)


def score_predictions_collapsed(gold_spans, pred_spans):
    matched_gold = [False] * len(gold_spans)
    tp, fp = [], []
    for p in pred_spans:
        p_group = _collapsed_type(p["type"])
        found = False
        for i, g in enumerate(gold_spans):
            if (not matched_gold[i] and _collapsed_type(g["type"]) == p_group
                    and p["start"] == g["start"] and p["end"] == g["end"]):
                matched_gold[i] = True
                tp.append(g)
                found = True
                break
        if not found:
            fp.append(p)
    fn = [g for i, g in enumerate(gold_spans) if not matched_gold[i]]
    return tp, fp, fn


def _update_confusion_exact(confusion, gold_spans, pred_spans, collapse=False):
    key_fn = _collapsed_type if collapse else (lambda t: t)
    gold_matched = [False] * len(gold_spans)
    pred_matched = [False] * len(pred_spans)
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if (not pred_matched[pi] and g["start"] == p["start"]
                    and g["end"] == p["end"]):
                key = (key_fn(g["type"]), key_fn(p["type"]))
                confusion[key] = confusion.get(key, 0) + 1
                gold_matched[gi] = True
                pred_matched[pi] = True
                break
    for gi, g in enumerate(gold_spans):
        if not gold_matched[gi]:
            key = (key_fn(g["type"]), "MISSED")
            confusion[key] = confusion.get(key, 0) + 1
    for pi, p in enumerate(pred_spans):
        if not pred_matched[pi]:
            key = ("SPURIOUS", key_fn(p["type"]))
            confusion[key] = confusion.get(key, 0) + 1


def _update_confusion_overlap(confusion, gold_spans, pred_spans):
    def overlaps(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]
    gold_matched = [False] * len(gold_spans)
    pred_matched = [False] * len(pred_spans)
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if overlaps(g, p):
                key = (g["type"], p["type"])
                confusion[key] = confusion.get(key, 0) + 1
                gold_matched[gi] = True
                pred_matched[pi] = True
    for gi, g in enumerate(gold_spans):
        if not gold_matched[gi]:
            key = (g["type"], "MISSED")
            confusion[key] = confusion.get(key, 0) + 1
    for pi, p in enumerate(pred_spans):
        if not pred_matched[pi]:
            key = ("SPURIOUS", p["type"])
            confusion[key] = confusion.get(key, 0) + 1


def _safe_filename(name):
    # Strip a "(...)" parenthetical (e.g. "(r64_alpha128)") without dropping whatever
    # comes after it. The old version used name.split("(")[0], which silently discarded
    # the "on <dataset>" suffix -- every eval script's MODEL_NAME follows that same
    # "X (params) on Y" shape, so they all collapsed to the same "confusion_matrix_
    # lora_deberta.png" and overwrote each other's output on every run.
    paren_start = name.find("(")
    paren_end = name.find(")", paren_start)
    if paren_start != -1 and paren_end != -1:
        name = name[:paren_start] + name[paren_end + 1:]
    out, prev_underscore = [], False
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    return "".join(out).strip("_")


_RAMP = ["#f7fbff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_DIAG_COLOR = "#eb6834"


def render_confusion_matrix_png(acc, model_name, in_scope_types, out_path, axis="strict"):
    if not _HAS_MPL:
        print(f"[confusion matrix] matplotlib not available, skipping PNG render for {model_name}")
        return
    confusion = acc.confusion_strict if axis == "strict" else acc.confusion_collapsed

    cell, row_totals, spurious = {}, {}, {}
    for (g, p), c in confusion.items():
        if g == "SPURIOUS":
            spurious[p] = spurious.get(p, 0) + c
            continue
        if g not in in_scope_types:
            continue
        cell.setdefault(g, {})[p] = cell.get(g, {}).get(p, 0) + c
        row_totals[g] = row_totals.get(g, 0) + c

    rows = [t for t in in_scope_types if row_totals.get(t)]
    if not rows:
        print(f"[confusion matrix] no in-scope gold rows found, skipping PNG render for {model_name}")
        return
    cols = set()
    for g in rows:
        cols.update(cell.get(g, {}).keys())
    cols.discard("MISSED")
    cols = sorted(cols)
    cols.append("MISSED")

    matrix = [[cell.get(g, {}).get(c, 0) for c in cols] for g in rows]
    spurious_top = sorted(spurious.items(), key=lambda x: -x[1])[:10]
    spurious_total = sum(spurious.values())

    cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
    n_rows, n_cols = len(rows), len(cols)
    fig_w = max(6.5, 0.55 * n_cols)
    fig_h = max(3.0, 0.5 * n_rows + 1.8) + 2.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=200,
                                    gridspec_kw={"height_ratios": [max(1.2, 0.4 * n_rows + 0.8), 1.8]})

    norm = [[matrix[r][c] / max(1, max(matrix[r])) for c in range(n_cols)] for r in range(n_rows)]
    ax1.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for r in range(n_rows):
        for c in range(n_cols):
            v = matrix[r][c]
            if v == 0:
                continue
            is_diag = rows[r] == cols[c]
            txt_color = "white" if norm[r][c] > 0.55 else "#111111"
            ax1.text(c, r, f"{v:,}", ha="center", va="center", fontsize=7.5,
                      color=txt_color, fontweight="bold" if is_diag else "normal")
            if is_diag:
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                             edgecolor=_DIAG_COLOR, linewidth=1.6))
    ax1.set_xticks(range(n_cols))
    ax1.set_xticklabels(cols, rotation=90, fontsize=7)
    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels([f"{g} (n={row_totals[g]:,})" for g in rows], fontsize=8)
    ax1.set_xlabel("predicted type", fontsize=8.5)
    ax1.set_ylabel("gold type", fontsize=8.5)
    ax1.set_title(f"{model_name}  ({acc.total_records:,} records)", fontsize=10.5, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax1.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax1.grid(which="minor", color="white", linewidth=1.2)
    ax1.tick_params(which="minor", bottom=False, left=False)

    if spurious_top:
        labels = [t[0] for t in spurious_top][::-1]
        counts = [t[1] for t in spurious_top][::-1]
        ax2.barh(labels, counts, color="#eb6834")
        ax2.set_xlabel("count", fontsize=8)
        ax2.set_title(f"Top predicted types with NO gold overlap at all: {spurious_total:,} total spurious predictions",
                       fontsize=9)
        ax2.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
    else:
        ax2.axis("off")

    fig.suptitle(f"{axis.upper()} confusion (gold vs predicted type, exact span boundary), "
                  "in-scope gold rows only; native model categories the benchmark has no "
                  "equivalent for are kept as columns, not discarded",
                  fontsize=7.5, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[confusion matrix] wrote {out_path}")


class ScoreAccumulator:
    def __init__(self):
        self.per_type_strict = {}
        self.per_type_relaxed = {}
        self.per_type_collapsed = {}
        self.confusion_strict = {}
        self.confusion_relaxed = {}
        self.confusion_collapsed = {}
        self.total_records = 0

    def add(self, gold_spans, pred_spans):
        self.total_records += 1
        tp, fp, fn = score_predictions_strict(gold_spans, pred_spans)
        self._tally(self.per_type_strict, tp, fp, fn)
        pred_tp, fp, fn, gold_tp = score_predictions_relaxed(gold_spans, pred_spans)
        self._tally_relaxed(self.per_type_relaxed, pred_tp, fp, fn, gold_tp)
        tp, fp, fn = score_predictions_collapsed(gold_spans, pred_spans)
        self._tally(self.per_type_collapsed, tp, fp, fn, collapse=True)
        _update_confusion_exact(self.confusion_strict, gold_spans, pred_spans, collapse=False)
        _update_confusion_overlap(self.confusion_relaxed, gold_spans, pred_spans)
        _update_confusion_exact(self.confusion_collapsed, gold_spans, pred_spans, collapse=True)

    @staticmethod
    def _tally(bucket, tp, fp, fn, collapse=False):
        key = _collapsed_type if collapse else (lambda t: t)
        for g in tp:
            bucket.setdefault(key(g["type"]), [0, 0, 0])[0] += 1
        for p in fp:
            bucket.setdefault(key(p["type"]), [0, 0, 0])[1] += 1
        for g in fn:
            bucket.setdefault(key(g["type"]), [0, 0, 0])[2] += 1

    @staticmethod
    def _tally_relaxed(bucket, pred_tp, fp, fn, gold_tp):
        for p in pred_tp:
            bucket.setdefault(p["type"], [0, 0, 0, 0])[0] += 1
        for p in fp:
            bucket.setdefault(p["type"], [0, 0, 0, 0])[1] += 1
        for g in gold_tp:
            bucket.setdefault(g["type"], [0, 0, 0, 0])[2] += 1
        for g in fn:
            bucket.setdefault(g["type"], [0, 0, 0, 0])[3] += 1

    @staticmethod
    def _print_table(bucket, title):
        print(f"--- {title} ---")
        totals = [0, 0, 0]
        for etype in sorted(bucket):
            tp, fp, fn = bucket[etype]
            totals[0] += tp; totals[1] += fp; totals[2] += fn
            p = tp / (tp + fp) if (tp + fp) else float("nan")
            r = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = 2 * p * r / (p + r) if (p + r) and (tp + fp) and (tp + fn) else float("nan")
            print(f"  {etype:<24} P={p:.2%}  R={r:.2%}  F1={f1:.2%}  (tp={tp} fp={fp} fn={fn})")
        tp, fp, fn = totals
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
        print(f"\n  OVERALL{'':<18} P={p:.2%}  R={r:.2%}  F1={f1:.2%}  (tp={tp} fp={fp} fn={fn})")
        print()

    @staticmethod
    def _print_table_relaxed(bucket, title):
        print(f"--- {title} ---")
        totals = [0, 0, 0, 0]
        for etype in sorted(bucket):
            ptp, fp, gtp, fn = bucket[etype]
            totals[0] += ptp; totals[1] += fp; totals[2] += gtp; totals[3] += fn
            p = ptp / (ptp + fp) if (ptp + fp) else float("nan")
            r = gtp / (gtp + fn) if (gtp + fn) else float("nan")
            f1 = 2 * p * r / (p + r) if (ptp + fp) and (gtp + fn) and (p + r) else float("nan")
            print(f"  {etype:<24} P={p:.2%}  R={r:.2%}  F1={f1:.2%}  (tp_pred={ptp} fp={fp} tp_gold={gtp} fn={fn})")
        ptp, fp, gtp, fn = totals
        p = ptp / (ptp + fp) if (ptp + fp) else float("nan")
        r = gtp / (gtp + fn) if (gtp + fn) else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
        print(f"\n  OVERALL{'':<18} P={p:.2%}  R={r:.2%}  F1={f1:.2%}  (tp_pred={ptp} fp={fp} tp_gold={gtp} fn={fn})")
        print()

    @staticmethod
    def _print_confusion(confusion, title):
        print(f"--- {title} (gold -> predicted : count, sorted by count desc) ---")
        for (g, p), c in sorted(confusion.items(), key=lambda kv: -kv[1]):
            print(f"  {g:<24} -> {p:<24} {c}")
        print()

    def report(self, model_name, in_scope_types=None, out_dir=None):
        print("=" * 90)
        print(f"RESULTS: {model_name}  ({self.total_records} records scored)")
        print("=" * 90)
        self._print_table(self.per_type_strict, "STRICT (exact type + exact span boundary)")
        self._print_table_relaxed(self.per_type_relaxed, "RELAXED (exact type, any character overlap)")
        self._print_table(self.per_type_collapsed, "COLLAPSED (same identifier/contact family, exact span boundary)")
        self._print_confusion(self.confusion_strict,
                               "CONFUSION MATRIX: STRICT (exact span boundary, any type pairing)")
        self._print_confusion(self.confusion_relaxed,
                               "CONFUSION MATRIX: RELAXED (any character overlap, any type pairing)")
        self._print_confusion(self.confusion_collapsed,
                               "CONFUSION MATRIX: COLLAPSED (exact span boundary, collapsed family pairing)")
        print("=== CONFUSION MATRIX JSON ===")
        print(json.dumps({
            "model": model_name,
            "n_records": self.total_records,
            "per_type_strict": self.per_type_strict,
            "per_type_relaxed": self.per_type_relaxed,
            "per_type_collapsed": self.per_type_collapsed,
            "confusion_strict": [[k[0], k[1], v] for k, v in self.confusion_strict.items()],
            "confusion_relaxed": [[k[0], k[1], v] for k, v in self.confusion_relaxed.items()],
            "confusion_collapsed": [[k[0], k[1], v] for k, v in self.confusion_collapsed.items()],
        }))
        if in_scope_types:
            safe_name = _safe_filename(model_name)
            out_path = os.path.join(out_dir or ".", f"confusion_matrix_{safe_name}.png")
            render_confusion_matrix_png(self, model_name, sorted(set(in_scope_types)), out_path)


# ==========================================
# 5. FETCH VIA HF DATASETS-SERVER ROWS API (script-based repo, load_dataset() refuses to
# run it -- see module docstring), CONVERT, RUN
# ==========================================
REQUEST_TIMEOUT = 60

def _fetch_bytes(url, retries=5, backoff=2):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phi-eval/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = backoff * 5 * (attempt + 1) if "429" in str(e) else backoff * (attempt + 1)
                time.sleep(wait)
    raise last_err

def fetch_json(url, **kw):
    return json.loads(_fetch_bytes(url, **kw))

def get_split_row_count(dataset_id, config, split):
    url = ("https://datasets-server.huggingface.co/size"
           f"?dataset={urllib.parse.quote(dataset_id, safe='')}&config={urllib.parse.quote(config, safe='')}")
    data = fetch_json(url)
    for s in data.get("size", {}).get("splits", []):
        if s["split"] == split:
            return s["num_rows"]
    return 0

def fetch_rows_via_datasets_server(dataset_id, config, split, page_size=50):
    total_rows = get_split_row_count(dataset_id, config, split)
    if total_rows == 0:
        print(f"Warning: {dataset_id}/{config}/{split} reports 0 rows.")
        return []
    rows, offset = [], 0
    while offset < total_rows:
        length = min(page_size, total_rows - offset)
        url = ("https://datasets-server.huggingface.co/rows"
               f"?dataset={urllib.parse.quote(dataset_id, safe='')}&config={urllib.parse.quote(config, safe='')}"
               f"&split={split}&offset={offset}&length={length}")
        data = fetch_json(url)
        page_rows = [r["row"] for r in data.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        time.sleep(0.3)
    print(f"Loaded {dataset_id}/{config}/{split}: {len(rows)} rows")
    return rows


print(f"Loading {DATASET_REPO} ({DATASET_CONFIG})...")
eval_data = fetch_rows_via_datasets_server(DATASET_REPO, DATASET_CONFIG, "train")
print(f"Using split: 'train' ({len(eval_data)} documents) -- OSIRIS has no separate "
      f"test split, so this is its only split, used here purely as eval data since none "
      f"of it was ever used in this project's training data.")
print("NOTE: this corpus is small -- read the numbers below with that scale in mind.\n")

native_types_seen = set()
for row in eval_data:
    for ent in row["entities"]:
        native_types_seen.add(ent["type"])
print("Native entity types seen:", sorted(native_types_seen))
print(f"map_native_type resolves these to: "
      f"{sorted(set(filter(None, (map_native_type(t) for t in native_types_seen))))}\n")

acc = ScoreAccumulator()
n = 0
batch = []


def flush_batch():
    global n
    if not batch:
        return
    texts = [t for t, _ in batch]
    pred_lists = predict_spans_batch(texts)
    for (_, gold), preds in zip(batch, pred_lists):
        acc.add(gold, preds)
        n += 1
    batch.clear()


for row in eval_data:
    if MAX_RECORDS and n >= MAX_RECORDS:
        break
    for passage in row["passages"]:
        text = passage["text"][0] if passage["text"] else ""
        if not text.strip():
            continue
        p_start = passage["offsets"][0][0] if passage["offsets"] else 0
        for chunk_text, gold in document_to_chunks(text, p_start, row["entities"]):
            if not gold:
                continue
            batch.append((chunk_text, gold))
            if len(batch) >= BATCH_SIZE:
                flush_batch()
flush_batch()

print(f"Done: {n} records.\n")
acc.report(MODEL_NAME, in_scope_types=IN_SCOPE_TYPES, out_dir=OUT_DIR)
