"""
Scores the LoRA-fine-tuned DeBERTa NER model against BC2GM (BioCreative II Gene Mention
corpus), an open gene-mention dataset never seen during training, testing GENE recognition
specifically against real, independently-annotated text.

Loaded via spyysalo/bc2gm_corpus, a plain parquet mirror of the corpus (bigbio/blurb's
"bc2gm" config was tried first but its loader is script-based and fails outright on
current datasets versions, which no longer support script-based loading at all). The
dataset's own native label scheme is printed before use rather than assumed, since
guessing a label scheme wrong has bitten this project before.

Model loading mirrors LoRa-Score.py/LoRa-Raw.py exactly: base DeBERTa model + PEFT adapter,
merged and unloaded. ADAPTER_PATH below points at the v16 checkpoint, the 7-category model
trained on merged_clinical_phi_v16 (see dataset/build_lora_dataset_v16.py for exactly what
went into it). Change ADAPTER_PATH to review a different checkpoint instead.

Dependencies: transformers, torch, peft, datasets, matplotlib.
"""
import json
import os
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# ==========================================
# 1. PATHS -- adjust to your actual Drive layout / adapter version.
# ==========================================
BASE_MODEL_PATH = "/content/drive/MyDrive/MyProjectFolder/local_deberta_model_base"
ADAPTER_PATH = "/content/drive/MyDrive/MyProjectFolder/v16"
OUT_DIR = os.environ.get("LORA_EVAL_OUT_DIR", "/content/drive/MyDrive/MyProjectFolder")
MAX_RECORDS = int(os.environ.get("LORA_EVAL_MAX_RECORDS", "0") or 0)
BATCH_SIZE = int(os.environ.get("LORA_EVAL_BATCH_SIZE", "16") or 16)

MODEL_NAME = "LoRA-DeBERTa (r64_alpha128) on BC2GM"
DATASET_REPO = "spyysalo/bc2gm_corpus"
IN_SCOPE_TYPES = ["GENE"]  # the only schema type this dataset can meaningfully test

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
    """native_type is the entity type with any B-/I- prefix already stripped (e.g. "GENE",
    or sometimes a bare dataset with no suffix at all uses just "B"/"I" for its one and only
    type). Returns our schema type, or None if this native type is out of scope for this
    dataset's comparison (excluded from gold entirely, not merged into anything)."""
    nt = native_type.upper()
    if "GENE" in nt or "PROTEIN" in nt or nt in ("", "B", "I"):
        return "GENE"
    return None


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
    """DeBERTa's SentencePiece tokenizer includes the leading space in a word-initial
    subword's character offset, so a merged span's raw [start:end] frequently starts one
    character before the real entity (confirmed directly against real predictions: 'JAPAN'
    came back as ' JAPAN', off by exactly one leading space, in 98.9% of CoNLL-2003's
    boundary mismatches and 66.7% of BC2GM's). Trimming whitespace off both ends of the
    merged span is a pure postprocessing fix, not a retraining fix."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def predict_spans_batch(texts):
    """Same B-/I- span-merge logic as LoRa-Raw.py's predict_entities, but batched for
    throughput. LoRa-Raw.py processes one text per call, fine for a live typing loop but
    far too slow for scoring thousands of records, the same class of fix already applied
    to GLiNER and the specialist ensemble elsewhere in this project."""
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
# 3. DETOKENIZE + GOLD SPAN EXTRACTION (generic, dataset-agnostic)
# ==========================================
_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%"}
_NO_SPACE_BEFORE_PREFIXES = ("'",)
_NO_SPACE_AFTER = {"(", "[", "{"}


def detokenize_with_offsets(tokens):
    parts, offsets, pos, prev_token = [], [], 0, None
    for i, tok in enumerate(tokens):
        needs_space = i > 0
        if needs_space and (tok in _NO_SPACE_BEFORE or tok.startswith(_NO_SPACE_BEFORE_PREFIXES)):
            needs_space = False
        if needs_space and prev_token in _NO_SPACE_AFTER:
            needs_space = False
        if needs_space:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(tok)
        pos += len(tok)
        offsets.append((start, pos))
        prev_token = tok
    return "".join(parts), offsets


def _split_tag(tag):
    """Handles both "B-TYPE"/"I-TYPE" and bare "B"/"I" schemes (some Hub mirrors, e.g. the
    species-800 promptsource one, use plain "B"/"I" with no dash since there is only one
    entity type). A tag that matches neither form is treated as a fresh single-token B
    rather than silently continuing whatever span came before it."""
    if tag.startswith("B-"):
        return "B", tag[2:]
    if tag.startswith("I-"):
        return "I", tag[2:]
    if tag == "B":
        return "B", ""
    if tag == "I":
        return "I", ""
    return "B", tag


def extract_mapped_gold_spans(native_tag_names, text, offsets):
    """native_tag_names: list of strings like "O", "B-GENE", "I-GENE" (already decoded from
    the dataset's own label scheme). Merges consecutive same-mapped-type tags into spans;
    out-of-scope native types (map_native_type returns None) contribute nothing to gold."""
    spans = []
    cur = None  # {"type": mapped_type, "start_tok": i}

    def flush(end_idx):
        nonlocal cur
        if cur:
            cs = offsets[cur["start_tok"]][0]
            ce = offsets[end_idx - 1][1]
            spans.append({"type": cur["type"], "start": cs, "end": ce, "text": text[cs:ce]})
        cur = None

    for i, tag in enumerate(native_tag_names):
        if tag == "O":
            flush(i)
            continue
        prefix, native_type = _split_tag(tag)
        mapped = map_native_type(native_type)
        if mapped is None:
            flush(i)
            continue
        if prefix == "B" or cur is None or cur["type"] != mapped:
            flush(i)
            cur = {"type": mapped, "start_tok": i}
    flush(len(native_tag_names))
    return spans


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
    """Same type + any character overlap, not exact boundaries. Precision and recall use
    separate numerators, each counted once per item. Sharing a single tp count between
    them lets a fragmenting model get one gold entity "recalled" multiple times, inflating
    recall past the true number of gold entities."""
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
    """Pairs a gold span with a prediction sharing the exact same boundary, regardless of
    type, so a "right place, wrong type" mistake shows up directly instead of as an
    anonymous fp+fn pair. Each gold and each prediction claim at most one pairing here,
    mirroring the 1:1 matching score_predictions_strict/collapsed already enforce."""
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
# 5. LOAD DATASET, CONVERT, RUN
# ==========================================
def _load_dataset_with_retry(repo_id, retries=5, rate_limit_wait=90):
    # HF's public API caps unauthenticated requests at 500 per 300s window, easy to hit
    # when running several of these eval scripts back to back (each downloads dataset
    # info + parquet files). This doesn't change what gets loaded, only retries a
    # transient 429 instead of dying on it -- see the 429 error this project actually hit.
    last_err = None
    for attempt in range(retries):
        try:
            return load_dataset(repo_id)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = rate_limit_wait if "429" in str(e) else 10 * (attempt + 1)
                print(f"  load_dataset({repo_id!r}) failed: {e.__class__.__name__} -- "
                      f"retrying in {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
    raise last_err


print(f"Loading {DATASET_REPO}...")
dataset = _load_dataset_with_retry(DATASET_REPO)
print("Available splits:", list(dataset.keys()))
test_split = "test" if "test" in dataset else list(dataset.keys())[-1]
test_data = dataset[test_split]
print(f"Using split: {test_split!r} ({len(test_data)} records)")

tag_feature = test_data.features["ner_tags"].feature
native_label_names = tag_feature.names
print("Native label scheme:", native_label_names)
print(f"map_native_type resolves these to: "
      f"{sorted(set(map_native_type(t[2:]) for t in native_label_names if t != 'O'))}\n")

acc = ScoreAccumulator()
n = 0
batch = []  # list of (text, gold_spans)


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
    if n % (BATCH_SIZE * 20) < BATCH_SIZE:
        print(f"  ...{n} records scored")


for i, row in enumerate(test_data):
    if MAX_RECORDS and i >= MAX_RECORDS:
        break
    tokens = row["tokens"]
    if not tokens:
        continue
    native_tags = [native_label_names[t] for t in row["ner_tags"]]
    text, offsets = detokenize_with_offsets(tokens)
    gold = extract_mapped_gold_spans(native_tags, text, offsets)
    batch.append((text, gold))
    if len(batch) >= BATCH_SIZE:
        flush_batch()
flush_batch()

print(f"\nDone: {n} records.\n")
acc.report(MODEL_NAME, in_scope_types=IN_SCOPE_TYPES, out_dir=OUT_DIR)
