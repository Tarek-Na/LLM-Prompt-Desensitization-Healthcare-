"""
Scores d4data/biomedical-ner-all (DistilBERT) against benchmark_clinical_phi.jsonl -- same
MACCROBAT clinical annotation scheme as Clinical-AI-Apollo/Medical-NER (41 native categories,
only 4 with a real equivalent here: Age, Date, Disease_disorder, Occupation -- see
sota_eval_clinical_ai_apollo.py for why).

Same task and label taxonomy as Apollo, different backbone (DistilBERT ~66M params vs
DeBERTa-v2) -- an architecture-controlled comparison on the same benchmark.

Self-contained for Colab:
    !pip install transformers torch
"""
import json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATASET_PATH = "/content/drive/MyDrive/Benchmark/benchmark_clinical_phi.jsonl"
MAX_RECORDS = int(os.environ.get("SOTA_EVAL_MAX_RECORDS", "0") or 0)

MACCROBAT_TO_SCHEMA = {
    "Age": "AGE", "Date": "DATE", "Disease_disorder": "DISEASE", "Occupation": "PROFESSION",
}


def _normalize(name):
    # Tolerates "Disease_disorder" / "Disease disorder" / "DISEASE_DISORDER" etc -- the
    # exact separator/casing used by each checkpoint's own label strings wasn't something
    # worth gambling on getting byte-exact from the model card alone.
    return name.replace("_", "").replace(" ", "").replace("-", "").lower()


_MACCROBAT_TO_SCHEMA_NORM = {_normalize(k): v for k, v in MACCROBAT_TO_SCHEMA.items()}

MODEL_NAME = "d4data/biomedical-ner-all"
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

    # Map to schema type BEFORE merging, not after (same reasoning as the obi/Stanford
    # scripts): merge on the same schema-mapped type, ignoring the B-/I- prefix entirely,
    # since subword-level BIO boundary signal has repeatedly proven unreliable across every
    # transformer NER model tested in this benchmark so far.
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
        etype = label.split("-", 1)[1] if "-" in label else label
        schema_type = _MACCROBAT_TO_SCHEMA_NORM.get(_normalize(etype), f"D4DATA_{etype}")
        if cur and cur["type"] == schema_type:
            cur["end"] = ce
        else:
            if cur:
                entities.append(cur)
            # Apollo's SentencePiece tokenizer put a token's leading space inside its own
            # offset, silently shifting entity starts left by one char. This model is
            # DistilBERT/WordPiece, which doesn't have that problem (confirmed clean here),
            # but the guard is a harmless no-op either way.
            while cs < ce and text[cs].isspace():
                cs += 1
            cur = {"start": cs, "end": ce, "type": schema_type}
    if cur:
        entities.append(cur)
    return [{"type": e["type"], "start": e["start"], "end": e["end"],
              "text": text[e["start"]:e["end"]]} for e in entities]


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    """Precision and recall are tracked with SEPARATE numerators, each counted at most once
    per item -- see the sibling scripts for why sharing a single tp count between P and R is
    wrong whenever predictions and gold aren't 1:1."""
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
    anonymous fp+fn pair. collapse=True tallies by COLLAPSED_GROUPS family instead of raw
    type, matching the COLLAPSED scoring axis."""
    key_fn = _collapsed_type if collapse else (lambda t: t)
    # Each gold and each prediction claim at most one pairing here, mirroring the 1:1
    # matching score_predictions_strict/collapsed already enforce.
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
    """Pairs a gold span with EVERY prediction that overlaps it at all, regardless of type
    -- the RELAXED axis's own matching rule, but tallied as (gold_type, pred_type) instead
    of a tp/fp/fn count, so granularity/fragmentation confusion is visible directly."""
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
    out, prev_underscore = [], False
    for ch in name.split("(")[0]:
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
    """Render a confusion matrix (in-scope gold type vs predicted type) straight to a PNG on
    disk -- no manual copy/paste from console output required. axis="strict" or "collapsed"
    only: both pair gold/pred on an EXACT span boundary, so summing a row reproduces that
    type's true gold count. "relaxed" pairs a gold span with EVERY overlapping prediction, so
    it isn't meaningful to render the same way."""
    if not _HAS_MPL:
        print(f"[confusion matrix] matplotlib not available -- skipping PNG render for {model_name}")
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
        print(f"[confusion matrix] no in-scope gold rows found -- skipping PNG render for {model_name}")
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
        ax2.set_title(f"Top predicted types with NO gold overlap at all -- {spurious_total:,} total spurious predictions",
                       fontsize=9)
        ax2.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
    else:
        ax2.axis("off")

    fig.suptitle(f"{axis.upper()} confusion (gold vs predicted type, exact span boundary) -- "
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
        self.confusion_strict = {}      # (gold_type, pred_type) -> count, exact boundary
        self.confusion_relaxed = {}     # (gold_type, pred_type) -> count, any overlap
        self.confusion_collapsed = {}   # (gold_family, pred_family) -> count, exact boundary
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
                               "CONFUSION MATRIX -- STRICT (exact span boundary, any type pairing)")
        self._print_confusion(self.confusion_relaxed,
                               "CONFUSION MATRIX -- RELAXED (any character overlap, any type pairing)")
        self._print_confusion(self.confusion_collapsed,
                               "CONFUSION MATRIX -- COLLAPSED (exact span boundary, collapsed family pairing)")
        print("=== CONFUSION MATRIX JSON (paste back to reconstruct heatmaps) ===")
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


acc = ScoreAccumulator()
t0 = time.time()
n = 0
for rec in iter_records(DATASET_PATH):
    if MAX_RECORDS and n >= MAX_RECORDS:
        break
    text = rec["text"]
    gold = rec["entities"]
    if not text.strip():
        continue
    preds = predict_spans(text)
    acc.add(gold, preds)
    n += 1
    if n % 1000 == 0:
        elapsed = time.time() - t0
        print(f"  ...{n} records scored ({n/elapsed:.1f} rec/s, {elapsed/60:.1f} min elapsed)")

print(f"\nDone: {n} records in {(time.time()-t0)/60:.1f} min.\n")
acc.report(MODEL_NAME, in_scope_types=MACCROBAT_TO_SCHEMA.values(), out_dir=os.path.dirname(DATASET_PATH))
