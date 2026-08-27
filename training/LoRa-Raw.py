import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# ==========================================
# 1. PATHS & LABELS
# ==========================================
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

# Aligned with your updated local paths
base_model_path = "/content/drive/MyDrive/MyProjectFolder/local_deberta_model_base"
adapter_path = "/content/drive/MyDrive/MyProjectFolder/v16"

# ==========================================
# 2. LOAD TOKENIZER & MODELS (FIXED)
# ==========================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    local_files_only=True
)

print("\nLoading base DeBERTa model...")
base_model = AutoModelForTokenClassification.from_pretrained(
    base_model_path,
    num_labels=len(master_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    local_files_only=True
)

print(f"Attaching adapters from: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights and preparing for pipeline...")
merged_model = model.merge_and_unload()

# Set device and model precision
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    merged_model = merged_model.half().to(device)
    print("Model successfully cast to FP16 and loaded to GPU!")
else:
    merged_model = merged_model.to(device)
    print("Running on CPU.")

merged_model.eval()

# ==========================================
# 3. RAW TEXT INFERENCE FUNCTION
#
# Printing every raw subword's own prediction (the previous version of this
# function) is exactly what produced the fragmented output seen in testing --
# e.g. "Vancomycin" -> "Van"=B-CHEMICAL, "c"=O, "omycin"=I-CHEMICAL printed as
# three separate rows, or "Ivacaftor" showing up as two disconnected
# low-confidence entities "I" / "tor". train.py now trains every subword of a
# word (not just the first) with the correct B-X/I-X label, so this function
# merges consecutive B-X/I-X subword predictions into whole entity spans and
# reads the original text back out by character offset -- never by re-joining
# subword strings (which mangles punctuation/casing/accents).
# ==========================================
def _trim_span_whitespace(text, start, end):
    """DeBERTa's SentencePiece tokenizer includes the leading space in a word-initial
    subword's character offset, so a merged span's raw [start:end] frequently starts one
    character before the real entity -- confirmed directly against real predictions: real
    entities like "JAPAN" and "Igor Shkvyrin" were coming back as " JAPAN" and " Igor
    Shkvyrin", off by exactly one leading space, in the large majority of boundary
    mismatches measured on real CoNLL-2003/BC2GM text. Trimming whitespace off both ends of
    the merged span is a pure postprocessing fix."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def predict_entities(text: str):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=192,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = merged_model(**inputs)

    probs = torch.softmax(outputs.logits[0].float(), dim=-1)
    confidences, pred_ids = probs.max(dim=-1)

    entities = []
    cur = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue  # special token ([CLS]/[SEP]/padding)
        char_start, char_end = offset_mapping[idx]
        if char_start == char_end:
            continue
        label = id2label[pred_ids[idx].item()]
        conf = confidences[idx].item()

        if label == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, ent_type = label[:2], label[2:]
        if prefix == "B-" or cur is None or cur["type"] != ent_type:
            if cur:
                entities.append(cur)
            cur = {"start": char_start, "end": char_end, "type": ent_type, "confidences": [conf]}
        else:  # I- continuing the same entity type
            cur["end"] = char_end
            cur["confidences"].append(conf)
    if cur:
        entities.append(cur)

    trimmed = []
    for e in entities:
        s, en = _trim_span_whitespace(text, e["start"], e["end"])
        if s < en:
            trimmed.append({
                "text": text[s:en],
                "type": e["type"],
                "confidence": sum(e["confidences"]) / len(e["confidences"]),
            })
    return trimmed

def print_predictions(entities):
    print("\nDETECTED CLINICAL ENTITIES & PHI:")
    if not entities:
        print("  No clinical entities or PHI detected in this text sample.")
        return
    for e in entities:
        print(f"  [{e['type']:<10}] -> \"{e['text']}\" (Confidence: {e['confidence']:.2%})")

# ==========================================
# 4. LIVE TERMINAL LOOP
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 NER MODEL LIVE INFERENCE LOOP READY")
    print("Type your clinical text below and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        try:
            user_input = input("\n📝 Enter text: ")

            if user_input.strip().lower() in ['exit', 'quit']:
                print("\nExiting live inference loop. Goodbye!")
                break

            if not user_input.strip():
                print("Empty input detected. Please enter some text.")
                continue

            predictions = predict_entities(user_input)
            print_predictions(predictions)

        except KeyboardInterrupt:
            print("\n\nLoop interrupted by user. Exiting!")
            break

