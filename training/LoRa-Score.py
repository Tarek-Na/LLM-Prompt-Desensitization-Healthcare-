import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from peft import PeftModel
from seqeval.metrics import classification_report, f1_score

# ==========================================
# 1. PATHS & LABELS (MATCHED TO YOUR SCRIPT)
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

# Aligned with your training paths
base_model_path = "/content/drive/MyDrive/MyProjectFolder/local_deberta_model_base"
test_file_path = "/content/drive/MyDrive/MyProjectFolder/v16/merged_clinical_phi_v16.test.jsonl"
adapter_path = "/content/drive/MyDrive/MyProjectFolder/v16"

# ==========================================
# 2. LOAD & TOKENIZE TEST DATASET
# ==========================================
print("Loading testing dataset...")
dataset = load_dataset("json", data_files={"test": test_file_path})

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=192  # must match train.py
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # Must match train.py: propagate I-X to continuation subwords of the
                # same word instead of masking with -100 (see train.py for why).
                cur_label = master_labels[label[word_idx]]
                if cur_label == "O":
                    label_ids.append(label2id["O"])
                else:
                    ent_type = cur_label.split("-", 1)[1]
                    label_ids.append(label2id[f"I-{ent_type}"])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing test dataset...")
tokenized_test_dataset = dataset["test"].map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ==========================================
# 3. LOAD MODELS & MERGE WEIGHTS
# ==========================================
print("\nLoading base DeBERTa model...")
base_model = AutoModelForTokenClassification.from_pretrained(
    base_model_path,
    num_labels=len(master_labels),
    id2label=id2label,
    label2id=label2id,
    # local_deberta_model_base still carries a classifier head sized for the old
    # 19-label (9-category) schema; harmless to reinit since PeftModel.from_pretrained
    # below replaces it with the adapter's own trained 15-label classifier anyway.
    ignore_mismatched_sizes=True
)

print(f"Attaching adapters from: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights and preparing for evaluation...")
merged_model = model.merge_and_unload()

# ⚡ Cast model to FP16 to avoid dtype mismatch errors on GPU
if torch.cuda.is_available():
    merged_model = merged_model.half().to("cuda")
    print("Model successfully cast to FP16 and loaded to GPU!")
else:
    print("Warning: Running on CPU. This will be slow.")

# ==========================================
# 4. RUN INFERENCE & EVALUATION
# ==========================================
eval_trainer = Trainer(
    model=merged_model,
    args=TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=16,
        fp16=torch.cuda.is_available(),  # Ensures inputs match model FP16 precision
        report_to="none"
    ),
    data_collator=data_collator
)

print("\nRunning predictions on test set...")
predictions, labels, _ = eval_trainer.predict(tokenized_test_dataset)
predictions = np.argmax(predictions, axis=-1)

# Clean up padding tokens (-100)
true_predictions = [
    [master_labels[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [master_labels[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# ==========================================
# 5. PRINT REPORT
# ==========================================
micro_f1 = f1_score(true_labels, true_predictions)

print("\n" + "="*75)
print(f"🏆 TEST EVALUATION COMPLETED FOR: {os.path.basename(adapter_path)}")
print("="*75)
print(f"⭐ Overall Micro F1 Score: {micro_f1:.4f}\n")
print("Classification Report:")
print(classification_report(true_labels, true_predictions))
