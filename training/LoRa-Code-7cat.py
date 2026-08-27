import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from seqeval.metrics import precision_score, recall_score, f1_score

# 7-category schema. v16 adds CADEC's real train split as new DISEASE data (informal
# patient-forum language, the diagnosed gap in DISEASE's formerly all-formal-PubMed
# training data) -- see dataset/build_lora_dataset_v16.py for the exact diff and its
# leakage/balance checks.
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

data_files = {
    "train": "/content/drive/MyDrive/MyProjectFolder/v16/merged_clinical_phi_v16.train.jsonl",
    "validation": "/content/drive/MyDrive/MyProjectFolder/v16/merged_clinical_phi_v16.validation.jsonl",
    "test": "/content/drive/MyDrive/MyProjectFolder/v16/merged_clinical_phi_v16.test.jsonl"
}

print("Loading datasets...")
dataset = load_dataset("json", data_files=data_files)

base_model_path = "/content/drive/MyDrive/MyProjectFolder/local_deberta_model_base"
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = AutoModelForTokenClassification.from_pretrained(
    base_model_path,
    num_labels=len(master_labels),
    id2label=id2label,
    label2id=label2id,
    # local_deberta_base already carries a classifier head from the project's original
    # 19-label (9-category) schema, saved locally alongside the backbone. That head is
    # about to be fully retrained anyway (see modules_to_save below), so the size
    # mismatch against this run's 15-label schema is fine to reinit over, not an error.
    ignore_mismatched_sizes=True
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=192,  # covers about the 99th percentile sentence length after subword expansion, 128 wasn't enough
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
                # continuation subword of the same word, e.g. "rafenib" in
                # "Dabrafenib" -> ["Dab", "rafenib"]. Propagate the entity type as I-X
                # instead of masking with -100. Drug names and gene/variant notation
                # get split into subwords almost every time, so masking these gave the
                # model zero signal on how to continue an entity past its first
                # subword, which showed up as erratic mid-word predictions on raw text
                # (e.g. "Piperacillin" -> B-CHEMICAL / O / I-CHEMICAL, "Doxorubicin"
                # split into two separate entities). O stays O either way. This is
                # invisible in seqeval's test F1 since eval uses the same -100 scheme,
                # but it breaks free-form raw-text inference, see LoRa-Raw.py.
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

print("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

print("\nApplying LoRA Configuration...")

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    # the classifier head started randomly initialized, not pretrained, so it needs
    # full training rather than a small LoRA delta on top of noise
    modules_to_save=["classifier"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# trainable params need to be fp32 or grad scaling blows up with
# "Attempting to unscale FP16 gradients"
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

print("Trainable parameters successfully cast to FP32. Ready for safe gradient scaling.")

# eval_loss is a bad proxy for checkpoint selection here: most tokens are "O", so a
# model can drive loss down just by being confident about "O" everywhere while barely
# improving on the rare entity classes. Selecting on entity-level F1 (seqeval, which
# scores whole spans, not per-token) is what actually reflects NER quality. This also
# doubles as the overfitting guard: load_best_model_at_end below picks the checkpoint
# with the best validation F1 across all 5 epochs, not whichever one finished last, so a
# late epoch that starts memorizing train data and drifting on validation gets discarded.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [master_labels[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [master_labels[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
output_dir = "/content/drive/MyDrive/MyProjectFolder"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,  # LoRA adapters train fine with a much higher LR than a full fine-tune would use
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training with LoRA adapters...")
trainer.train()

final_save_path = os.path.join(output_dir, "r64_alpha128_lr5e-04_v16data_7cat")
trainer.save_model(final_save_path)
print(f"\nTraining complete! LoRA adapters safely saved to {final_save_path}")
