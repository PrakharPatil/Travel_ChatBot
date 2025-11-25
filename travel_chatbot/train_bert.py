# train_bert.py
import json
import os
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "data/bert_training_data.json"
MODEL_CHECKPOINT = "bert-base-uncased"
OUTPUT_DIR = "./models/bert_classifier"


def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_bert_classifier():
    # 1. Load raw data
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} training examples")

    # 2. Train/val split (no stratify for small datasets)
    if len(data) < 15:
        print("Warning: Very small dataset. Using 80% for training without stratification.")
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    else:
        # Stratify only if enough samples
        labels = [d["label"] for d in data]
        label_counts = {l: labels.count(l) for l in set(labels)}
        min_count = min(label_counts.values())

        if min_count >= 2:
            train_data, val_data = train_test_split(
                data, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 3. Load tokenizer & model
    tokenizer = BertTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=3,  # Conversational=0, RAG=1, API_Call=2
    )

    # 4. Tokenize datasets
    train_dataset = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer), batched=True
    )

    # Rename 'label' to 'labels' for Trainer
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # Remove text column; set format for PyTorch
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # 5. Training arguments (compatible with all transformers versions)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 9. Save model & tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"BERT classifier trained and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("Training BERT Query Classifier")
    print("=" * 50)
    train_bert_classifier()
