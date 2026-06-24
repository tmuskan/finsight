"""
train_finbert.py — Fine-tune bert-base-uncased on Financial PhraseBank
for 3-class sentiment classification (negative / neutral / positive).

This script is designed to run on a Kaggle notebook with a T4 GPU.
On a T4 (16GB VRAM), training takes ~12-15 minutes for 3 epochs.

Outputs (under OUTPUT_DIR):
    config.json, model.safetensors, tokenizer files  — the trained model
    eval_results.json                                — final test metrics
    trainer_state.json                               — training history
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ────────────────────────────────────────────────────────────────────
# Configuration — change these if you want to experiment
# ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "bert-base-uncased"
DATASET_NAME  = "atrost/financial_phrasebank"
OUTPUT_DIR    = Path("models/finsight-finbert")

# Hyperparameters — defaults from the BERT paper, refined for this task
SEED          = 42
LEARNING_RATE = 2e-5
BATCH_SIZE    = 16
NUM_EPOCHS    = 3
MAX_LENGTH    = 128             # Financial PhraseBank sentences are short
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

# Labels — must match the dataset's ClassLabel feature order
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def set_seed(seed: int) -> None:
    """Set every RNG so runs are reproducible (within a single GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred) -> dict:
    """
    Called by Trainer at the end of each evaluation step.
    Returns accuracy, macro-F1, and per-class F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    # Per-class F1 so we can see if the model neglects minority classes
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,           # main metric — class-imbalance-aware
        "f1_weighted": f1_weighted,
        "f1_negative": f1_per_class[0],
        "f1_neutral":  f1_per_class[1],
        "f1_positive": f1_per_class[2],
    }


def main() -> None:
    set_seed(SEED)

    print(f"Loading dataset: {DATASET_NAME}")
    raw_dataset = load_dataset(DATASET_NAME)
    print(f"  train:      {len(raw_dataset['train']):>5}")
    print(f"  validation: {len(raw_dataset['validation']):>5}")
    print(f"  test:       {len(raw_dataset['test']):>5}")

    print(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # The classification head is fresh — its weights are random. The encoder
    # is pre-trained. This combination is exactly what makes fine-tuning work.

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            # No padding here — we use a dynamic-padding collator below.
            # That way each batch is only padded to its longest member,
            # not to MAX_LENGTH every time. Much more efficient.
        )

    print("\nTokenizing...")
    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=["sentence"])

    # Dynamic padding — pads each batch to the longest sequence in it.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,

        # Training schedule
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        # Evaluation: run on validation set at end of every epoch.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,                # keep only the best 2 checkpoints

        # Reproducibility & logging
        seed=SEED,
        logging_steps=50,
        report_to="none",                  # disable WandB etc. — keep it simple

        # GPU optimization
        fp16=torch.cuda.is_available(),    # half-precision on GPU = ~2x faster
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")
    trainer.train()

    # Evaluate the final (best) model on the TEST set — held out, only used once.
    # This is the honest measure of generalization.
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("\nTest set results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key:25s} {value:.4f}")

    # Save trained model + tokenizer to OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save metrics as JSON for the model card later
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps(test_results, indent=2))
    print(f"\nModel + metrics saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()