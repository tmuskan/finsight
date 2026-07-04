"""
train_ner.py — Fine-tune bert-base-uncased on the tner/fin financial NER dataset.

Token-level classification with the BIO tagging scheme over 9 labels:
  O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC

Designed to run on a Kaggle notebook with a T4 GPU. Training takes ~3-5 minutes
for 5 epochs on this small dataset (~1000 training sentences).

Outputs to OUTPUT_DIR:
    config.json, model.safetensors, tokenizer files   — trained model
    test_results.json                                  — entity-level metrics
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Custom entity-level scorer (seqeval-equivalent, but seqeval doesn't
# build on Python 3.12). See ner_metrics.py for details.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ner_metrics import entity_scores, format_report


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "bert-base-uncased"
# DATASET_ID    = "tner/fin"
DATASET_ID    = "gtfintechlab/finer-ord-bio"
OUTPUT_DIR    = Path("models/finsight-ner")

# Hyperparameters — NER on small datasets benefits from more epochs
SEED          = 42
LEARNING_RATE = 3e-5         # slightly higher than sentiment; small dataset, want faster adaptation
BATCH_SIZE    = 16
NUM_EPOCHS    = 4            # more epochs than Phase 2 — only 1018 training sentences
MAX_LENGTH    = 192          # tner/fin sentences are short (~36 tokens) but allow slack for subwords
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

# The tner/fin label mapping. Order = id.
# LABEL_LIST = [
#     "O", "B-PER", "I-PER", "B-LOC", "I-LOC",
#     "B-ORG", "I-ORG", "B-MISC", "I-MISC",
# ]
# FiNER-ORD label mapping. Verified by inspection (see inspect_finer_ord.py).
# The dataset stores tags as plain integers, so we encode the mapping here.
LABEL_LIST = [
    "O",         # 0
    "B-PER",     # 1
    "I-PER",     # 2
    "B-LOC",     # 3
    "I-LOC",     # 4
    "B-ORG",     # 5
    "I-ORG",     # 6
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Auto-converted Parquet files — bypass the broken loader script
# DATA_FILES = {
#     "train":      f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/train/0000.parquet",
#     "validation": f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/validation/0000.parquet",
#     "test":       f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/test/0000.parquet",
# }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_and_align(examples, tokenizer):
    """
    Tokenize word-level inputs into BERT subwords AND realign labels.

    Input batch has parallel arrays:
        examples["tokens"] : list of list of word strings
        examples["tags"]   : list of list of int label IDs

    Output adds:
        labels : list of list of int label IDs aligned to subwords,
                 with -100 for "ignore this subword in the loss"
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,          # input is already a list of words
        truncation=True,
        max_length=MAX_LENGTH,
        # No padding here — dynamic padding via DataCollator below
    )

    aligned_labels = []
    for batch_idx, word_tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        previous_word = None
        labels = []
        for word_id in word_ids:
            if word_id is None:
                # Special tokens like [CLS], [SEP], pad
                labels.append(-100)
            elif word_id != previous_word:
                # First subword of a new word: keep the word's label
                labels.append(word_tags[word_id])
            else:
                # Continuation subwords: mark as ignore
                labels.append(-100)
            previous_word = word_id
        aligned_labels.append(labels)

    tokenized["labels"] = aligned_labels
    return tokenized


def make_compute_metrics_fn():
    """Closure for HuggingFace Trainer: entity-level P/R/F1 via ner_metrics."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Drop -100 (special tokens, subword continuations); map IDs -> strings
        true_labels = [
            [ID2LABEL[l] for l in label_seq if l != -100]
            for label_seq in labels
        ]
        pred_labels = [
            [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
            for pred_seq, label_seq in zip(preds, labels)
        ]

        scores = entity_scores(true_labels, pred_labels)
        return {
            "precision": scores["precision"],
            "recall":    scores["recall"],
            "f1":        scores["f1"],
        }
    return compute_metrics


def main() -> None:
    set_seed(SEED)

    print(f"Loading {DATASET_ID} ...")
    # raw_dataset = load_dataset("parquet", data_files=DATA_FILES)
    raw_dataset = load_dataset(DATASET_ID)
    for split_name in raw_dataset:
        print(f"  {split_name:11s}: {len(raw_dataset[split_name])} sentences")

    print(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("\nTokenizing + aligning labels...")
    tokenized = raw_dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "tags"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = make_compute_metrics_fn()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_RATIO,                    # float = ratio in transformers v5
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        seed=SEED,
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")
    trainer.train()

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("\nTest set results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key:30s} {value:.4f}")

    # Per-class breakdown using our custom entity-level scorer
    print("\nPer-class breakdown (test set):")
    predictions = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    true_labels = [
        [ID2LABEL[l] for l in label_seq if l != -100]
        for label_seq in labels
    ]
    pred_labels = [
        [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]

    scores = entity_scores(true_labels, pred_labels)
    report_str = format_report(scores)
    print(report_str)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    (OUTPUT_DIR / "test_results.json").write_text(json.dumps(test_results, indent=2))
    (OUTPUT_DIR /"ner_report.txt").write_text(report_str)
    print(f"\nModel + metrics saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()