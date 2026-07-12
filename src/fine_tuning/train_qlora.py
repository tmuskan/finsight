"""
train_qlora.py — QLoRA fine-tune Mistral 7B Instruct v0.3 for financial QA.

Uses PEFT + bitsandbytes for 4-bit quantization + LoRA adapters.
Trains on virattt/financial-qa-10K (7,000 real QA pairs from 2023 10-K filings).

Designed to run on Kaggle T4 (16GB VRAM). Expected training time: ~1-2 hours.

Outputs (under OUTPUT_DIR):
    adapter_config.json         — LoRA config
    adapter_model.safetensors   — LoRA adapter weights (~100MB)
    tokenizer files
    training_metrics.json       — final loss + eval metrics
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_ID     = "virattt/financial-qa-10K"
OUTPUT_DIR     = Path("models/finsight-qlora-mistral")

# Reproducibility
SEED           = 42

# Training hyperparameters — QLoRA defaults tuned for T4 constraints
LEARNING_RATE  = 2e-4          # LoRA tolerates higher LR than full fine-tuning
BATCH_SIZE     = 2             # Small batch to fit in 16GB VRAM
GRAD_ACCUM     = 8             # Effective batch = 2 × 8 = 16
NUM_EPOCHS     = 3
WARMUP_RATIO   = 0.03
WEIGHT_DECAY   = 0.001

# LoRA-specific hyperparameters
LORA_R         = 16            # Rank of adapter matrices — controls capacity
LORA_ALPHA     = 32            # Scaling — commonly set to 2×R
LORA_DROPOUT   = 0.05
LORA_TARGET_MODULES = [        # Which layers to attach LoRA to (attention projections)
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# Train/val split
VAL_SPLIT_FRACTION = 0.05      # 5% of data held out for validation

# ────────────────────────────────────────────────────────────────────
# Reproducibility helper
# ────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────────────
# Dataset prep
# ────────────────────────────────────────────────────────────────────
INSTRUCTION_TEMPLATE = (
    "Answer the question based on the provided context from a company's "
    "10-K filing. Be concise and factual.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)


def build_training_dataset(tokenizer):
    """
    Load, filter, split, and format virattt/financial-qa-10K.

    Applies Mistral's chat template to produce a single `text` field per row.
    Returns (train_dataset, val_dataset).
    """
    print(f"Loading {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID, split="train")
    print(f"  raw records: {len(raw)}")

    # Drop rows with any empty required field
    def is_valid(row):
        return (
            row.get("question")
            and row.get("answer")
            and row.get("context")
            and len(row["question"].strip()) > 0
            and len(row["answer"].strip()) > 0
            and len(row["context"].strip()) > 0
        )

    filtered = raw.filter(is_valid)
    print(f"  after filtering empty rows: {len(filtered)} (dropped {len(raw) - len(filtered)})")

    # Format each row using Mistral's chat template.
    # Hardcoded because transformers 5.0's tokenizer.apply_chat_template
    # has a bug where it treats a Python list of dicts as if it were
    # a Dataset object and fails with KeyError: -1.
    # Mistral's template has been stable across v0.1, v0.2, v0.3:
    #   <s>[INST] user_message [/INST] assistant_response</s>
    # We omit the leading <s> because SFTTrainer's tokenizer will
    # add it as the BOS token automatically.
    def format_row(row):
        user_message = INSTRUCTION_TEMPLATE.format(
            context=row["context"].strip(),
            question=row["question"].strip(),
        )
        answer = row["answer"].strip()
        text = f"[INST] {user_message} [/INST] {answer}</s>"
        return {"text": text}

    formatted = filtered.map(
        format_row,
        remove_columns=filtered.column_names,   # drop originals, keep only 'text'
    )

    # Deterministic 95/5 split
    split = formatted.train_test_split(test_size=VAL_SPLIT_FRACTION, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

    print(f"  train: {len(train_ds)}   val: {len(val_ds)}")
    return train_ds, val_ds

# ────────────────────────────────────────────────────────────────────
# Model prep — QLoRA setup
# ────────────────────────────────────────────────────────────────────
def build_model_and_tokenizer():
    """
    Load Mistral 7B in 4-bit and attach LoRA adapters.

    Returns (model, tokenizer) where:
        model     — Mistral-7B-Instruct-v0.3, 4-bit quantized, LoRA-wrapped
        tokenizer — matching Mistral tokenizer
    """
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Mistral doesn't ship a pad_token by default. Standard fix:
    # reuse the EOS token as pad — the attention mask keeps them
    # from affecting the loss.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For causal LMs, padding should be on the right so the model
    # sees a normal sequence, then padding after.
    tokenizer.padding_side = "right"

    print(f"Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # NormalFloat4 — best for LLMs
        bnb_4bit_compute_dtype=torch.float16,   # T4 does fp16 well, bf16 not so well
        bnb_4bit_use_double_quant=True,         # extra ~0.4 bits/param saved
    )

    print(f"Loading model in 4-bit: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                      # place layers on GPU automatically
        # `torch_dtype` is set implicitly by bnb_config; don't override
    )

    print("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print how many params are actually trainable — sanity check.
    # For Mistral 7B with our LoRA config, expect ~13-15M trainable
    # out of ~7.2B total, so ~0.2%.
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"  trainable params: {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.3f}%)")

    return model, tokenizer

# ────────────────────────────────────────────────────────────────────
# Training + save
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer()
    train_ds, val_ds = build_training_dataset(tokenizer)

    print("\nBuilding SFTTrainer...")
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # Schedule
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        # Precision + memory
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # SFT-specific — TRL 1.7 handles sequence length via tokenizer defaults;
        # our data is well under Mistral's 2048 default so no explicit cap needed.
        dataset_text_field="text",
        packing=False,                 # keep each example separate — simpler for QA
        completion_only_loss=False,    # train on full sequence

        # Logging
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\nStarting training...\n")
    train_result = trainer.train()

    # Eval on val set at the end (best model already loaded)
    print("\nFinal evaluation on validation set...")
    eval_result = trainer.evaluate()

    # Persist adapter + tokenizer
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save metrics as JSON
    import math
    metrics = {
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_result.get("eval_loss", float("nan"))),
        "eval_perplexity": float(math.exp(eval_result["eval_loss"]))
                           if "eval_loss" in eval_result else float("nan"),
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0)),
        "train_samples_per_second": float(
            train_result.metrics.get("train_samples_per_second", 0)
        ),
        "base_model": BASE_MODEL,
        "dataset": DATASET_ID,
        "seed": SEED,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "num_epochs": NUM_EPOCHS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
    }
    (OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nDone.")
    print(f"  Train loss:      {metrics['train_loss']:.4f}")
    print(f"  Eval loss:       {metrics['eval_loss']:.4f}")
    print(f"  Eval perplexity: {metrics['eval_perplexity']:.2f}")
    print(f"  Runtime:         {metrics['train_runtime_sec']/60:.1f} min")


if __name__ == "__main__":
    main()