"""
inspect_phrasebank.py — Quick look at the Financial PhraseBank dataset.

We use atrost/financial_phrasebank — a Parquet mirror of the
sentences_50agree variant, pre-split 64/16/20 into train/validation/test
following the FinBERT paper's convention. The original
takala/financial_phrasebank uses a legacy loader script incompatible
with datasets v4+.

This script reports the splits, label distribution, and shows a few
sample sentences per label.
"""

from collections import Counter
from datasets import load_dataset


DATASET_ID = "atrost/financial_phrasebank"


def show_split(ds, split_name: str) -> None:
    print(f"\n--- {split_name} ({len(ds)} examples) ---")
    label_names = ds.features["label"].names
    counts = Counter(ds["label"])
    print("Label distribution:")
    for label_id, count in sorted(counts.items()):
        pct = 100 * count / len(ds)
        print(f"  {label_id} ({label_names[label_id]:8s}): {count:5d}  ({pct:5.1f}%)")


def show_samples(ds) -> None:
    """One example per label from train."""
    label_names = ds.features["label"].names
    print("\nSample sentences from train split:")
    seen = set()
    for row in ds:
        name = label_names[row["label"]]
        if name not in seen:
            seen.add(name)
            print(f"  [{name:8s}] {row['sentence'][:140]}")
        if len(seen) == 3:
            break


if __name__ == "__main__":
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    print(f"\nSplits: {list(ds.keys())}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"Label feature: {ds['train'].features['label']}")

    for split_name in ds.keys():
        show_split(ds[split_name], split_name)

    show_samples(ds["train"])