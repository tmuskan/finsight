"""
inspect_fin_ner.py — Quick look at the tner/fin financial NER dataset.

The tner/fin repo ships an old Python loader script that datasets v5 won't run.
We bypass it by pointing load_dataset at the auto-converted Parquet files,
which HF Hub generates and exposes at a stable URL.

Reports:
- size of each split
- BIO label inventory
- per-label token distribution
- annotated example sentences
"""

from collections import Counter
from datasets import load_dataset


DATASET_ID = "tner/fin"

# Auto-converted Parquet files. The 'refs/convert/parquet' branch is created
# automatically by HF for every dataset; this is the standard escape hatch.
DATA_FILES = {
    "train":      f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/train/0000.parquet",
    "validation": f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/validation/0000.parquet",
    "test":       f"hf://datasets/{DATASET_ID}@refs/convert/parquet/fin/test/0000.parquet",
}


# The tner/fin label2id mapping, copied from its README.
# Order matters — index = label id.
LABEL_LIST = [
    "O",         # 0
    "B-PER",     # 1
    "I-PER",     # 2
    "B-LOC",     # 3
    "I-LOC",     # 4
    "B-ORG",     # 5
    "I-ORG",     # 6
    "B-MISC",    # 7
    "I-MISC",    # 8
]
ID2LABEL = {i: name for i, name in enumerate(LABEL_LIST)}


def show_split(ds, name: str) -> None:
    print(f"\n--- {name} ({len(ds)} sentences) ---")
    total_tokens = sum(len(row["tokens"]) for row in ds)
    print(f"Tokens: {total_tokens:,} (avg {total_tokens / len(ds):.1f}/sentence)")

    label_counts = Counter()
    for row in ds:
        for tag_id in row["tags"]:
            label_counts[ID2LABEL[tag_id]] += 1

    print("Tag distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        print(f"  {label:8s}: {count:6,d}  ({pct:5.2f}%)")


def show_examples(ds, n: int = 3) -> None:
    print("\nExample annotations from train:")
    for i, row in enumerate(ds):
        if i >= n:
            break
        print(f"\n  [{i}]")
        for tok, tag_id in zip(row["tokens"], row["tags"]):
            label = ID2LABEL[tag_id]
            marker = "  " if label == "O" else f" {label}"
            print(f"      {tok:20s}{marker}")


if __name__ == "__main__":
    print(f"Loading {DATASET_ID} from Parquet files...")
    ds = load_dataset("parquet", data_files=DATA_FILES)

    print(f"\nSplits: {list(ds.keys())}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"\nLabel inventory ({len(ID2LABEL)} labels):")
    for i, name in ID2LABEL.items():
        print(f"  {i}: {name}")

    for split_name in ds.keys():
        show_split(ds[split_name], split_name)

    show_examples(ds["train"], n=3)