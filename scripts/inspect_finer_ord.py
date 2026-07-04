"""
inspect_finer_ord.py — Quick look at gtfintechlab/finer-ord-bio.

Financial news NER dataset, BIO format, 3 entity types (PER, ORG, LOC).
Source: Shah et al., 2024, "FiNER-ORD: Financial Named Entity Recognition
Open Research Dataset" (arXiv 2302.11157).
"""

from collections import Counter
from datasets import load_dataset

DATASET_ID = "gtfintechlab/finer-ord-bio"


def show_split(ds, name: str, id2label: dict) -> None:
    print(f"\n--- {name} ({len(ds)} sentences) ---")

    total_tokens = sum(len(row["tokens"]) for row in ds)
    print(f"Tokens: {total_tokens:,} (avg {total_tokens / len(ds):.1f}/sentence)")

    label_counts = Counter()
    for row in ds:
        for tag_id in row["tags"]:
            label_counts[id2label[tag_id]] += 1

    print("Tag distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        print(f"  {label:8s}: {count:6,d}  ({pct:5.2f}%)")


def show_examples(ds, id2label: dict, n: int = 3) -> None:
    print("\nExample annotations from train:")
    for i, row in enumerate(ds):
        if i >= n:
            break
        print(f"\n  [{i}]")
        for tok, tag_id in zip(row["tokens"], row["tags"]):
            label = id2label[tag_id]
            marker = "  " if label == "O" else f" {label}"
            print(f"      {tok:20s}{marker}")


if __name__ == "__main__":
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    print(f"\nSplits: {list(ds.keys())}")
    print(f"Features: {ds['train'].features}")

    # tags is a Value(int) not a ClassLabel — no built-in names.
    # Print raw ID distribution + a couple of example sentences to figure
    # out the schema by inspection.
    print("\nRaw tag ID distribution (train):")
    from collections import Counter
    id_counts = Counter()
    for row in ds["train"]:
        for tag_id in row["tags"]:
            id_counts[tag_id] += 1
    for tag_id in sorted(id_counts):
        print(f"  {tag_id}: {id_counts[tag_id]:6,d} tokens")

    print("\nFirst 3 sentences (tokens + raw tag IDs):")
    for i in range(3):
        row = ds["train"][i]
        for tok, tag_id in zip(row["tokens"], row["tags"]):
            marker = "  " if tag_id == 0 else f" [{tag_id}]"
            print(f"      {tok:20s}{marker}")
        print()