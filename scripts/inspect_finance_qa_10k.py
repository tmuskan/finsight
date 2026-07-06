"""
inspect_finance_qa_10k.py — Look at virattt/financial-qa-10K.

10K QA pairs derived from 10-K filings. Each record has:
    question, answer, context, ticker, filing (year)

We verify the format, look at length distribution, and see samples.
"""

from collections import Counter
from datasets import load_dataset


DATASET_ID = "virattt/financial-qa-10K"


def stats(arr, label):
    if not arr:
        return
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    print(f"  {label:10s}: "
          f"min={min(arr_sorted):>5}  "
          f"p50={arr_sorted[n // 2]:>5}  "
          f"p95={arr_sorted[int(n * 0.95)]:>5}  "
          f"max={max(arr_sorted):>6}")


if __name__ == "__main__":
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    print(f"\nSplits: {list(ds.keys())}")

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n--- {split_name} ({len(split)} records) ---")
        print(f"Columns: {split.column_names}")

        # Character-length distributions per field
        for col in split.column_names:
            values = [str(row.get(col, "")) for row in split]
            lengths = [len(v) for v in values]
            stats(lengths, col)

        # If there's a ticker/company column, see coverage
        for key in ["ticker", "company", "filing", "year"]:
            if key in split.column_names:
                counts = Counter(str(row[key]) for row in split)
                print(f"\n  Unique {key} values: {len(counts)}")
                top5 = counts.most_common(5)
                print(f"  Top 5 by frequency:")
                for value, count in top5:
                    print(f"    {value}: {count}")

    # Samples
    train = ds[list(ds.keys())[0]]
    print(f"\n\nFirst 3 records:")
    for i in range(min(3, len(train))):
        row = train[i]
        print(f"\n  [{i}]")
        for key, value in row.items():
            display = str(value)
            if len(display) > 300:
                display = display[:300] + "..."
            print(f"      {key}: {display}")