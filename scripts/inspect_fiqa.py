"""
inspect_fiqa.py — Look at FinGPT/fingpt-fiqa_qa before we train on it.

Reports:
- record counts
- schema (what fields does each row have)
- text length distribution (so we can size max_seq_length correctly)
- sample records so we can see the actual task format
"""

from collections import Counter
from datasets import load_dataset


DATASET_ID = "FinGPT/fingpt-fiqa_qa"


def show_samples(ds, n: int = 5) -> None:
    print(f"\nSample records:")
    for i in range(min(n, len(ds))):
        row = ds[i]
        print(f"\n  [{i}]")
        for key, value in row.items():
            display = str(value)
            if len(display) > 200:
                display = display[:200] + "..."
            print(f"      {key}: {display}")


if __name__ == "__main__":
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    print(f"\nSplits: {list(ds.keys())}")

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n--- {split_name} ({len(split)} records) ---")
        print(f"Columns: {split.column_names}")

        # Length distribution — critical for training config
        instruction_lens = [len(row.get("instruction", "")) for row in split]
        input_lens = [len(row.get("input", "")) for row in split]
        output_lens = [len(row.get("output", "")) for row in split]

        # Total prompt length = instruction + input + output
        total_lens = [i + n + o for i, n, o in zip(instruction_lens, input_lens, output_lens)]

        def stats(arr, label):
            if not arr:
                return
            arr_sorted = sorted(arr)
            n = len(arr_sorted)
            print(f"  {label:12s}: "
                  f"min={min(arr_sorted):>5}  "
                  f"p50={arr_sorted[n // 2]:>5}  "
                  f"p95={arr_sorted[int(n * 0.95)]:>5}  "
                  f"max={max(arr_sorted):>6}")

        stats(instruction_lens, "instruction")
        stats(input_lens, "input")
        stats(output_lens, "output")
        stats(total_lens, "total")

    show_samples(ds[list(ds.keys())[0]], n=3)