"""
upload_to_hf.py — Push processed SEC filings to Hugging Face Hub as a dataset.

This script reads every .txt file under data/processed/sec_filings/,
attaches structured metadata to each, and uploads the whole thing as
a Hugging Face dataset repo. After upload, any environment with the
`datasets` library can pull it with:

    from datasets import load_dataset
    ds = load_dataset("musk1209/finsight-sec-filings")

This is the bridge from local development to Kaggle compute.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not in .env")

# These define the dataset's identity on HF Hub.
HF_USERNAME = "musk1209"
DATASET_NAME = "finsight-sec-filings"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

PROCESSED_DIR = Path("data/processed/sec_filings")
COMPANIES_CONFIG = Path("configs/companies.json")


# Filenames look like: 10-K_2025-10-31_0000320193-25-000079.txt
FILENAME_RE = re.compile(r"^(10-[KQ])_(\d{4}-\d{2}-\d{2})_(.+)\.txt$")


def parse_filename(filename: str) -> dict:
    """Extract form, filing_date, accession from our naming convention."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    return {
        "form": m.group(1),
        "filing_date": m.group(2),
        "accession": m.group(3),
    }


def build_jsonl_dataset(processed_dir: Path, output_path: Path) -> int:
    """
    Combine every .txt filing into a single JSONL file — one record per filing,
    each with text + metadata. JSONL is the natural format for HF Datasets.

    Returns the number of records written.
    """
    companies = {c["ticker"]: c for c in json.loads(COMPANIES_CONFIG.read_text())}

    records = []
    for txt_path in sorted(processed_dir.rglob("*.txt")):
        ticker = txt_path.parent.name           # parent folder = ticker
        company = companies.get(ticker, {})

        try:
            parsed = parse_filename(txt_path.name)
        except ValueError as e:
            print(f"  skip: {e}")
            continue

        text = txt_path.read_text(encoding="utf-8")

        records.append({
            "ticker": ticker,
            "cik": company.get("cik", ""),
            "company_name": company.get("name", ""),
            "form": parsed["form"],
            "filing_date": parsed["filing_date"],
            "accession": parsed["accession"],
            "text": text,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "source_filename": txt_path.name,
        })

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def build_readme(num_records: int) -> str:
    """Generate the dataset card (README.md on HF Hub)."""
    return f"""---
license: mit
language: en
task_categories:
  - question-answering
  - text-classification
tags:
  - finance
  - SEC
  - 10-K
  - 10-Q
  - EDGAR
size_categories:
  - n<1K
---

# FinSight — SEC EDGAR Filings

Cleaned plain-text 10-K (annual) and 10-Q (quarterly) filings from the
US SEC EDGAR system for 20 large publicly-traded companies across 6 sectors.

Created as part of the [FinSight](https://github.com/tmuskan/finsight) project —
a financial research AI assistant combining BERT fine-tuning, RAG, and
multi-agent systems.

## Stats
- **Records:** {num_records}
- **Companies:** 20 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, BAC, GS,
  JNJ, PFE, UNH, WMT, PG, KO, MCD, XOM, CVX, CAT)
- **Forms:** 10-K, 10-Q
- **Source:** [SEC EDGAR](https://www.sec.gov/edgar)
- **Generated:** {datetime.utcnow().strftime('%Y-%m-%d')}

## Schema
| Field           | Type   | Description                                |
|-----------------|--------|--------------------------------------------|
| ticker          | string | Stock ticker, e.g. 'AAPL'                  |
| cik             | string | SEC Central Index Key                      |
| company_name    | string | Full company name                          |
| form            | string | '10-K' or '10-Q'                           |
| filing_date     | string | ISO date, e.g. '2025-10-31'                |
| accession       | string | SEC accession number                       |
| text            | string | Plain text of filing (XBRL stripped)       |
| n_chars         | int    | Character count                            |
| n_words         | int    | Word count                                 |
| source_filename | string | Original filename                          |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}", split="train")
print(ds[0]['ticker'], ds[0]['form'], ds[0]['filing_date'])
```

## Notes
SEC filings are US Government public-domain documents.
The text has been extracted from raw HTML and stripped of XBRL/iXBRL
metadata, then whitespace-normalized. Tables are flattened to
tab-separated rows.
"""


if __name__ == "__main__":
    # 1. Build a single JSONL file from all the .txt files
    output_jsonl = PROCESSED_DIR / "sec_filings.jsonl"
    print(f"Building {output_jsonl} ...")
    n = build_jsonl_dataset(PROCESSED_DIR, output_jsonl)
    size_mb = output_jsonl.stat().st_size / 1e6
    print(f"  wrote {n} records, {size_mb:.1f} MB")

    # 2. Create the HF dataset repo (no-op if it exists)
    print(f"\nCreating HF dataset repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    # 3. Upload the JSONL file
    print("Uploading JSONL...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output_jsonl),
        path_in_repo="sec_filings.jsonl",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("  done")

    # 4. Upload the README
    readme_path = PROCESSED_DIR / "_README.md"
    readme_path.write_text(build_readme(n), encoding="utf-8")
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("  done")

    print(f"\n✓ Dataset available at: https://huggingface.co/datasets/{REPO_ID}")