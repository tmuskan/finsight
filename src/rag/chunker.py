"""
chunker.py — Split SEC filings into overlapping text chunks with metadata.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
progressively finer boundaries (paragraph -> line -> sentence -> word)
so chunks don't break in the middle of a word or sentence.

Produces `Chunk` objects: (text, metadata) pairs suitable for embedding
and storing in a vector database.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Character sizes tuned for financial-filing text.
# Rule of thumb: ~4 chars per token for English SEC prose.
CHUNK_SIZE_CHARS    = 2000    # ~500 tokens
CHUNK_OVERLAP_CHARS = 200     # ~50 tokens


# Filename pattern from Phase 1: 10-K_2025-10-31_0000320193-25-000079.txt
FILENAME_RE = re.compile(r"^(10-[KQ])_(\d{4}-\d{2}-\d{2})_(.+)\.txt$")


@dataclass
class Chunk:
    """One chunk of a filing, ready for embedding."""
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        Deterministic ID for use as a primary key in the vector store.
        Format: TICKER_FORM_DATE_ACCESSION_chunkIdx
        e.g. AAPL_10-K_2025-10-31_0000320193-25-000079_0042
        """
        return "_".join([
            self.metadata["ticker"],
            self.metadata["form"],
            self.metadata["filing_date"],
            self.metadata["accession"],
            f"{self.metadata['chunk_index']:04d}",
        ])


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        # Order matters: try paragraph breaks first, then lines, then sentences.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_filing(text: str, filing_metadata: dict) -> list[Chunk]:
    """
    Split one filing's plain text into overlapping chunks.

    Args:
        text: The full plaintext of the filing.
        filing_metadata: dict with keys: ticker, cik, company_name,
                         form, filing_date, accession.

    Returns:
        List of Chunk objects, each with its own metadata dict copy.
    """
    splitter = _make_splitter()
    pieces = splitter.split_text(text)

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        meta = dict(filing_metadata)      # per-chunk copy
        meta["chunk_index"] = i
        meta["chunk_char_count"] = len(piece)
        chunks.append(Chunk(text=piece, metadata=meta))

    return chunks


def parse_filename(filename: str) -> dict:
    """Recover form/date/accession from our Phase 1 naming convention."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename doesn't match: {filename}")
    return {
        "form": m.group(1),
        "filing_date": m.group(2),
        "accession": m.group(3),
    }


def chunk_processed_dir(
    processed_dir: Path,
    companies_config: Path,
) -> list[Chunk]:
    """
    Walk data/processed/sec_filings/ and chunk every .txt file.

    Args:
        processed_dir: Root with per-ticker subfolders of .txt filings.
        companies_config: Path to configs/companies.json.

    Returns:
        Flat list of Chunk objects across all filings.
    """
    companies = {c["ticker"]: c for c in json.loads(companies_config.read_text())}

    all_chunks: list[Chunk] = []
    for txt_path in sorted(processed_dir.rglob("*.txt")):
        ticker = txt_path.parent.name
        if ticker not in companies:
            continue                       # skip unknown

        company = companies[ticker]
        try:
            parsed = parse_filename(txt_path.name)
        except ValueError:
            continue                       # skip malformed

        meta = {
            "ticker": ticker,
            "cik": company["cik"],
            "company_name": company["name"],
            "form": parsed["form"],
            "filing_date": parsed["filing_date"],
            "accession": parsed["accession"],
            "source_filename": txt_path.name,
        }

        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_filing(text, meta)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    # Smoke test: chunk Apple's most recent 10-K, print stats
    sample = Path("data/processed/sec_filings/AAPL/10-K_2025-10-31_0000320193-25-000079.txt")
    if not sample.exists():
        raise SystemExit(f"Sample file not found: {sample}")

    text = sample.read_text(encoding="utf-8")
    metadata = {
        "ticker": "AAPL",
        "cik": "320193",
        "company_name": "Apple Inc.",
        "form": "10-K",
        "filing_date": "2025-10-31",
        "accession": "0000320193-25-000079",
        "source_filename": sample.name,
    }

    chunks = chunk_filing(text, metadata)
    print(f"Source text: {len(text):,} chars, {len(text.split()):,} words")
    print(f"Chunks:      {len(chunks)}")

    sizes = [len(c.text) for c in chunks]
    print(f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={sum(sizes)/len(sizes):.0f} chars")

    print(f"\nFirst chunk id:  {chunks[0].id}")
    print(f"First chunk preview:\n---\n{chunks[0].text[:400]}...\n---")
    print(f"\nLast chunk id:   {chunks[-1].id}")
    print(f"Last chunk preview:\n---\n...{chunks[-1].text[-400:]}\n---")