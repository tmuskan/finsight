"""
retrieval.py — High-level semantic search API for FinSight.

Loads the vector store (from HF Hub if not present locally), wraps queries
in a clean interface, and returns structured search hits. This is what the
FastAPI backend and the multi-agent system consume.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from vector_store import VectorStore


DEFAULT_STORE_REPO = "musk1209/finsight-vectorstore"
DEFAULT_LOCAL_STORE_PATH = Path("data/chroma_db")


@dataclass
class SearchHit:
    """One retrieval result."""
    text: str
    ticker: str
    company_name: str
    form: str                    # "10-K" or "10-Q"
    filing_date: str             # ISO date
    accession: str
    chunk_index: int
    distance: float              # lower = more relevant
    chunk_id: str

    @property
    def source_label(self) -> str:
        """One-line human-readable citation."""
        return f"{self.ticker} {self.form} ({self.filing_date}) chunk#{self.chunk_index}"


class Retriever:
    """
    High-level retrieval interface.

    Usage:
        r = Retriever()                       # auto-downloads store from HF Hub
        hits = r.search("query", top_k=10)
        for hit in hits:
            print(hit.source_label, hit.text[:200])
    """

    def __init__(
        self,
        store_repo: str = DEFAULT_STORE_REPO,
        local_path: Path = DEFAULT_LOCAL_STORE_PATH,
        hf_token: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            store_repo: HF Hub dataset repo containing the ChromaDB files.
            local_path: Where to keep a local copy of the store.
            hf_token: HF token (falls back to HF_TOKEN env var).
            device: 'cuda', 'cpu', or 'auto'.
        """
        hf_token = hf_token or os.getenv("HF_TOKEN")

        # Fetch the store from HF Hub if we don't already have a local copy.
        # snapshot_download is idempotent — re-runs are cached.
        if not local_path.exists() or not any(local_path.iterdir()):
            print(f"Vector store not found locally. Downloading from {store_repo} ...")
            local_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=store_repo,
                repo_type="dataset",
                local_dir=str(local_path),
                token=hf_token,
            )
            print(f"Downloaded to: {local_path}")
        else:
            print(f"Using local vector store at: {local_path}")

        self.store = VectorStore(db_path=local_path, device=device)
        print(f"Loaded {self.store.count():,} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
    ) -> list[SearchHit]:
        """
        Semantic search with optional metadata filters.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            ticker: If set, restrict to filings from this company (e.g. "AAPL").
            form: If set, restrict to a form type ("10-K" or "10-Q").

        Returns:
            List of SearchHit objects, ordered best -> worst.
        """
        # Build ChromaDB metadata filter
        where: dict = {}
        if ticker:
            where["ticker"] = ticker
        if form:
            where["form"] = form
        # ChromaDB expects None (not {}) if no filter
        where_arg = where if where else None

        raw_hits = self.store.search(query, top_k=top_k, where=where_arg)

        return [
            SearchHit(
                text=hit["text"],
                ticker=hit["metadata"]["ticker"],
                company_name=hit["metadata"]["company_name"],
                form=hit["metadata"]["form"],
                filing_date=hit["metadata"]["filing_date"],
                accession=hit["metadata"]["accession"],
                chunk_index=int(hit["metadata"]["chunk_index"]),
                distance=hit["distance"],
                chunk_id=hit["id"],
            )
            for hit in raw_hits
        ]


if __name__ == "__main__":
    # Local smoke test — will download the store if not present.
    from dotenv import load_dotenv
    load_dotenv()

    retriever = Retriever()

    queries = [
        ("What are the biggest cybersecurity risks facing tech companies?", None),
        ("How is Tesla's margin evolving?", "TSLA"),
        ("Bank capital requirements", None),
    ]

    for query, ticker in queries:
        print(f"\n{'=' * 70}")
        filter_desc = f"[ticker={ticker}]" if ticker else "[no filter]"
        print(f"Query: {query}  {filter_desc}")
        print("=" * 70)

        hits = retriever.search(query, top_k=3, ticker=ticker)
        for i, hit in enumerate(hits):
            print(f"\nRank {i+1}  distance={hit.distance:.4f}  {hit.source_label}")
            print(f"  {hit.text[:250]}...")