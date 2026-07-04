"""
vector_store.py — Build and query a ChromaDB vector store of SEC filing chunks.

Wraps ChromaDB behind a class so the rest of the code doesn't need to know
the details. Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) for
embeddings — the industry-standard choice for speed/quality balance.

Public methods:
    index_chunks(chunks)     — embed and store a list of Chunk objects
    search(query, top_k=10)  — semantic search returning ranked chunks
    count()                   — total docs in the collection
"""

from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import Chunk


# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
COLLECTION_NAME = "sec_filings"
DEFAULT_DB_PATH = Path("data/chroma_db")


class VectorStore:
    """Chunk storage + semantic search over ChromaDB."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
        device: str = "auto",
    ):
        """
        Args:
            db_path: Directory where ChromaDB persists to disk.
            embedding_model_name: Any sentence-transformers model on HF Hub.
            device: 'cuda', 'cpu', or 'auto' (picks GPU if available).
        """
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for embeddings
        )

        # Load embedding model. sentence-transformers picks GPU automatically
        # if available; passing 'cpu' forces CPU.
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {embedding_model_name} on {device}")
        self.embedder = SentenceTransformer(embedding_model_name, device=device)

    def count(self) -> int:
        return self.collection.count()

    def _embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode texts to embeddings. Batches for efficient GPU use."""
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,           # cosine similarity needs unit vectors
        )
        return embeddings.tolist()

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> None:
        """
        Embed a list of Chunk objects and upsert them into ChromaDB.

        Args:
            chunks: List of Chunk objects (from chunker.py).
            batch_size: How many chunks per Chroma upsert call.
                        Smaller = safer (less memory), larger = fewer round trips.
        """
        total = len(chunks)
        print(f"Indexing {total:,} chunks in batches of {batch_size}...")

        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]

            ids = [c.id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            embeddings = self._embed(texts)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            done = min(start + batch_size, total)
            print(f"  [{done:>5}/{total}]  batch upserted")

        print(f"Done. Collection now holds {self.count():,} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search: find the top_k chunks most similar to a query.

        Args:
            query: Natural-language query.
            top_k: Number of results to return.
            where: Optional metadata filter, e.g. {"ticker": "AAPL"}.

        Returns:
            List of dicts with keys: text, metadata, distance, id.
        """
        query_embedding = self._embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        # Chroma returns parallel arrays; zip them into per-result dicts
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits


if __name__ == "__main__":
    # Smoke test: index a handful of Apple chunks locally and query them.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from chunker import chunk_filing

    sample = Path("data/processed/sec_filings/AAPL/10-K_2025-10-31_0000320193-25-000079.txt")
    if not sample.exists():
        raise SystemExit(f"Missing: {sample}")

    print("Reading + chunking sample filing...")
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
    print(f"Got {len(chunks)} chunks")

    # Use a small subset to keep the smoke test fast on CPU
    store = VectorStore(db_path=Path("data/chroma_db_smoke"))
    store.index_chunks(chunks[:20], batch_size=10)

    print("\nSearching for: 'What are Apple's main risk factors?'")
    hits = store.search("What are Apple's main risk factors?", top_k=3)
    for i, hit in enumerate(hits):
        print(f"\n  Rank {i+1}  distance={hit['distance']:.4f}  id={hit['id']}")
        print(f"  {hit['text'][:250]}...")