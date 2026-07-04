"""
reranker.py — Cross-encoder reranking for higher-quality retrieval.

Two-stage retrieval pattern:
    1. Retriever fetches top-N (~50) candidates via fast bi-encoder embeddings.
    2. Reranker rescores those candidates with a cross-encoder that sees
       query and chunk jointly, then returns the top-k best.

This typically improves Recall@k by 10-20% at the cost of ~100ms extra
latency, well worth it for financial Q&A where accuracy matters.
"""

from dataclasses import replace
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from retrieval import SearchHit


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for SearchHit lists.

    Usage:
        rr = Reranker()
        reranked = rr.rerank(query, initial_hits, top_k=10)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading cross-encoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: Optional[int] = None,
    ) -> list[SearchHit]:
        """
        Rescore and reorder hits by cross-encoder relevance.

        Args:
            query: The original query.
            hits: SearchHit list from Retriever (candidate pool).
            top_k: If set, keep only the top-k after reranking. If None,
                   return all reranked hits.

        Returns:
            SearchHit list ordered by reranker score (highest first).
            Note: `distance` field is replaced with `-score` so lower still
            means better (consistent with the rest of the pipeline).
        """
        if not hits:
            return []

        # Cross-encoder expects a list of [query, passage] pairs
        pairs = [[query, h.text] for h in hits]

        # `predict` returns a numpy array of raw logits (higher = more relevant)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Pair each hit with its new score, then sort descending
        scored = list(zip(hits, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        # Replace the distance field with -score so "lower distance = better"
        # is preserved. Uses dataclasses.replace to make a new SearchHit
        # with only that field changed.
        return [replace(hit, distance=float(-score)) for hit, score in scored]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from retrieval import Retriever

    print("Loading retriever + reranker...")
    retriever = Retriever()
    reranker = Reranker()

    test_queries = [
        ("What are the biggest cybersecurity risks facing tech companies?", None),
        ("How is Apple's research and development strategy structured?", "AAPL"),
        ("What is JPMorgan's approach to managing interest rate risk?", "JPM"),
    ]

    for query, ticker in test_queries:
        print(f"\n{'=' * 70}")
        filter_desc = f"[ticker={ticker}]" if ticker else "[no filter]"
        print(f"Query: {query}  {filter_desc}")
        print("=" * 70)

        # Retrieve 20 candidates, rerank down to 3
        initial = retriever.search(query, top_k=20, ticker=ticker)
        reranked = reranker.rerank(query, initial, top_k=3)

        # Show what changed
        initial_top_ids = [h.chunk_id for h in initial[:3]]
        reranked_ids = [h.chunk_id for h in reranked]

        print("\n--- Bi-encoder top 3 ---")
        for i, hit in enumerate(initial[:3]):
            print(f"  {i+1}. {hit.source_label}")
            print(f"     {hit.text[:180]}...")

        print("\n--- Cross-encoder top 3 (after reranking 20 candidates) ---")
        for i, hit in enumerate(reranked):
            change = "" if hit.chunk_id in initial_top_ids else " [NEW]"
            print(f"  {i+1}. {hit.source_label}{change}")
            print(f"     {hit.text[:180]}...")