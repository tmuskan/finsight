"""
eval_retrieval.py — Measure retrieval quality with Recall@K and MRR.

Loads a hand-crafted set of queries with ground-truth ticker labels
from configs/eval_queries.json. For each query, runs retrieval (with and
without reranking), then measures:

    Recall@K — did any of the top-K results come from a relevant ticker?
    MRR      — reciprocal rank of the first relevant result (0 if none).

Aggregates across queries and prints a comparison table.
"""

import json
import sys
import time
from pathlib import Path

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from retrieval import Retriever
from reranker import Reranker


EVAL_QUERIES_PATH = Path("configs/eval_queries.json")
K_VALUES = [1, 3, 5, 10]      # compute Recall@K for these
CANDIDATE_POOL = 50           # bi-encoder returns this many for the reranker to work on


def recall_at_k(retrieved_tickers: list[str], relevant_tickers: set[str], k: int) -> float:
    """1 if any of top-k retrieved tickers is relevant, else 0."""
    top_k = retrieved_tickers[:k]
    return 1.0 if any(t in relevant_tickers for t in top_k) else 0.0


def reciprocal_rank(retrieved_tickers: list[str], relevant_tickers: set[str]) -> float:
    """1/rank of the first relevant retrieved ticker, or 0 if none found."""
    for i, ticker in enumerate(retrieved_tickers, start=1):
        if ticker in relevant_tickers:
            return 1.0 / i
    return 0.0


def evaluate_config(
    label: str,
    queries: list[dict],
    retriever: Retriever,
    reranker: Reranker | None,
) -> dict:
    """
    Run all queries under one config (with or without reranker) and aggregate.

    Returns:
        {
            "label": str,
            "recall@k": {k: value for k in K_VALUES},
            "mrr": float,
            "avg_latency_ms": float,
            "per_query": list of dicts with individual results,
        }
    """
    print(f"\n--- Evaluating: {label} ---")

    recalls = {k: [] for k in K_VALUES}
    rrs = []
    latencies = []
    per_query = []

    for i, q in enumerate(queries):
        query = q["query"]
        relevant = set(q["relevant_tickers"])

        t0 = time.perf_counter()
        # Always fetch a pool, then optionally rerank
        hits = retriever.search(query, top_k=CANDIDATE_POOL)
        if reranker is not None:
            hits = reranker.rerank(query, hits, top_k=max(K_VALUES))
        else:
            hits = hits[:max(K_VALUES)]
        t1 = time.perf_counter()

        retrieved_tickers = [h.ticker for h in hits]

        # Metrics for this query
        q_recalls = {k: recall_at_k(retrieved_tickers, relevant, k) for k in K_VALUES}
        q_rr = reciprocal_rank(retrieved_tickers, relevant)
        q_latency_ms = (t1 - t0) * 1000

        for k in K_VALUES:
            recalls[k].append(q_recalls[k])
        rrs.append(q_rr)
        latencies.append(q_latency_ms)

        per_query.append({
            "query": query,
            "relevant": list(relevant),
            "top_5_tickers": retrieved_tickers[:5],
            "recall@1": q_recalls[1],
            "recall@5": q_recalls[5],
            "rr": q_rr,
            "latency_ms": q_latency_ms,
        })

        found = "✓" if q_rr > 0 else "✗"
        print(f"  [{i+1:2}/{len(queries)}] {found}  R@1={q_recalls[1]:.0f} R@5={q_recalls[5]:.0f} "
              f"RR={q_rr:.3f}  {q_latency_ms:>5.0f}ms  {query[:50]}")

    return {
        "label": label,
        "recall@k": {k: sum(recalls[k]) / len(recalls[k]) for k in K_VALUES},
        "mrr": sum(rrs) / len(rrs),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "per_query": per_query,
    }


def print_comparison(results_a: dict, results_b: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {results_a['label']:>18}  {results_b['label']:>18}  {'Δ':>7}")
    print("=" * 70)
    for k in K_VALUES:
        a = results_a["recall@k"][k]
        b = results_b["recall@k"][k]
        print(f"{'Recall@'+str(k):<20} {a:>18.3f}  {b:>18.3f}  {b - a:>+7.3f}")
    mrr_a, mrr_b = results_a["mrr"], results_b["mrr"]
    print(f"{'MRR':<20} {mrr_a:>18.3f}  {mrr_b:>18.3f}  {mrr_b - mrr_a:>+7.3f}")
    lat_a, lat_b = results_a["avg_latency_ms"], results_b["avg_latency_ms"]
    print(f"{'Avg latency (ms)':<20} {lat_a:>18.0f}  {lat_b:>18.0f}  {lat_b - lat_a:>+7.0f}")
    print("=" * 70)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print(f"Loading eval queries from {EVAL_QUERIES_PATH}")
    queries = json.loads(EVAL_QUERIES_PATH.read_text())
    print(f"Loaded {len(queries)} queries")

    print("\nLoading retriever + reranker...")
    retriever = Retriever()
    reranker = Reranker()

    # Baseline: bi-encoder only, no reranking
    baseline = evaluate_config("bi-encoder only", queries, retriever, reranker=None)

    # With reranker
    with_rerank = evaluate_config("+ cross-encoder rerank", queries, retriever, reranker)

    print_comparison(baseline, with_rerank)

    # Save detailed results for the model card / README
    output = {
        "config": {
            "candidate_pool": CANDIDATE_POOL,
            "k_values": K_VALUES,
            "num_queries": len(queries),
        },
        "baseline": {k: v for k, v in baseline.items() if k != "per_query"},
        "with_rerank": {k: v for k, v in with_rerank.items() if k != "per_query"},
        "baseline_per_query": baseline["per_query"],
        "rerank_per_query": with_rerank["per_query"],
    }
    out_path = Path("data/eval_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nDetailed results saved to: {out_path}")