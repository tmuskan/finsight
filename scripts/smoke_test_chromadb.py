"""
smoke_test_chromadb.py — Sanity check that ChromaDB works.

Adds 4 sentences with hand-generated dummy embeddings, then queries
with one of them. We should get that same sentence back as the top hit.

No embedding model is used here — the goal is to confirm the DB
mechanics (add, query, distance calculation) work end-to-end.
"""

import chromadb
import numpy as np


def main() -> None:
    print("Creating in-memory Chroma client...")
    client = chromadb.Client()
    # An in-memory client — data disappears when the script ends.
    # Perfect for a smoke test.

    print("Creating a collection...")
    # get_or_create_collection is idempotent — safe to re-run
    collection = client.get_or_create_collection(name="smoke_test")

    print("Adding 4 documents with hand-made embeddings...")
    # In a real system, embeddings come from a model. Here we
    # just make up 8-dim vectors that are close/far by design.
    docs = [
        "Apple reported record revenue this quarter.",
        "Microsoft announced a new cloud partnership.",
        "The board of directors will meet next week.",
        "Tesla's operating margin contracted year-over-year.",
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]

    # Fake embeddings: sentence 0 and sentence 3 are "similar" (both about earnings)
    # so we give them close vectors. Sentence 1 and 2 are unrelated.
    fake_embeddings = np.array([
        [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # earnings-shaped
        [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0],   # cloud partnership
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0],   # generic meeting
        [0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # earnings-shaped, similar to doc 0
    ]).tolist()

    collection.add(
        documents=docs,
        embeddings=fake_embeddings,
        ids=ids,
        metadatas=[{"ticker": t} for t in ["AAPL", "MSFT", "N/A", "TSLA"]],
    )

    print(f"Collection now has {collection.count()} documents\n")

    # Query using the "earnings" vector — should return docs 0 and 3
    query_embedding = [0.95, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print("Querying for the top 3 nearest documents to an 'earnings-shaped' vector...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        print(f"  Rank {i+1}: distance={dist:.4f}  ticker={meta['ticker']}")
        print(f"           {doc}")

    print("\n(expected: docs about earnings — AAPL and TSLA — should be top 2)")


if __name__ == "__main__":
    main()