"""
pipeline.py — Top-level RAG pipeline for FinSight.

Wires the Retriever + Reranker + PromptedLLM into a single "ask" API:

    pipeline = FinSightPipeline()
    result = pipeline.ask("What are Apple's main supply chain risks?")
    print(result.answer)
    for src in result.sources:
        print(src.source_label)

This is the layer FastAPI (Phase 8) and Streamlit (Phase 9) will call.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from retrieval import Retriever, SearchHit
from reranker import Reranker
from llm import PromptedLLM


DEFAULT_SYSTEM_PROMPT_PATH = Path("configs/system_prompt.txt")


@dataclass
class RAGResponse:
    """The full response from a RAG query — answer plus its supporting sources."""
    question: str
    answer: str
    sources: list[SearchHit]                # the reranked chunks fed to the LLM
    context: str = field(default="", repr=False)   # what the LLM actually saw
    n_candidates: int = 0                   # how many chunks were retrieved before reranking


class FinSightPipeline:
    """
    Retrieval-augmented generation over the SEC filings vector store.

    Usage:
        pipeline = FinSightPipeline()
        result = pipeline.ask("How is Apple doing on cybersecurity?")
        print(result.answer)

    All heavy components load at construction time, which takes a couple
    minutes. For serving, keep one instance alive for the process lifetime.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[PromptedLLM] = None,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH,
        hf_token: Optional[str] = None,
    ):
        """
        Args are all optional — anything not passed is constructed with defaults.
        This lets callers share components (e.g., use one Retriever for both
        the pipeline AND a standalone search endpoint).
        """
        hf_token = hf_token or os.getenv("HF_TOKEN")

        print("Initializing FinSight RAG pipeline...")
        self.retriever = retriever if retriever is not None else Retriever(hf_token=hf_token)
        self.reranker = reranker if reranker is not None else Reranker()
        self.llm = llm if llm is not None else PromptedLLM(
            system_prompt_path=system_prompt_path,
            hf_token=hf_token,
        )
        print("Pipeline ready.")

    def _assemble_context(self, sources: list[SearchHit]) -> str:
        """
        Format retrieved chunks into a numbered, cite-able context block.

        Format:
            [Source 1: AAPL 10-K (2025-10-31), chunk 42]
            <text>

            [Source 2: MSFT 10-Q (2025-04-24), chunk 8]
            <text>
        """
        pieces = []
        for i, hit in enumerate(sources, start=1):
            header = f"[Source {i}: {hit.source_label}]"
            pieces.append(f"{header}\n{hit.text.strip()}")
        return "\n\n".join(pieces)

    def ask(
        self,
        question: str,
        rerank_pool_size: int = 50,
        top_k: int = 5,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
        max_new_tokens: int = 400,
    ) -> RAGResponse:
        """
        Answer a question using retrieval + reranking + LLM generation.

        Args:
            question: Natural-language question.
            rerank_pool_size: How many chunks the retriever returns before reranking.
                              50 is standard — big enough to give the reranker
                              real choices, small enough to be fast.
            top_k: How many chunks to include in the final context sent to the LLM.
                   5 is the sweet spot for our chunk sizes.
            ticker: Optional filter — restrict retrieval to one company.
            form: Optional filter — restrict to "10-K" or "10-Q".
            max_new_tokens: Cap on LLM generation length.

        Returns:
            RAGResponse with answer + sources.
        """
        # 1. Retrieve a pool of candidates from the vector store
        candidates = self.retriever.search(
            question,
            top_k=rerank_pool_size,
            ticker=ticker,
            form=form,
        )

        # 2. Rerank down to top_k
        sources = self.reranker.rerank(question, candidates, top_k=top_k)

        # 3. Assemble the context the LLM will see
        context = self._assemble_context(sources)

        # 4. Generate the answer
        answer = self.llm.answer(
            context=context,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
            n_candidates=len(candidates),
        )


if __name__ == "__main__":
    # Local smoke test. Loading everything CPU-only will be very slow
    # (~5 minutes to load Mistral). Really this should run on Kaggle.
    from dotenv import load_dotenv
    load_dotenv()

    pipeline = FinSightPipeline()

    queries = [
        "What are the main risk factors Apple faces from supply chain disruption?",
        "How is Meta investing in AI infrastructure?",
    ]

    for q in queries:
        print(f"\n{'=' * 78}")
        print(f"Q: {q}")
        print('=' * 78)

        result = pipeline.ask(q)
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources used ({len(result.sources)}):")
        for i, src in enumerate(result.sources, start=1):
            print(f"  [{i}] {src.source_label}")