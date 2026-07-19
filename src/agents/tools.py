"""
tools.py — Agent tools for FinSight.

Three tools the agent can call:
    search_filings(query, ticker?, form?)  — the RAG pipeline
    compare_companies(question, tickers)   — same question, multiple companies
    list_available_filings(ticker)         — cheap metadata lookup

Tools are defined with LangChain's @tool decorator so they carry proper
schemas that the agent can introspect. The @tool decorator produces a
BaseTool object with .name, .description, .args_schema, and .invoke().

The pipeline is injected via `set_pipeline(pipeline)` so this module
doesn't own the heavy RAG stack — the notebook loads it once and
registers it here.
"""

from typing import List, Optional

from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────
# Global pipeline handle. Set by the notebook after loading the pipeline.
# ─────────────────────────────────────────────────────────────────
_pipeline = None


def set_pipeline(pipeline) -> None:
    """Register a loaded FinSightPipeline for the tools to use."""
    global _pipeline
    _pipeline = pipeline


def _require_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "No pipeline registered. Call set_pipeline(pipeline) first."
        )
    return _pipeline


# ─────────────────────────────────────────────────────────────────
# Tool 1: single-question search over filings
# ─────────────────────────────────────────────────────────────────
@tool
def search_filings(query: str, ticker: Optional[str] = None, form: Optional[str] = None) -> str:
    """
    Search SEC filings for information related to a query. Returns a
    synthesized answer with citations.

    Args:
        query: Natural-language question about a company or topic.
        ticker: Optional stock ticker (e.g. "AAPL", "MSFT") to restrict
                the search to one company.
        form: Optional form type ("10-K" or "10-Q") to restrict the search.

    Returns:
        A structured answer with citation markers like [Source 1].
    """
    pipeline = _require_pipeline()
    result = pipeline.ask(query, ticker=ticker, form=form)
    sources_list = "\n".join(
        f"[Source {i+1}] {src.source_label}" for i, src in enumerate(result.sources)
    )
    return f"{result.answer}\n\nSources:\n{sources_list}"


# ─────────────────────────────────────────────────────────────────
# Tool 2: multi-company comparison
# ─────────────────────────────────────────────────────────────────
@tool
def compare_companies(question: str, tickers: List[str]) -> str:
    """
    Run the SAME question against multiple companies' filings, one at a time.
    Use this when the user asks a comparative question across companies.

    Args:
        question: The question to ask about each company.
        tickers: List of stock tickers to compare (e.g. ["AAPL", "MSFT", "GOOGL"]).

    Returns:
        Each company's answer labeled by ticker.
    """
    pipeline = _require_pipeline()
    parts = []
    for ticker in tickers:
        result = pipeline.ask(question, ticker=ticker)
        parts.append(f"=== {ticker} ===\n{result.answer}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Tool 3: metadata lookup — no LLM/retrieval, cheap
# ─────────────────────────────────────────────────────────────────
@tool
def list_available_filings(ticker: str) -> str:
    """
    List all SEC filings available for a given ticker in our corpus.
    Use this to check what data is available before asking a specific question.

    Args:
        ticker: Stock ticker (e.g. "AAPL").

    Returns:
        Comma-separated list of filings like "10-K (2025-10-31), 10-Q (2026-05-01)".
    """
    pipeline = _require_pipeline()
    collection = pipeline.retriever.store.collection
    result = collection.get(where={"ticker": ticker}, limit=500)
    if not result["ids"]:
        return f"No filings found for ticker {ticker}."

    seen = set()
    for meta in result["metadatas"]:
        seen.add((meta["form"], meta["filing_date"]))
    filings = sorted(seen, key=lambda x: x[1], reverse=True)
    return ", ".join(f"{form} ({date})" for form, date in filings)


# ─────────────────────────────────────────────────────────────────
# Convenience: all tools in one list for the agent
# ─────────────────────────────────────────────────────────────────
ALL_TOOLS = [search_filings, compare_companies, list_available_filings]