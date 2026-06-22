"""
extract_text.py — Convert SEC EDGAR HTML filings to plain text.

Modern SEC filings are inline XBRL (iXBRL): a hybrid of XML metadata
and human-readable HTML. The XBRL elements describe what financial
data items mean under US GAAP, but they're useless as natural language.

This module strips XBRL/metadata cruft and returns clean prose suitable
for downstream NLP tasks.
"""

import re
import warnings
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress the noisy "parsing XML as HTML" warning — we know.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# XML/XBRL namespaces commonly used in SEC filings.
# We strip every tag whose name starts with one of these prefixes.
XBRL_NAMESPACES = (
    "ix:",        # inline XBRL
    "xbrli:",     # XBRL instance
    "xbrldi:",    # XBRL dimensions
    "link:",      # XBRL link
    "us-gaap:",   # US GAAP taxonomy
    "dei:",       # SEC entity tags
    "iso4217:",   # currency codes
    "srt:",       # SEC reporting taxonomy
)

# Hidden elements that contain only metadata, never visible text.
HIDDEN_TAGS = ("ix:hidden", "ix:references", "ix:resources")


def _is_xbrl_tag(tag) -> bool:
    """Return True if a tag is an XBRL/metadata element."""
    if not tag.name:
        return False
    name = tag.name.lower()
    return any(name.startswith(ns) for ns in XBRL_NAMESPACES)


def extract_text(html_path: Path) -> str:
    """
    Read an SEC HTML filing and return its plain visible text.

    Args:
        html_path: Path to a downloaded .html filing.

    Returns:
        The filing's text content with XBRL stripped and whitespace normalized.
    """
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    # 1. Remove browser scaffolding
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # 2. Remove iXBRL hidden blocks entirely
    for hidden in HIDDEN_TAGS:
        for tag in soup.find_all(hidden):
            tag.decompose()

    # 3. Unwrap XBRL inline tags — keep the text inside <ix:nonFraction>123</ix:nonFraction>
    #    but discard the tag itself so downstream sees plain text.
    for tag in soup.find_all(_is_xbrl_tag):
        tag.unwrap()

    # 4. Flatten tables: each <tr> becomes one line, <td>s tab-separated.
    #    Doing this before get_text() keeps tabular data readable.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]  # drop empty cells
            if cells:
                rows.append("\t".join(cells))
        table.replace_with("\n".join(rows) + "\n")

    # 5. Extract text with block separators
    text = soup.get_text(separator="\n")

    # 6. Whitespace cleanup
    text = re.sub(r"[ \t]+", " ", text)        # collapse horizontal whitespace
    text = re.sub(r" *\n *", "\n", text)       # trim each line
    text = re.sub(r"\n{3,}", "\n\n", text)     # cap blank-line runs

    return text.strip()


def process_all(
    raw_dir: Path,
    processed_dir: Path,
) -> dict:
    """
    Run extract_text() across every .html filing in raw_dir,
    saving plain-text counterparts under processed_dir with .txt extensions.

    Folder structure mirrors raw_dir:
        raw_dir/AAPL/10-K_2025-10-31_...html
        processed_dir/AAPL/10-K_2025-10-31_...txt

    Args:
        raw_dir:       Root containing per-ticker subfolders of .html files.
        processed_dir: Root where .txt files will be written.

    Returns:
        Summary dict with total counts and per-company results.
    """
    summary = {"successful": [], "failed": []}

    html_files = sorted(raw_dir.rglob("*.html"))
    print(f"Found {len(html_files)} HTML filings to process")

    for i, html_path in enumerate(html_files, start=1):
        # Mirror the raw_dir/TICKER/file.html structure into processed_dir
        rel = html_path.relative_to(raw_dir)
        txt_path = (processed_dir / rel).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            text = extract_text(html_path)
            txt_path.write_text(text, encoding="utf-8")
            size_kb = txt_path.stat().st_size / 1024
            print(f"  [{i:3d}/{len(html_files)}] {rel}  ->  {size_kb:>8,.1f} KB")
            summary["successful"].append({
                "file": str(rel),
                "chars": len(text),
                "words": len(text.split()),
            })
        except Exception as e:
            print(f"  [{i:3d}/{len(html_files)}] {rel}  FAILED: {type(e).__name__}: {e}")
            summary["failed"].append({
                "file": str(rel),
                "error": f"{type(e).__name__}: {e}",
            })

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files:    {len(html_files)}")
    print(f"Successful:     {len(summary['successful'])}")
    print(f"Failed:         {len(summary['failed'])}")
    if summary["successful"]:
        total_chars = sum(s["chars"] for s in summary["successful"])
        total_words = sum(s["words"] for s in summary["successful"])
        avg_chars = total_chars / len(summary["successful"])
        print(f"Total chars:    {total_chars:,}")
        print(f"Total words:    {total_words:,}")
        print(f"Avg per filing: {avg_chars:,.0f} chars")

    return summary


if __name__ == "__main__":
    import json

    raw_dir = Path("data/raw/sec_filings")
    processed_dir = Path("data/processed/sec_filings")

    summary = process_all(raw_dir=raw_dir, processed_dir=processed_dir)

    summary_path = processed_dir / "_processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {summary_path}")