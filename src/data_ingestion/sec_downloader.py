"""
sec_downloader.py — Downloads SEC EDGAR filings.

SEC requires every API caller to identify themselves via a User-Agent
header. We read it from the SEC_USER_AGENT environment variable,
which is loaded from .env at the project root.

Reference: https://www.sec.gov/os/accessing-edgar-data
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import time
import json 

load_dotenv()  # reads .env into os.environ

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT not set. Add it to .env at the project root."
    )

# Headers SEC requires on every request.
# We don't set Host explicitly — requests fills it in from the URL,
# which is important because submissions and filings use different hosts.
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# SEC asks for <= 10 requests/sec. We aim for ~5 to stay safely under.
# Every request goes through _polite_get() which enforces this.
MIN_REQUEST_INTERVAL = 0.2  # seconds between requests (5 req/sec)
_last_request_time = 0.0


def _polite_get(url: str, timeout: int = 30) -> requests.Response:
    """
    Wrapper around requests.get() that enforces SEC's rate limit
    and sets the required headers.
    """
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    response = requests.get(url, headers=HEADERS, timeout=timeout)
    _last_request_time = time.time()
    response.raise_for_status()
    return response


def fetch_submissions(cik: str) -> dict:
    """
    Fetch the submissions JSON for one company from SEC EDGAR.

    Args:
        cik: Central Index Key, the company's SEC-assigned ID.
             Numeric string; will be zero-padded to 10 digits.

    Returns:
        Parsed JSON as a Python dict.
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    response = _polite_get(url, timeout=30)
    return response.json()


def list_filings(submissions: dict, form_types: tuple = ("10-K", "10-Q")) -> list[dict]:
    """
    Filter a submissions JSON for filings of specific types.

    Args:
        submissions: The dict returned by fetch_submissions().
        form_types: Tuple of SEC form types to keep. Defaults to ("10-K", "10-Q").

    Returns:
        List of filing dicts, each containing:
            form           - e.g. '10-K'
            filing_date    - e.g. '2024-11-01'
            accession      - SEC's unique ID for this filing
            primary_doc    - filename of the main document in the filing
            report_date    - the date the report covers (period of report)
    """
    recent = submissions.get("filings", {}).get("recent", {})

    # SEC stores each field as a parallel list — index N across all lists
    # describes the same filing.
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    results = []
    for i, form in enumerate(forms):
        if form in form_types:
            results.append({
                "form": form,
                "filing_date": dates[i],
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
                "report_date": report_dates[i],
            })
    return results


def download_filing(cik: str, accession: str, primary_doc: str) -> str:
    """
    Download the primary HTML document for a single SEC filing.

    SEC's archive URL is built from:
      - cik           (unpadded, just digits)
      - accession     (the dashed accession number, e.g. '0000320193-25-000079')
      - primary_doc   (filename like 'aapl-20250927.htm')

    Args:
        cik: Numeric CIK as string.
        accession: Accession number with dashes.
        primary_doc: Filename of the filing's main HTML document.

    Returns:
        The raw HTML of the filing as a string.
    """
    cik_unpadded = str(int(cik))                   # strip leading zeros
    accession_clean = accession.replace("-", "")   # remove dashes

    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_unpadded}/{accession_clean}/{primary_doc}"
    )

    response = _polite_get(url, timeout=60)
    return response.text


def save_filing(
    cik: str,
    ticker: str,
    filing: dict,
    output_dir: Path,
) -> Path:
    """
    Download a filing and save it to disk under a tidy folder layout:

        output_dir/
          {ticker}/
            {form}_{filing_date}_{accession}.html

    Args:
        cik: Numeric CIK as string.
        ticker: Company ticker (e.g. 'AAPL'), used for the folder name.
        filing: One dict from list_filings().
        output_dir: Root directory under which to save.

    Returns:
        Path to the saved file. If the file already exists, returns its
        path without re-downloading.
    """
    company_dir = output_dir / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{filing['form']}_{filing['filing_date']}_{filing['accession']}.html"
    target = company_dir / filename

    # Skip re-downloading if we already have it — makes re-running cheap.
    if target.exists() and target.stat().st_size > 0:
        return target

    html = download_filing(cik, filing["accession"], filing["primary_doc"])
    target.write_text(html, encoding="utf-8")
    return target


def download_company(
    cik: str,
    ticker: str,
    output_dir: Path,
    max_filings: int = 5,
) -> list[Path]:
    """
    Download up to `max_filings` most recent 10-K/10-Q filings for one company.

    Args:
        cik: Numeric CIK as string.
        ticker: Company ticker, used for folder name.
        output_dir: Root output directory.
        max_filings: Cap on how many filings to fetch. Defaults to 5.

    Returns:
        List of Paths to saved files.
    """
    submissions = fetch_submissions(cik)
    filings = list_filings(submissions)

    to_fetch = filings[:max_filings]
    print(f"\n[{ticker}] {submissions.get('name')} — downloading {len(to_fetch)} filings")

    saved_paths = []
    for i, filing in enumerate(to_fetch, start=1):
        path = save_filing(cik, ticker, filing, output_dir)
        size_kb = path.stat().st_size / 1024
        marker = "·" if size_kb > 1 else "!"  # tiny files are suspicious
        print(f"  [{i}/{len(to_fetch)}] {marker} {filing['form']:5s} {filing['filing_date']}  "
              f"{size_kb:>8,.1f} KB  {path.name}")
        saved_paths.append(path)

    return saved_paths


def download_all(
    companies_config: Path,
    output_dir: Path,
    max_filings_per_company: int = 5,
) -> dict:
    """
    Download filings for every company listed in the config JSON.

    Args:
        companies_config: Path to JSON file with list of
            {ticker, cik, name} dicts.
        output_dir: Where to save filings.
        max_filings_per_company: How many recent filings per company.

    Returns:
        Summary dict with totals and per-company counts.
    """
    companies = json.loads(companies_config.read_text())
    print(f"Loaded {len(companies)} companies from {companies_config}")

    summary = {
        "total_companies": len(companies),
        "successful": [],
        "failed": [],
    }

    for company in companies:
        try:
            paths = download_company(
                cik=company["cik"],
                ticker=company["ticker"],
                output_dir=output_dir,
                max_filings=max_filings_per_company,
            )
            summary["successful"].append({
                "ticker": company["ticker"],
                "filings_saved": len(paths),
            })
        except Exception as e:
            print(f"  [!] {company['ticker']} FAILED: {type(e).__name__}: {e}")
            summary["failed"].append({
                "ticker": company["ticker"],
                "error": f"{type(e).__name__}: {e}",
            })

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Companies attempted: {summary['total_companies']}")
    print(f"Successful:          {len(summary['successful'])}")
    print(f"Failed:              {len(summary['failed'])}")
    total_filings = sum(s["filings_saved"] for s in summary["successful"])
    print(f"Total filings saved: {total_filings}")
    if summary["failed"]:
        print("\nFailed companies:")
        for f in summary["failed"]:
            print(f"  {f['ticker']}: {f['error']}")

    return summary


if __name__ == "__main__":
    config_path = Path("configs/companies.json")
    output_dir = Path("data/raw/sec_filings")

    summary = download_all(
        companies_config=config_path,
        output_dir=output_dir,
        max_filings_per_company=5,
    )

    # Save the summary as JSON for reference
    summary_path = output_dir / "_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {summary_path}")