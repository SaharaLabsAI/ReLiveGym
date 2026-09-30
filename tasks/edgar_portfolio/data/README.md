# EDGAR snapshot for edgar_portfolio

Source: SEC EDGAR public data endpoints (https://www.sec.gov/search-filings/edgar-application-programming-interfaces), free, no key. Fair-access policy: a declared `User-Agent` with a contact address and at most 10 requests/second — the fetch script enforces both.

```
SEC_USER_AGENT="<your name> <contact-email>" \
python -m tasks.edgar_portfolio.data.fetch_edgar        # -> data/raw/
```

Files (not shipped; ~170 MB for the 35-ticker roster):

- `raw/company_tickers.json` — the ticker → CIK map (served verbatim)
- `raw/submissions/CIK##########.json` — filing index, `filings.recent` block (1000 newest filings) with acceptance timestamps to the second
- `raw/companyfacts/CIK##########.json` — every XBRL fact, each row tagged with the accession number (`accn`) it came from
- `raw/manifest.json` — fetch time, per-ticker sizes

The replayed host serves these files cut as of the sim instant (`edgar.py`): a filing is visible once accepted; a fact is visible once its filing is. Ground truth is the same files (`EdgarStore.events`). Snapshot in use: 2026-09-16 (`manifest.json`). Re-fetching later changes nothing inside a window that has already closed, except that the SEC occasionally corrects acceptance timestamps.
