"""Harvest the EDGAR replay substrate for edgar_portfolio.

    SEC_USER_AGENT="<name> <contact-email>" \\
    python -m tasks.edgar_portfolio.data.fetch_edgar [--out tasks/edgar_portfolio/data/raw]

One snapshot, taken today, of the three public endpoints the replayed
`sec` host serves: the ticker map, each CIK's `submissions` index (the
`filings.recent` block — accession numbers, forms, acceptance times to
the second) and each CIK's XBRL `companyfacts`. Everything the mock
serves at sim time t is a filter of these files (acceptance <= t); the
ground truth of every scorable event is the same files. Fair-access
policy: a declared User-Agent with a contact address and <= 10 req/s.
Raw files are gitignored (tasks/*/data/*); rebuild with this script.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tasks.edgar_portfolio.roster import ALL_TICKERS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_INTERVAL_S = 0.12  # < 10 req/s


def _get(url: str, ua: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": ua, "Accept-Encoding": "gzip",
        "Host": url.split("/")[2]})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "raw"))
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="subset (default: the task roster incl. quiet + "
                         "roster-change tickers)")
    args = ap.parse_args(argv)
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        sys.exit("set SEC_USER_AGENT='<name> <contact-email>' (SEC fair-access "
                 "policy requires a declared client with a contact address)")
    out = Path(args.out)
    (out / "submissions").mkdir(parents=True, exist_ok=True)
    (out / "companyfacts").mkdir(parents=True, exist_ok=True)
    tickers = args.tickers or ALL_TICKERS

    raw = _get(TICKERS_URL, ua)
    (out / "company_tickers.json").write_bytes(raw)
    by_ticker = {v["ticker"]: int(v["cik_str"]) for v in json.loads(raw).values()}
    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "user_agent_declared": True, "tickers": {}}
    for t in tickers:
        cik = by_ticker.get(t)
        if cik is None:
            print(f"!! {t}: not in company_tickers.json", file=sys.stderr)
            continue
        sizes = {}
        for kind, url in (("submissions", SUBMISSIONS_URL),
                          ("companyfacts", FACTS_URL)):
            data = _get(url.format(cik=cik), ua)
            path = out / kind / f"CIK{cik:010d}.json"
            path.write_bytes(data)
            sizes[kind] = len(data)
            time.sleep(MIN_INTERVAL_S)
        manifest["tickers"][t] = {"cik": cik, **sizes}
        print(f"{t:6s} CIK{cik:010d} submissions={sizes['submissions']:>9,d} "
              f"companyfacts={sizes['companyfacts']:>11,d}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json ({len(manifest['tickers'])} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
