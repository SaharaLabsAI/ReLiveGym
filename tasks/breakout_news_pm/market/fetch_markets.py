"""Fetch all Polymarket markets whose life overlaps the 2026-03..07 window.

Paginates gamma-api /markets/keyset (offset pagination caps at 2,100 rows;
keyset does not). Two passes because the endpoint silently excludes closed
markets unless closed=true is passed explicitly. Output: one slim record per
market in raw/markets.jsonl, deduped by id.

Usage:
    python fetch_markets.py [--out raw/markets.jsonl]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import requests

GAMMA = "https://gamma-api.polymarket.com/markets/keyset"
WINDOW_START = "2026-03-01T00:00:00Z"
WINDOW_END = "2026-07-01T00:00:00Z"
UA = {"User-Agent": "program-engineering-dataset-build/0.1"}


def fetch_page(params: dict, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = requests.get(GAMMA, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return json.JSONDecoder(strict=False).decode(r.text)
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"gamma fetch failed after {tries} tries: {params}")


def crawl(closed: bool) -> list[dict]:
    params = {
        "limit": 100,  # server caps keyset pages at 100
        "include_tag": "true",  # tags are omitted unless asked for
        "end_date_min": WINDOW_START,  # overlap: ends after window start...
        "start_date_max": WINDOW_END,  # ...and starts before window end
    }
    if closed:
        params["closed"] = "true"
    out, cursor, pages = [], None, 0
    while True:
        if cursor:
            params["after_cursor"] = cursor
        d = fetch_page(params)
        page = d.get("markets") or []
        for m in page:
            ev = (m.get("events") or [{}])[0]
            out.append({
                "id": m["id"],
                "question": m.get("question"),
                "slug": m.get("slug"),
                "startDate": m.get("startDate"),
                "endDate": m.get("endDate"),
                "closed": m.get("closed"),
                "closedTime": m.get("closedTime"),
                "umaResolutionStatus": m.get("umaResolutionStatus"),
                "volumeNum": m.get("volumeNum"),
                "liquidityNum": m.get("liquidityNum"),
                "clobTokenIds": m.get("clobTokenIds"),
                "outcomes": m.get("outcomes"),
                "event_id": ev.get("id"),
                "event_title": ev.get("title"),
                "neg_risk": m.get("negRisk"),
                "tags": [t.get("label") for t in (m.get("tags") or [])],
            })
        pages += 1
        if pages % 50 == 0:
            print(f"  closed={closed} page {pages}: {len(out)} markets", flush=True)
        cursor = d.get("next_cursor")
        if not cursor or not page:
            return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "raw" / "markets.jsonl"))
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for closed in (True, False):
        for m in crawl(closed):
            rows[m["id"]] = m
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for m in rows.values():
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    n_closed = sum(1 for m in rows.values() if m["closed"])
    print(f"wrote {len(rows)} markets ({n_closed} closed) -> {out}")


if __name__ == "__main__":
    main()
