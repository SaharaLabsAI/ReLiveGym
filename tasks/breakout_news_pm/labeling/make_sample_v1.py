#!/usr/bin/env python3
"""Freeze the v1 hindsight-labeling sample.

Stratified roster: 50 markets per domain across 7 swm-style domains, from
markets with >=1 detected breakpoint (post-warmup detection, 2026-07-23).
Volume floors: $250k everywhere except Politics at $150k (only 31 Politics
markets clear $250k; at $150k exactly 51 qualify and the $130-190k band is
genuinely news-driven — decided 2026-07-23).

Selection is a deterministic seeded random sample per domain (not top-by-
volume, to avoid biasing the sample toward mega-markets). Output is one
JSON line per market with its full breakpoint records, plus the market
`description` (resolution criteria) fetched from the Gamma API for sampled
markets only (--no-fetch-descriptions to skip).

Usage:
  python3 make_sample_v1.py [--data-dir ../market] [--out sample_v1.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "market"
GAMMA = "https://gamma-api.polymarket.com/markets/{id}"
UA = {"User-Agent": "program-engineering-dataset-build/0.1"}

PER_DOMAIN = 50
SEED = 42
FLOORS = {"Politics": 150_000}
DEFAULT_FLOOR = 250_000

BUCKETS = [
    ("Elections", {"Elections", "US Election", "Global Elections", "World Elections",
                   "Main Election", "Primaries", "Midterms", "Nov 4 Elections",
                   "House Elections", "primary elections"}),
    ("Crypto", {"Crypto", "Crypto Prices", "FDV", "Bitcoin", "Ethereum", "Solana",
                "Memecoins", "Stablecoins"}),
    ("Finance", {"Finance", "Economy", "Business", "Commodities", "Oil", "Fed Rates",
                 "Stocks", "Inflation", "Fed", "Treasuries", "Gold", "Earnings",
                 "Pre-Market", "Hit Price", "Monthly"}),
    ("Tech/AI", {"Tech", "Big Tech", "AI", "Science", "Space", "SpaceX", "OpenAI"}),
    ("Culture/Ent", {"Culture", "Entertainment", "Movies", "Music", "Celebrities",
                     "Awards", "TV", "Gaming", "Tweet Markets", "GTA VI", "Weather",
                     "Pop Culture"}),
    ("Geopolitics", {"Geopolitics", "Middle East", "Iran", "Israel", "Ukraine",
                     "Russia", "China", "World", "U.S. x Iran", "Israel x Iran",
                     "NATO", "Taiwan"}),
    ("Politics", {"Politics", "Trump", "US Politics", "White House", "Congress",
                  "Courts", "Immigration"}),
]


def category(tags) -> str:
    ts = set(tags or [])
    for name, keys in BUCKETS:
        if ts & keys:
            return name
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(HERE / "sample_v1.jsonl"))
    ap.add_argument("--no-fetch-descriptions", action="store_true")
    a = ap.parse_args()
    data = Path(a.data_dir)

    bps: dict[str, list[dict]] = {}
    for line in open(data / "raw" / "breakpoints.jsonl"):
        b = json.loads(line)
        bps.setdefault(b.pop("market_id"), []).append(b)
    have_prices = {os.path.basename(f)[:-5]
                   for f in glob.glob(str(data / "raw" / "prices" / "*.json"))}

    pools: dict[str, list[dict]] = {}
    for line in open(data / "raw" / "markets.jsonl"):
        m = json.loads(line)
        mid = m["id"]
        if mid not in have_prices or mid not in bps:
            continue
        cat = category(m.get("tags"))
        if cat == "Other":
            continue
        vol = m.get("volumeNum") or 0
        if vol < FLOORS.get(cat, DEFAULT_FLOOR):
            continue
        pools.setdefault(cat, []).append({
            "market_id": mid, "question": (m.get("question") or "").strip(),
            "slug": m.get("slug"), "category": cat, "volume": vol,
            "outcomes": m.get("outcomes"), "start_date": m.get("startDate"),
            "end_date": m.get("endDate"), "closed_time": m.get("closedTime"),
            "event_title": m.get("event_title"), "tags": m.get("tags"),
            "breakpoints": sorted(bps[mid], key=lambda b: b["t"]),
        })

    rng = random.Random(SEED)
    sample = []
    for cat, _ in BUCKETS:
        pool = sorted(pools.get(cat, []), key=lambda r: r["market_id"])
        take = pool if len(pool) <= PER_DOMAIN else rng.sample(pool, PER_DOMAIN)
        sample.extend(sorted(take, key=lambda r: r["market_id"]))
        print(f"{cat:12s} pool={len(pool):3d} sampled={min(len(pool), PER_DOMAIN)}")

    if not a.no_fetch_descriptions:
        for i, r in enumerate(sample):
            resp = requests.get(GAMMA.format(id=r["market_id"]), headers=UA,
                                timeout=30)
            r["description"] = (resp.json().get("description") or "").strip() \
                if resp.status_code == 200 else ""
            if (i + 1) % 50 == 0:
                print(f"  descriptions {i + 1}/{len(sample)}", flush=True)
            time.sleep(0.15)

    with open(a.out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    n_bp = sum(len(r["breakpoints"]) for r in sample)
    print(f"\nfroze {len(sample)} markets, {n_bp} breakpoints -> {a.out}")
    print("bps by category:", dict(Counter(
        r["category"] for r in sample for _ in r["breakpoints"])))


if __name__ == "__main__":
    main()
