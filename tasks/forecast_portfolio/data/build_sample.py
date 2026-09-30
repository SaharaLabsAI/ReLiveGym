"""Stratified, hindsight-free market sample for the forecast_portfolio task.

Universe: tasks/breakout_news_pm/market/raw/markets.jsonl (all Polymarket
markets whose scheduled life overlaps 2026-03-01..07-01, fetched 2026-07-21).

Eligibility (all observable at market open or at sim_start — no future info):
  - non-sports, non-mechanical tags (recurring price/weather binaries out)
  - volumeNum >= $100k (fetch-time lifetime volume; a quality proxy
    despite its mild hindsight tinge)
  - live during the window: NOT (closed with closedTime < 2026-03-01)

Stratification (open-time features ONLY — never realized resolution):
  - cells = tag family x scheduled-endDate month bucket (03/04/05/06/07+)
  - proportional allocation (largest remainder) to N_SAMPLE
  - per-event cap: <= 3 markets per event_id (Iran-cluster guard)
  - seeded shuffle -> deterministic sample

Resolution facts (start/resolution date + resolved answer) are then fetched
per market from gamma /markets/<id> (no batch-by-id endpoint exists: `id`
query filters return empty, comma lists 422) and
recorded as OUTCOME data, not used for selection.

Usage:
    python build_sample.py            # writes sample_v1.jsonl next to this file
    python build_sample.py --dry-run  # sampling only, no network
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
MARKETS = HERE.parents[1] / "breakout_news_pm" / "market" / "raw" / "markets.jsonl"
OUT = HERE / "sample_v1.jsonl"

SEED = 20260811
N_SAMPLE = 300
EVENT_CAP = 3
MIN_VOLUME = 100_000
W0 = dt.datetime(2026, 3, 1)
W1 = dt.datetime(2026, 7, 1)

SPORTS_TAGS = {
    "Sports", "NBA", "NFL", "MLB", "NHL", "Soccer", "EPL", "Champions League",
    "La Liga", "Serie A", "Bundesliga", "Tennis", "UFC", "MMA", "Boxing",
    "Golf", "F1", "NASCAR", "Esports", "College Football", "College Basketball",
    "Cricket", "Baseball", "Basketball", "Hockey", "Olympics",
}
MECH_TAGS = {
    "Recurring", "Crypto Prices", "Up or Down", "5M", "15M", "1H",
    "Weather", "Daily Temperature", "Highest temperature", "Hit Price",
    "Multi Strikes", "Tweet Markets",
}

# first match wins (a market tagged Iran + Politics is geopolitics)
FAMILIES = [
    ("geopolitics", {"Geopolitics", "World", "Middle East", "Iran", "Israel",
                     "U.S. x Iran", "Iran Ceasefire", "Israel x Iran",
                     "Ukraine", "Russia", "China", "Foreign Policy", "War",
                     "Military", "Gaza", "India", "Pakistan"}),
    ("politics", {"Politics", "Elections", "Global Elections",
                  "World Elections", "Trump", "US Politics", "Congress",
                  "Supreme Court", "NYC Mayor", "Trump Presidency"}),
    ("econ", {"Economy", "Finance", "Fed", "Fed Rates", "Inflation",
              "Stocks", "Trade War", "Tariffs", "Business"}),
    ("tech", {"Tech", "Big Tech", "AI", "Science", "SpaceX", "OpenAI"}),
    ("crypto", {"Crypto", "Bitcoin", "Ethereum", "Solana", "Memecoins"}),
    ("culture", {"Culture", "Entertainment", "Movies", "Music", "Awards",
                 "Celebrities", "TV", "Pop Culture", "GTA VI"}),
]


def parse_dt(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(
            s.replace("Z", "+00:00").replace(" ", "T")).replace(tzinfo=None)
    except ValueError:
        return None


def family_of(tags):
    ts = set(tags)
    for name, fam_tags in FAMILIES:
        if ts & fam_tags:
            return name
    return "other"


def end_bucket(ed):
    if ed is None or ed >= W1:
        return "07+"
    return ed.strftime("%m")


def load_universe():
    rows = []
    with open(MARKETS) as f:
        for line in f:
            m = json.loads(line)
            tags = set(m.get("tags") or [])
            if tags & SPORTS_TAGS or tags & MECH_TAGS:
                continue
            if any(t.startswith("Rewards Automation") for t in tags):
                continue
            if (m.get("volumeNum") or 0) < MIN_VOLUME:
                continue
            ct = parse_dt(m.get("closedTime"))
            if m.get("closed") and ct and ct < W0:
                continue  # already closed at sim_start: observable, excludable
            m["_end"] = parse_dt(m.get("endDate"))
            m["_family"] = family_of(tags)
            m["_bucket"] = end_bucket(m["_end"])
            rows.append(m)
    return rows


def allocate(universe):
    """Proportional allocation over family x end-bucket cells, largest
    remainder, then a seeded within-cell draw under the per-event cap."""
    cells = defaultdict(list)
    for m in universe:
        cells[(m["_family"], m["_bucket"])].append(m)
    total = len(universe)
    exact = {c: len(v) * N_SAMPLE / total for c, v in cells.items()}
    quota = {c: int(e) for c, e in exact.items()}
    for c, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)[: N_SAMPLE - sum(quota.values())]:
        quota[c] += 1

    rng = random.Random(SEED)
    event_n = Counter()
    picked = []
    for c, members in sorted(cells.items()):
        members = sorted(members, key=lambda m: m["id"])
        rng.shuffle(members)
        want, got = quota[c], 0
        for m in members:
            if got >= want:
                break
            ev = m.get("event_id") or f"solo:{m['id']}"
            if event_n[ev] >= EVENT_CAP:
                continue
            event_n[ev] += 1
            picked.append(m)
            got += 1
    return picked


def fetch_market(mid, tries=4):
    for i in range(tries):
        try:
            r = requests.get(
                f"https://gamma-api.polymarket.com/markets/{mid}",
                headers={"User-Agent": "program-engineering-dataset-build/0.1"},
                timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {"_fetch_error": "404"}
        except requests.RequestException:
            pass
    return {"_fetch_error": "failed"}


def resolved_answer(g):
    """Winning outcome from gamma outcomePrices; None unless one outcome
    priced 1.0 (anything else is flagged, not guessed)."""
    try:
        outcomes = json.loads(g.get("outcomes") or "[]")
        prices = [float(p) for p in json.loads(g.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return None, "unparseable"
    if not outcomes or len(outcomes) != len(prices):
        return None, "missing"
    top = max(prices)
    if top < 0.999:
        return None, "no_unit_price"
    return outcomes[prices.index(top)], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    universe = load_universe()
    print(f"eligible universe: {len(universe):,}")
    sample = allocate(universe)
    print(f"sampled: {len(sample)} (seed {SEED}, event cap {EVENT_CAP})")
    if args.dry_run:
        return

    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = list(ex.map(fetch_market, [m["id"] for m in sample]))

    n_err = 0
    recs = []
    for m, g in zip(sample, fetched):
        if g.get("_fetch_error"):
            n_err += 1
            recs.append({"market_id": m["id"], "question": m.get("question"),
                         "fetch_error": g["_fetch_error"]})
            continue
        ct = parse_dt(g.get("closedTime"))
        ans, status = (resolved_answer(g) if g.get("closed") else (None, "open"))
        recs.append({
            "market_id": m["id"],
            "question": g.get("question") or m.get("question"),
            "description": g.get("description"),
            "outcomes": json.loads(g.get("outcomes") or "[]"),
            "event_id": m.get("event_id"),
            "event_title": m.get("event_title"),
            "tags": m.get("tags"),
            "family": m["_family"],
            "sched_end_bucket": m["_bucket"],
            "volume_usd": round(m.get("volumeNum") or 0),
            "start_date": g.get("startDate") or m.get("startDate"),
            "scheduled_end": g.get("endDate") or m.get("endDate"),
            "resolution_date": g.get("closedTime"),
            "resolved_in_window": bool(ct and W0 <= ct < W1),
            "resolution_answer": ans,
            "answer_status": status,
            "uma_status": g.get("umaResolutionStatus"),
        })

    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in recs if not r.get("fetch_error")]
    riw = [r for r in ok if r["resolved_in_window"]]
    print(f"\nwrote {len(recs)} rows -> {OUT}  (fetch errors: {n_err})")
    print(f"resolved in window: {len(riw)}/{len(ok)} "
          f"({len(riw)/max(len(ok),1):.1%}) — outcome, not criterion")
    print("answer_status:", dict(Counter(r["answer_status"] for r in ok)))
    print("family x resolved_in_window:")
    fam = Counter((r["family"], r["resolved_in_window"]) for r in ok)
    for f_ in sorted({r["family"] for r in ok}):
        print(f"  {f_:12s} resolved {fam[(f_, True)]:3d} / open-past-window "
              f"{fam[(f_, False)]:3d}")
    months = Counter(r["resolution_date"][:7] for r in riw
                     if r["resolution_date"])
    print("resolution months (diagnostic):", dict(sorted(months.items())))


if __name__ == "__main__":
    sys.exit(main())
