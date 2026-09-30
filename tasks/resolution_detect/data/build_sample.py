"""Stratified, hindsight-free market roster for the resolution_detect task.

Universe: tasks/breakout_news_pm/market/raw/markets.jsonl (all Polymarket
markets whose scheduled life overlaps 2026-03-01..07-01, fetched 2026-07-21).

Far-deadline admission criterion:
  - startDate < sim_end (2026-07-01): the market opens inside the window
  - endDate >= sim_end + 30 d (2026-07-31): the scheduled deadline is far
    outside the window, so any in-window resolution is event-triggered by
    construction
  - non-sports, non-mechanical tags; volumeNum >= $100k (same quality proxy
    sanctioned for forecast_portfolio sample_v1)
  - alive at sim_start: NOT (closed with closedTime < 2026-03-01)

Sizing: the far-deadline universe at the $100k floor is 1,359 markets of
which 25.2% resolved in-window (dump of 2026-07-21).
N_SAMPLE=700 targets the gate's ~150-200 scored questions in expectation
(hypergeometric mean ~177, sd ~8); the ~75% quiet majority is roster
content, not waste — unresolved standing questions are what makes claim
discipline measurable.

Stratification (open-time features ONLY — never realized resolution):
  - cells = tag family x open-month bucket (pre03/03/04/05/06; the
    scheduled-end bucket used by forecast_portfolio is degenerate here —
    every market is "07+" by construction)
  - proportional allocation (largest remainder) to N_SAMPLE
  - per-event cap: <= 3 markets per event_id (series-cluster guard; the
    resolver pool has 25-market single-event crypto-launch series)
  - seeded shuffle -> deterministic sample

Resolution facts (closedTime + resolved answer) are then fetched per market
from gamma /markets/<id> and recorded as OUTCOME data, not used for
selection. resolved_in_window is computed from the gamma closedTime.

Easy-at-activation exclusion (post-selection, OUTCOME-BASED):
scored rows whose frozen dip-tolerant t_det falls within 1 h of
activation are dropped after enrichment — they are claimable at full
credit by pure retrieval at t=0 (pre-window-settled questions whose UMA
resolution lags into the window, and startDate-skew rows determined
before activation) and measure nothing about detection. This step reads
realized prices, so the roster is hindsight-free ONLY up to this
documented exclusion; the dropped ids are printed and pinned in the
README audit block.

FDV exclusion (post-selection, METADATA-BASED): crypto-launch "FDV above $X" questions
are dropped by question-text match. Their determinations are feed-settled
(token-price threshold crossings that CC-NEWS never reports): in smoke
runs not one FDV question was ever claimed, so they function as
guaranteed misses
that compress recall identically for every arm while measuring nothing
about detection. Text match only — reads no outcome data.

Usage:
    python build_sample.py             # full build -> sample_detect_v1.jsonl
    python build_sample.py --dry-run   # sampling only, no network
    python build_sample.py --refilter  # re-apply the easy exclusion to the
                                       # existing artifact, no network
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
MARKETS = HERE.parents[1] / "breakout_news_pm" / "market" / "raw" / "markets.jsonl"
OUT = HERE / "sample_detect_v1.jsonl"

SEED = 20260812
N_SAMPLE = 700
EVENT_CAP = 3
MIN_VOLUME = 100_000
W0 = dt.datetime(2026, 3, 1)
W1 = dt.datetime(2026, 7, 1)
DEADLINE_MARGIN = dt.timedelta(days=30)

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


def open_bucket(sd):
    if sd < W0:
        return "pre03"
    return sd.strftime("%m")


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
            sd = parse_dt(m.get("startDate"))
            ed = parse_dt(m.get("endDate"))
            if sd is None or ed is None:
                continue
            if not (sd < W1 and ed >= W1 + DEADLINE_MARGIN):
                continue
            ct = parse_dt(m.get("closedTime"))
            if m.get("closed") and ct and ct < W0:
                continue  # already closed at sim_start: observable, excludable
            m["_start"] = sd
            m["_family"] = family_of(tags)
            m["_bucket"] = open_bucket(sd)
            rows.append(m)
    return rows


def allocate(universe):
    """Proportional allocation over family x open-bucket cells, largest
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

    # top-up: the event cap can exhaust series-heavy cells before their
    # quota fills; recirculate the shortfall via a seeded global draw over
    # the leftovers, same cap
    if len(picked) < N_SAMPLE:
        chosen = {m["id"] for m in picked}
        rest = sorted((m for m in universe if m["id"] not in chosen),
                      key=lambda m: m["id"])
        rng.shuffle(rest)
        for m in rest:
            if len(picked) >= N_SAMPLE:
                break
            ev = m.get("event_id") or f"solo:{m['id']}"
            if event_n[ev] >= EVENT_CAP:
                continue
            event_n[ev] += 1
            picked.append(m)
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


# FDV exclusion (see module docstring): question-text match, no outcomes
FDV_RE = re.compile(r"\bFDV\b|fully.diluted", re.IGNORECASE)


def fdv_filter(recs):
    """Drop crypto-launch FDV questions (feed-settled family, invisible
    in CC-NEWS). Returns (kept, dropped_ids)."""
    kept, dropped = [], []
    for r in recs:
        if FDV_RE.search(r.get("question") or ""):
            dropped.append(r["market_id"])
        else:
            kept.append(r)
    return kept, dropped


def easy_filter(recs):
    """Drop scored rows determined at (or before) activation + 1 h under
    the frozen dip-tolerant predicate (see module docstring). Returns
    (kept, dropped_ids). Rows without a price file are kept — not
    easy-classifiable."""
    from audit_sample import EASY_S, PRICES, t_det_tolerant, ts as a_ts

    kept, dropped = [], []
    for r in recs:
        if (not r.get("fetch_error") and r.get("resolved_in_window")
                and r.get("answer_status") == "ok"):
            pf = PRICES / f"{r['market_id']}.json"
            if pf.exists():
                pts = json.load(pf.open())["points"]
                t_res = min(a_ts(r["resolution_date"]),
                            a_ts(r["scheduled_end"]))
                act = max(a_ts(r["start_date"]),
                          W0.replace(tzinfo=dt.timezone.utc).timestamp())
                td = t_det_tolerant(
                    pts, r["outcomes"].index(r["resolution_answer"]), t_res)
                if td is not None and (td - act) <= EASY_S:
                    dropped.append(r["market_id"])
                    continue
        kept.append(r)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refilter", action="store_true",
                    help="re-apply the easy exclusion to the existing "
                         "artifact (no network)")
    args = ap.parse_args()

    if args.refilter:
        recs = [json.loads(l) for l in open(OUT)]
        kept, dropped = easy_filter(recs)
        print(f"easy-at-activation exclusion: dropped {len(dropped)} "
              f"{sorted(dropped)}")
        kept, dropped_fdv = fdv_filter(kept)
        print(f"FDV exclusion: dropped {len(dropped_fdv)} "
              f"{sorted(dropped_fdv)}")
        with open(OUT, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(kept)} rows -> {OUT}")
        return

    universe = load_universe()
    print(f"far-deadline eligible universe: {len(universe):,}")
    sample = allocate(universe)
    print(f"sampled: {len(sample)} (seed {SEED}, event cap {EVENT_CAP})")
    if args.dry_run:
        cells = Counter((m["_family"], m["_bucket"]) for m in sample)
        for c in sorted(cells):
            print(f"  {c[0]:12s} {c[1]}: {cells[c]}")
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
            "open_bucket": m["_bucket"],
            "volume_usd": round(m.get("volumeNum") or 0),
            "start_date": g.get("startDate") or m.get("startDate"),
            "scheduled_end": g.get("endDate") or m.get("endDate"),
            "resolution_date": g.get("closedTime"),
            "resolved_in_window": bool(ct and W0 <= ct < W1),
            "resolution_answer": ans,
            "answer_status": status,
            "uma_status": g.get("umaResolutionStatus"),
        })

    recs, dropped = easy_filter(recs)
    print(f"easy-at-activation exclusion: dropped {len(dropped)} "
          f"{sorted(dropped)}")
    recs, dropped_fdv = fdv_filter(recs)
    print(f"FDV exclusion: dropped {len(dropped_fdv)} {sorted(dropped_fdv)}")

    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in recs if not r.get("fetch_error")]
    riw = [r for r in ok if r["resolved_in_window"]]
    print(f"\nwrote {len(recs)} rows -> {OUT}  (fetch errors: {n_err})")
    print(f"resolved in window: {len(riw)}/{len(ok)} "
          f"({len(riw)/max(len(ok),1):.1%}) — outcome, not criterion")
    print("answer_status:", dict(Counter(r["answer_status"] for r in ok)))
    ans = Counter(r["resolution_answer"] for r in riw
                  if r["answer_status"] == "ok")
    print("in-window resolver answers:", dict(ans))
    print("family x resolved_in_window:")
    fam = Counter((r["family"], r["resolved_in_window"]) for r in ok)
    for f_ in sorted({r["family"] for r in ok}):
        print(f"  {f_:12s} resolved {fam[(f_, True)]:3d} / quiet "
              f"{fam[(f_, False)]:3d}")
    months = Counter(r["resolution_date"][:7] for r in riw
                     if r["resolution_date"])
    print("resolution months (diagnostic):", dict(sorted(months.items())))
    nonbinary = [r for r in ok if r["outcomes"] not in ([], ["Yes", "No"])]
    print(f"non-Yes/No outcome sets: {len(nonbinary)}")


if __name__ == "__main__":
    sys.exit(main())
