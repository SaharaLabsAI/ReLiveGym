"""Compile the frozen breakout_news_pm env world.

Inputs (all already built by other pipelines):
  labeling/sample_v1.jsonl          the 350 frozen Sample-v1 markets
  labeling/out_v1/*.json            hindsight attribution labels (1,525)
  market/raw/breakpoints.jsonl      hourly-localized breakpoints
  market/raw/prices_1m/<mid>.json   minute change-series prices
  news/corpus/*.parquet + news/ccnews/pubtime/published_at.parquet

Outputs (data/built/, gitignored except build_stats.json):
  markets.jsonl        env market roster + metadata
  prices/<mid>.json    minute change-series (copied verbatim)
  breakpoints.jsonl    all labeled episodes, minute-localized canonical window
  attributions.jsonl   labels with cited prefixes resolved to full corpus ids
                       + pub_ts — UNFILTERED: the scorer applies the
                       confidence floor and the W-hour gold window from run
                       config at load, so thresholds stay live without a
                       rebuild. build_stats.json pins the counts at the
                       defaults (attr_threshold 0.6, W 24h).
  build_stats.json     tracked calibration pins

Alternate oracle sources: a second hindsight labeler's episodes
(labeling/run_labeling.py --model ... --out labeling/out_<tag>) compile into
a sibling attributions file with `--labels labeling/out_<tag>
--attributions-out built/attributions_<tag>.jsonl`; nothing else is rebuilt.
A run selects it via `task.attributions_path` (task.py); the default stays
attributions.jsonl (gpt-5.6-sol, v1 index). Episodes without a valid submit
(label null) are compiled as no_attribution with `no_submit: true`.

Usage: python3 build.py [--skip-prices]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
OUT = HERE / "built"

GRID_S = 60  # minute grid of the price series
MAX_DESCRIPTION_CHARS = 2_000  # same truncation the labeler saw

# defaults used ONLY for the pinned stats preview; the scorer re-applies
# its own config values at load
DEFAULT_ATTR_THRESHOLD = 0.6
DEFAULT_W_HOURS = 24


def localize_1m(pts: list[list[float]], t0: float, t1: float,
                dp: float) -> dict | None:
    """Minimal interval (i, j] with |p_j - p_i| >= 0.8|dp| on the minute
    change-series; t_move_start is one grid step before the first repricing
    inside the interval (price provably still at pre-level then, by
    forward-fill). Same semantics as detect_breakpoints.localize."""
    seg = [(t, p) for t, p in pts if t0 < t <= t1]
    if len(seg) < 2:
        return None
    need = 0.8 * abs(dp)
    best = None  # (span, i, j)
    for i in range(len(seg) - 1):
        for j in range(i + 1, len(seg)):
            if abs(seg[j][1] - seg[i][1]) >= need:
                span = seg[j][0] - seg[i][0]
                if best is None or span < best[0]:
                    best = (span, i, j)
                break  # first j for this i is the shortest for this i
    if best is None:
        return None
    _, i, j = best
    t_start = max(seg[i][0], seg[i + 1][0] - GRID_S)
    steps = [abs(seg[k + 1][1] - seg[k][1]) for k in range(i, j)]
    return {"t_move_start": t_start, "t_move_end": seg[j][0],
            "step_frac": min(1.0, max(steps) / abs(dp)) if dp else 1.0}


def build_markets(sample_rows: list[dict]) -> list[dict]:
    out = []
    for s in sample_rows:
        desc = (s.get("description") or "").strip()
        if len(desc) > MAX_DESCRIPTION_CHARS:
            desc = desc[:MAX_DESCRIPTION_CHARS] + " [...truncated]"
        out.append({
            "market_id": s["market_id"], "question": s["question"],
            "category": s["category"], "event_title": s.get("event_title"),
            "volume": float(s["volume"]),
            "start_date": s.get("start_date"), "end_date": s.get("end_date"),
            "closed_time": s.get("closed_time"),
            "description": desc, "grid_minutes": 1,
        })
    return out


def build_breakpoints(sample_ids: set[str], labeled: set[str],
                      prices_dir: Path, bp_path: Path) -> tuple[list[dict], int]:
    prices: dict[str, list] = {}
    rows, fallbacks = [], 0
    with open(bp_path) as f:
        for line in f:
            b = json.loads(line)
            key = f"{b['market_id']}_{b['date']}"
            if b["market_id"] not in sample_ids or key not in labeled:
                continue
            mid = b["market_id"]
            if mid not in prices:
                prices[mid] = json.load(
                    open(prices_dir / f"{mid}.json"))["points"]
            loc = localize_1m(prices[mid], b["t_prev_trade"],
                              b["t_last_trade"], b["dp"])
            if loc is None:  # too few minute changes; keep hourly window
                loc = {"t_move_start": b["t_move_start"],
                       "t_move_end": b["t_move_end"],
                       "step_frac": b["step_frac"]}
                localization = "hourly_fallback"
                fallbacks += 1
            else:
                localization = "minute"
            rows.append({"market_id": mid, "date": b["date"],
                         "dp": b["dp"], "z": b["z"],
                         "p_prev": b["p_prev"], "p": b["p"],
                         **loc, "localization": localization})
    rows.sort(key=lambda r: (r["t_move_start"], r["market_id"]))
    return rows, fallbacks


def resolve_prefixes(prefixes: set[str], news_dir: Path) -> dict[str, list]:
    """Cited prefix -> [(full_id, pub_ts), ...] (collision-safe). Prefixes
    are normally the 16-char ids shown in search results, but the labeler's
    submit check accepts any true prefix (`startswith`), and some labelers
    (gpt-5.6-luna) cite 14–15 chars — so join per prefix length, each an
    equality join on substr(id, 1, len)."""
    con = duckdb.connect()
    out: dict[str, list] = {}
    for n in sorted({len(x) for x in prefixes}):
        con.execute("CREATE OR REPLACE TEMP TABLE pfx(p VARCHAR)")
        con.executemany("INSERT INTO pfx VALUES (?)",
                        [(x,) for x in prefixes if len(x) == n])
        rows = con.execute(f"""
            SELECT substr(c.id, 1, {n}), c.id,
                   least(epoch(CAST(c.warc_date AS TIMESTAMP)),
                         coalesce(epoch(CAST(s.published_at AS TIMESTAMP)),
                                  epoch(CAST(c.warc_date AS TIMESTAMP))))
            FROM read_parquet('{news_dir}/corpus/corpus_2026-*.parquet') c
            JOIN pfx ON substr(c.id, 1, {n}) = pfx.p
            LEFT JOIN read_parquet(
                '{news_dir}/ccnews/pubtime/published_at.parquet') s USING (id)
            ORDER BY 1, 2""").fetchall()
        for pfx, full_id, pub_ts in rows:
            out.setdefault(pfx, []).append((full_id, int(pub_ts)))
    return out


def build_attributions(labels_dir: Path, news_dir: Path) -> tuple[list[dict], dict]:
    episodes, prefixes = [], set()
    no_submit = 0
    for p in sorted(labels_dir.glob("*.json")):
        d = json.load(open(p))
        lab = d["label"]
        mid, date = p.stem.rsplit("_", 1)
        if lab is None:  # hit MAX_TURNS without a valid submit
            no_submit += 1
            episodes.append({"market_id": mid, "date": date,
                             "no_attribution": True, "no_submit": True,
                             "groups": []})
            continue
        groups = []
        for g in lab.get("groups") or []:
            ids = [i[:16] for i in g["news_ids"]]
            prefixes.update(ids)
            groups.append({"story": g["story"], "confidence": g["confidence"],
                           "likely_reports_move": g["likely_reports_move"],
                           "cited": ids})
        episodes.append({"market_id": mid, "date": date,
                         "no_attribution": bool(lab.get("no_attribution")),
                         "groups": groups})
    resolved = resolve_prefixes(prefixes, news_dir)
    unresolved = sorted(p for p in prefixes if p not in resolved)
    for e in episodes:
        for g in e["groups"]:
            g["articles"] = [{"news_id": fid, "pub_ts": ts}
                             for pfx in g.pop("cited")
                             for fid, ts in resolved.get(pfx, [])]
    stats = {"episodes": len(episodes),
             "attributed_episodes": sum(1 for e in episodes
                                        if not e["no_attribution"]),
             "groups": sum(len(e["groups"]) for e in episodes),
             "cited_prefixes": len(prefixes),
             "unresolved_prefixes": len(unresolved)}
    if no_submit:
        stats["no_submit_episodes"] = no_submit
    if unresolved:
        stats["unresolved_list"] = unresolved
    return episodes, stats


def gold_preview(episodes: list[dict], bps: list[dict],
                 attr_threshold: float, w_hours: float) -> dict:
    """The counts the scorer's default filters would produce — pinned so a
    config/data drift is caught by tests, not discovered in a run."""
    t_start = {(b["market_id"], b["date"]): b["t_move_start"] for b in bps}
    w_s = w_hours * 3600
    kept_groups = gold_articles = winnable = 0
    dropped_early = dropped_late = 0
    for e in episodes:
        ts = t_start[(e["market_id"], e["date"])]
        ep_winnable = False
        for g in e["groups"]:
            if g["confidence"] < attr_threshold:
                continue
            n_gold = 0
            for a in g["articles"]:
                if a["pub_ts"] < ts - w_s:
                    dropped_early += 1
                elif a["pub_ts"] >= ts:
                    dropped_late += 1
                else:
                    n_gold += 1
            if n_gold:
                kept_groups += 1
                gold_articles += n_gold
                ep_winnable = True
        winnable += ep_winnable
    return {"attr_threshold": attr_threshold, "w_hours": w_hours,
            "gold_groups": kept_groups, "gold_articles": gold_articles,
            "winnable_breakpoints": winnable,
            "unwinnable_breakpoints": len(episodes) - winnable,
            "articles_dropped_early": dropped_early,
            "articles_dropped_late": dropped_late}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prices", action="store_true",
                    help="skip copying minute price files")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labels-only mode: compile this episode dir into "
                         "--attributions-out (markets/prices/breakpoints "
                         "untouched)")
    ap.add_argument("--attributions-out", type=Path, default=None,
                    help="output path for --labels mode")
    ap.add_argument("--partial", action="store_true",
                    help="--labels mode: allow an episode dir covering only "
                         "some built breakpoints (roster-scoped self-oracle "
                         "labels, labeling/run_labeling_roster.py); the file "
                         "is valid only for runs whose windows it covers")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.labels is not None:
        if args.attributions_out is None:
            ap.error("--labels requires --attributions-out")
        bps = [json.loads(l) for l in open(OUT / "breakpoints.jsonl")]
        have = {(b["market_id"], b["date"]) for b in bps}
        episodes, attr_stats = build_attributions(args.labels, TASK / "news")
        got = {(e["market_id"], e["date"]) for e in episodes}
        missing = sorted(have - got)
        if missing and not args.partial:
            # the scorer joins by (market_id, date) inside the run windows:
            # a full-sample file must be total
            raise SystemExit(f"{len(missing)} built breakpoints have no "
                             f"episode in {args.labels}: {missing[:5]}... "
                             f"(pass --partial for a roster-scoped file)")
        if missing:
            print(f"PARTIAL: {len(got & have)} of {len(have)} built "
                  f"breakpoints labeled; {len(missing)} absent")
        episodes = [e for e in episodes if (e["market_id"], e["date"]) in have]
        args.attributions_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.attributions_out, "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")
        print(f"{args.attributions_out}: {attr_stats}")
        preview = gold_preview(episodes, bps, DEFAULT_ATTR_THRESHOLD,
                               DEFAULT_W_HOURS)
        print(f"gold preview @defaults: {preview}")
        stats_path = args.attributions_out.with_suffix(".stats.json")
        stats_path.write_text(json.dumps(
            {"labels_dir": str(args.labels), "partial": bool(missing),
             "built_breakpoints_labeled": len(got & have), **attr_stats,
             "gold_preview_defaults": preview}, indent=2) + "\n")
        return

    sample_rows = [json.loads(l)
                   for l in open(TASK / "labeling" / "sample_v1.jsonl")]
    sample_ids = {s["market_id"] for s in sample_rows}
    labeled = {p.stem for p in (TASK / "labeling" / "out_v1").glob("*.json")}

    markets = build_markets(sample_rows)
    with open(OUT / "markets.jsonl", "w") as f:
        for m in sorted(markets, key=lambda m: m["market_id"]):
            f.write(json.dumps(m) + "\n")
    print(f"markets.jsonl: {len(markets)} markets")

    if not args.skip_prices:
        (OUT / "prices").mkdir(exist_ok=True)
        for mid in sorted(sample_ids):
            shutil.copyfile(TASK / "market" / "raw" / "prices_1m" / f"{mid}.json",
                            OUT / "prices" / f"{mid}.json")
        print(f"prices/: {len(sample_ids)} minute change-series copied")

    bps, fallbacks = build_breakpoints(
        sample_ids, labeled, TASK / "market" / "raw" / "prices_1m",
        TASK / "market" / "raw" / "breakpoints.jsonl")
    with open(OUT / "breakpoints.jsonl", "w") as f:
        for b in bps:
            f.write(json.dumps(b) + "\n")
    print(f"breakpoints.jsonl: {len(bps)} episodes "
          f"({fallbacks} hourly fallbacks)")

    episodes, attr_stats = build_attributions(TASK / "labeling" / "out_v1", TASK / "news")
    with open(OUT / "attributions.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    print(f"attributions.jsonl: {attr_stats}")

    preview = gold_preview(episodes, bps, DEFAULT_ATTR_THRESHOLD,
                           DEFAULT_W_HOURS)
    print(f"gold preview @defaults: {preview}")

    stats = {"markets": len(markets), "breakpoints": len(bps),
             "hourly_fallbacks": fallbacks, **attr_stats,
             "gold_preview_defaults": preview}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"build_stats.json written -> {OUT}")


if __name__ == "__main__":
    main()
