"""Build the resolution_detect world from the frozen sample_detect_v1.jsonl.

Output (data/built/, gitignored):
  questions.jsonl   one row per roster market: an `agent` block (the ONLY
                    fields that may ever cross a tool boundary) and a
                    `scorer` block (ground truth + provenance slices)

There is NO built/prices/ and no price surface anywhere:
t_det is computed HERE, once, from the shared bnpm hourly price store
under the frozen settlement constants, and pinned into the scorer
block. The env never loads prices; the scorer reads t_det as data.

Frozen settlement constants (treatment constants; changing them is a new
task version):
  theta_det = 0.99, exit band = 0.95 (dip-tolerant sustain rule),
  grace g = 0, decay = gap-normalized linear.
Reference implementation of the predicate: audit_sample.t_det_tolerant
(imported — single source of truth).

Scoring facts pinned here:
  activation = max(open_date, sim_start at run time)   (staggered arrivals)
  t_res      = min(closedTime, scheduled_end)          (= closedTime on this
                                                        far-deadline roster)
  scored     = resolved_in_window and answer_status ok
  winnable   = scored and t_det exists (dip-tolerant crossing before t_res)
  trap       = losing side sustained >= 6 h inside the exit band (report)
  news_lead_flag = census-pinned proxy-error questions (confirming news
                   preceded t_det) — report-only

Invariant (easy exclusion): every winnable row must satisfy
t_det > W0-activation + 1 h; the build FAILS on violation (stale sample).

Usage:
    python build.py
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from audit_sample import (  # noqa: E402
    EASY_S, PRICES, t_det_tolerant, trapped, ts,
)

SAMPLE = HERE / "sample_detect_v1.jsonl"
BUILT = HERE / "built"

W0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc).timestamp()

# census-pinned confirming-lead questions (analysis/README.md): news
# reported the settling event well before the price predicate fired
NEWS_LEAD_QIDS = {"1184762", "2100840", "891191", "2510940"}


def main() -> None:
    rows = [json.loads(line) for line in SAMPLE.open()]
    bad = [r for r in rows if r.get("fetch_error")]
    if bad:
        raise SystemExit(f"sample has fetch_error rows: {bad}")

    BUILT.mkdir(exist_ok=True)
    out = []
    gaps_h = []
    n_scored = n_winnable = n_trap = 0
    no_cross = []
    split = {0: 0, 1: 0}
    for r in rows:
        mid = r["market_id"]
        scored = bool(r["resolved_in_window"] and r["answer_status"] == "ok")
        t_res = t_det = gap_s = None
        winnable = trap = False
        if scored:
            n_scored += 1
            t_res = min(ts(r["resolution_date"]), ts(r["scheduled_end"]))
            act = max(ts(r["start_date"]), W0)
            if t_res <= act:
                raise SystemExit(f"{mid}: t_res <= activation")
            win_idx = r["outcomes"].index(r["resolution_answer"])
            split[win_idx] += 1
            pf = PRICES / f"{mid}.json"
            pts = (json.loads(pf.read_text())["points"] if pf.exists()
                   else [])
            t_det = t_det_tolerant(pts, win_idx, t_res)
            if t_det is not None and t_res - t_det <= 0:
                t_det = None  # crossing at t_res exactly: no earnable window
            if t_det is None:
                no_cross.append(mid)
            else:
                if t_det - act <= EASY_S:
                    raise SystemExit(
                        f"{mid}: easy-at-activation row in the sample — "
                        f"stale artifact, re-run build_sample.py --refilter")
                winnable = True
                n_winnable += 1
                gap_s = t_res - t_det
                gaps_h.append(gap_s / 3600)
                trap = trapped(pts, win_idx, t_res)
                n_trap += trap
        out.append({
            "question_id": mid,
            "agent": {
                "question": r["question"],
                "description": r["description"],  # verbatim criteria
                "outcomes": r["outcomes"],
                "open_date": r["start_date"],
                "scheduled_end": r["scheduled_end"],
            },
            "scorer": {
                "scored": scored,
                "t_res": t_res,
                "closed_time": r["resolution_date"],
                "resolution_answer": r["resolution_answer"],
                "answer_status": r["answer_status"],
                "resolved_in_window": r["resolved_in_window"],
                "t_det": t_det,
                "gap_s": gap_s,
                "winnable": winnable,
                "trap": trap,
                "news_lead_flag": mid in NEWS_LEAD_QIDS,
                "family": r["family"],
                "open_bucket": r["open_bucket"],
                "volume_usd": r["volume_usd"],
                "event_id": r["event_id"],
                "event_title": r["event_title"],
                "tags": r["tags"],
            },
        })

    with (BUILT / "questions.jsonl").open("w") as f:
        for q in out:
            f.write(json.dumps(q) + "\n")

    # -- audit block (pinned in data/README.md + tests) --------------------
    maj = max(split.values()) / n_scored
    print(f"questions: {len(out)}  scored: {n_scored}  "
          f"winnable: {n_winnable}  no-crossing: {len(no_cross)} {no_cross}")
    print(f"winning-side split: {split[0]}/{split[1]}  majority {maj:.3f}")
    print(f"gap hours: median {statistics.median(gaps_h):.1f}  "
          f">=12h {sum(g >= 12 for g in gaps_h)}  "
          f">=24h {sum(g >= 24 for g in gaps_h)}")
    print(f"trap: {n_trap}  news_lead_flag: "
          f"{sum(1 for q in out if q['scorer']['news_lead_flag'])}")


if __name__ == "__main__":
    main()
