"""Roster audit for sample_detect_v1.

Reads sample_detect_v1.jsonl + the bnpm hourly price store and pins the
roster's audit block:
  - scored set (resolved in window, answer_status ok) and Yes/No split
    (gate: winning-side share of the majority label within [0.35, 0.65])
  - dip-tolerant t_det per scored question (0.99 entry judged against the
    0.95 exit band), gap = t_res - t_det: median / >=12h / >=24h,
    no-crossing count (unwinnable)
  - trap census: losing side sustained >= 6h inside the 0.95 band
  - easy-at-activation count: t_det within 1h of activation (P10 flag)
  - price coverage for the scored set

Outcome labels are handled by winning index (first outcome's token is the
priced one), so non-Yes/No binary race markets score generically.

Usage: python audit_sample.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample_detect_v1.jsonl"
PRICES = HERE.parents[1] / "breakout_news_pm" / "market" / "raw" / "prices"

W0 = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
W1 = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
THETA_DET = 0.99
EXIT_BAND = 0.95
TRAP_SUSTAIN_S = 6 * 3600
EASY_S = 3600


def ts(s):
    s = s.replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    if "T" not in s:
        s = s.replace(" ", "T", 1)
    return datetime.fromisoformat(s).timestamp()


def t_det_tolerant(pts, win_idx, t_res):
    """Dip-tolerant determination time over a forward-filled price series
    for the FIRST outcome's token: last exit from the winning side's
    EXIT_BAND, then the first THETA_DET entry after it. None = no crossing
    (unwinnable). win_idx 0 -> winning band near 1, win_idx 1 -> near 0."""
    if win_idx == 0:
        in_det = lambda p: p >= THETA_DET
        in_exit = lambda p: p >= EXIT_BAND
    else:
        in_det = lambda p: p <= 1 - THETA_DET
        in_exit = lambda p: p <= 1 - EXIT_BAND
    pre = [(t, p) for t, p in pts if t < t_res]
    last_exit = None
    for t, p in pre:
        if not in_exit(p):
            last_exit = t
    for t, p in pre:
        if (last_exit is None or t > last_exit) and in_det(p):
            return t
    return None


def trapped(pts, win_idx, t_res):
    """Losing side sustained >= TRAP_SUSTAIN_S inside the EXIT_BAND."""
    if win_idx == 0:
        losing_in = lambda p: p <= 1 - EXIT_BAND
    else:
        losing_in = lambda p: p >= EXIT_BAND
    pre = [(t, p) for t, p in pts if t < t_res]
    run_start = None
    for i, (t, p) in enumerate(pre):
        t_next = pre[i + 1][0] if i + 1 < len(pre) else t_res
        if losing_in(p):
            if run_start is None:
                run_start = t
            if t_next - run_start >= TRAP_SUSTAIN_S:
                return True
        else:
            run_start = None
    return False


def main():
    rows = [json.loads(l) for l in SAMPLE.open()]
    ok = [r for r in rows if not r.get("fetch_error")]
    scored = [r for r in ok if r["resolved_in_window"]
              and r["answer_status"] == "ok"]
    quiet = [r for r in ok if r not in scored]

    print(f"roster: {len(rows)} rows ({len(ok)} fetched ok)")
    print(f"scored (resolved in window, answer ok): {len(scored)}  "
          f"quiet: {len(quiet)}")
    print(f"non-Yes/No binary rows: "
          f"{sum(1 for r in ok if r['outcomes'] not in ([], ['Yes', 'No']))}; "
          f"answer_status: {dict(Counter(r['answer_status'] for r in ok))}")

    split = Counter(
        "first" if r["outcomes"].index(r["resolution_answer"]) == 0 else "second"
        for r in scored)
    yn = Counter(r["resolution_answer"] for r in scored
                 if r["outcomes"] == ["Yes", "No"])
    maj = max(split.values()) / len(scored)
    print(f"\nwinning-side split (by outcome index): {dict(split)}  "
          f"majority share {maj:.3f}  GATE [0.35,0.65]: "
          f"{'PASS' if 0.35 <= maj <= 0.65 else 'FAIL'}")
    print(f"Yes/No-labelled subset: {dict(yn)}")

    gaps, no_cross, traps, easy, no_prices = [], [], 0, 0, []
    for r in scored:
        pf = PRICES / f"{r['market_id']}.json"
        if not pf.exists():
            no_prices.append(r["market_id"])
            continue
        pts = json.load(pf.open())["points"]
        t_res = min(ts(r["resolution_date"]), ts(r["scheduled_end"]))
        act = max(ts(r["start_date"]), W0)
        win_idx = r["outcomes"].index(r["resolution_answer"])
        td = t_det_tolerant(pts, win_idx, t_res)
        if td is None:
            no_cross.append(r["market_id"])
            continue
        gaps.append((t_res - td) / 3600)
        traps += trapped(pts, win_idx, t_res)
        easy += (td - act) <= EASY_S

    n_win = len(gaps)
    print(f"\nprice coverage: {len(scored) - len(no_prices)}/{len(scored)} "
          f"scored (missing: {no_prices or 'none'})")
    print(f"winnable (t_det exists): {n_win}/{len(scored)}  "
          f"no-crossing (unwinnable): {len(no_cross)} {no_cross}")
    print(f"gap hours: median {statistics.median(gaps):.1f}  "
          f">=12h {sum(g >= 12 for g in gaps)}/{n_win}  "
          f">=24h {sum(g >= 24 for g in gaps)}/{n_win}  "
          f"min {min(gaps):.1f}  max {max(gaps):.1f}")
    print(f"trap questions (losing side >=6h inside {EXIT_BAND} band): {traps}")
    print(f"easy-at-activation (t_det within 1h of activation): {easy}")

    fam = Counter((r["family"], r["resolved_in_window"]) for r in ok)
    print("\nfamily x (scored/quiet):")
    for f_ in sorted({r["family"] for r in ok}):
        print(f"  {f_:12s} scored {fam[(f_, True)]:3d} / quiet "
              f"{fam[(f_, False)]:3d}")
    ev = Counter(r["event_id"] or f"solo:{r['market_id']}" for r in ok)
    print(f"event cap: max markets/event = {max(ev.values())}, "
          f"events = {len(ev)}")


if __name__ == "__main__":
    main()
