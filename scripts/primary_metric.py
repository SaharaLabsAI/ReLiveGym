#!/usr/bin/env python
"""Print the paper's primary metric of finished runs.

    python scripts/primary_metric.py runs/task_daily_reddit_digest/gpt-5.6-luna/*-s?
    python scripts/primary_metric.py runs/episodes_grid/broker_ops/*/*        # web episodes

A run directory holds results.json (a web episode: server/results.json). For
most tasks the value is `performance.primary` of results.json; three metrics
are derived here exactly as in the paper's tables:

  forecast_portfolio     Brier skill score over the base-rate forecast: the task's
                         `ta_bss` is 1 - Brier; BSS = 1 - (1 - ta_bss) / BS_ref with
                         BS_ref the Brier score of the constant base-rate forecast
                         over the run's resolved questions.
  reddit_ai_popularity   time-weighted F1: precision = popular / settled recommendations,
                         recall = sum of first-recommendation decay weights of popular
                         posts / sum_days min(cap, popular posts that day).
  crypto_price_consistency  price accuracy: share of symbol-hours settled `ok`
                         (report within the free band) among the hours that start
                         before the run's first rejected LLM call.
  daily_reddit_digest    the daily score averaged over the days whose delivery window
                         opened before the first rejected LLM call.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def death_ts(run: Path) -> float | None:
    """Sim time of the first LLM call refused for budget, if any."""
    ledger = run / "ledger.jsonl"
    if not ledger.exists():
        return None
    with open(ledger) as fh:
        for line in fh:
            if '"llm_rejected"' in line:
                e = json.loads(line)
                if e.get("type") == "llm_rejected":
                    return ts(e["sim_time"])
    return None


def fp_bss(res: dict) -> float:
    qs = res["task"]["questions"]
    n = len(qs)
    outcomes = sorted({q["outcome"] for q in qs})
    pi = {o: sum(q["outcome"] == o for q in qs) / n for o in outcomes}
    bs_ref = sum((pi[o] - (1.0 if o == q["outcome"] else 0.0)) ** 2
                 for q in qs for o in outcomes) / n
    bs = sum(1.0 - q["ta_bss"] for q in qs) / n
    return 1.0 - bs / bs_ref


def reddit_tw_f1(res: dict) -> float:
    perf = res["performance"]
    settled = [r for r in res["task"]["recommendations"]
               if r.get("status") in ("tail", "nontail")]
    denom = round(perf["twr_ceiling_at_cap"] * perf["tail_posts"])
    if not settled or not denom:
        return 0.0
    first: dict = {}
    for r in sorted(settled, key=lambda r: r["recommended_at"]):
        first.setdefault(r["root_id"], r)
    credit = sum(r["weight"] for r in first.values() if r["status"] == "tail")
    p = sum(r["status"] == "tail" for r in settled) / len(settled)
    r_tw = credit / denom
    return 0.0 if p + r_tw == 0 else 2 * p * r_tw / (p + r_tw)


def cpc_price_acc(run: Path) -> float | None:
    death, hours = None, {}
    with open(run / "ledger.jsonl") as fh:
        for line in fh:
            if death is None and '"llm_rejected"' in line:
                e = json.loads(line)
                if e.get("type") == "llm_rejected":
                    death = ts(e["sim_time"])
            elif '"type": "outcome"' in line:
                e = json.loads(line)
                if e.get("type") == "outcome" and "@" in e.get("ref", ""):
                    hours[e["ref"]] = (ts(e["ref"].split("@", 1)[1]), e["status"] == "ok")
    ev = [ok for h, ok in hours.values() if death is None or h < death]
    return sum(ev) / len(ev) if ev else None


def digest_acc(run: Path, res: dict) -> float | None:
    death = death_ts(run)
    hour = res.get("config", {}).get("task", {}).get("digest_hour_utc", 12)
    scores = [v["score"] for d, v in res["task"]["daily_outcomes"].items()
              if death is None or ts(f"{d}T{hour:02d}:00:00Z") < death]
    return sum(scores) / len(scores) if scores else None


DERIVED = {
    "forecast_portfolio": ("bss", lambda run, res: fp_bss(res)),
    "reddit_ai_popularity": ("tw_f1", lambda run, res: reddit_tw_f1(res)),
    "crypto_price_consistency": ("price_acc", lambda run, res: cpc_price_acc(run)),
    "daily_reddit_digest": ("digest_acc", lambda run, res: digest_acc(run, res)),
}


def one(path: Path) -> tuple[str, str, str, float | None, float | None, list[str]]:
    run = path / "server" if (path / "server" / "results.json").exists() else path
    res = json.loads((run / "results.json").read_text())
    task = res.get("config", {}).get("task", {}).get("name") or res.get("task_name", "?")
    prim = res["performance"]["primary"]
    name, value = prim["name"], prim["value"]
    if task in DERIVED:
        name, fn = DERIVED[task]
        value = fn(run, res)
    spent = res.get("resources", {}).get("spent_usd")
    flags = [k for k, v in (res.get("flags") or {}).items() if v] if isinstance(res.get("flags"), dict) \
        else list(res.get("flags") or [])
    return res.get("run_id", path.name), task, name, value, spent, flags


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__); return 2
    rows = [one(Path(a)) for a in argv]
    w = max(len(r[0]) for r in rows)
    print(f"{'run':{w}}  {'task':24} {'metric':13} {'value':>8}  {'spent $':>8}  flags")
    for rid, task, name, value, spent, flags in rows:
        v = "n/a" if value is None else f"{value:.4f}"
        s = "n/a" if spent is None else f"{spent:.2f}"
        print(f"{rid:{w}}  {task:24} {name:13} {v:>8}  {s:>8}  {','.join(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
