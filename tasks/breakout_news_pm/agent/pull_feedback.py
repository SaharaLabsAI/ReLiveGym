"""Sig-self feedback compilation cadence for breakout_news_pm: hindsight
verdicts over the last closed window, at most once per
HINDSIGHT_EVERY_HOURS. Runs inside the tlrn-scheduled learn step; the
throttle is nearly vacuous under tlrn=daily but guards a faster level.

The candidate registry is global (records.register_candidates): observed
articles are not market-partitioned, so each hindsight sweep verdicts the
window's candidates against every monitored market — attribution is the
LLM call's job, and it only fires for markets where the detector found a
breakpoint (quiet markets cost detector-only, i.e. one price call).
Per-market `attributed` verdicts are emitted as they land; candidates
attributed nowhere yield one `no_breakout` verdict each, once per sweep.
Provisioned only in sig=self cells (feedback_fn.py sits next to it)."""

from __future__ import annotations

from datetime import datetime, timedelta

import feedback_fn

HINDSIGHT_EVERY_HOURS = 24
HINDSIGHT_WINDOW_DAYS = 2  # look back this far per hindsight call
HINDSIGHT_LAG_HOURS = 24   # judge only periods closed at least this long


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def compile(env, state: dict) -> list[dict]:
    now = env.now()
    last = state.get("last_hindsight")
    if last and _parse(last) + timedelta(hours=HINDSIGHT_EVERY_HOURS) > now:
        return []
    until = now - timedelta(hours=HINDSIGHT_LAG_HOURS)
    since = until - timedelta(days=HINDSIGHT_WINDOW_DAYS)
    if until + timedelta(hours=feedback_fn.GRACE_HOURS) > now:
        return []  # period not closed yet; retry next firing, don't stamp
    seen = state.get("registered", {})
    candidates = [
        {"news_id": nid, "title": v["title"], "published": v["published"]}
        for nid, v in seen.items()
        if _iso(since) <= v["published"] <= _iso(until)]
    out: list[dict] = []
    if candidates:
        attributed: set[str] = set()
        for m in env.call("get_markets"):
            try:
                result = feedback_fn.hindsight_verdicts(
                    env, m["market_id"], since, until, candidates,
                    question=m["question"])
            except ValueError:
                continue
            for v in result["verdicts"]:
                if v["verdict"] == "attributed":
                    attributed.add(v["news_id"])
                    out.append({"kind": "verdict",
                                "market_id": m["market_id"],
                                "t_settled": _iso(until), **v})
        for c in candidates:
            if c["news_id"] not in attributed:
                out.append({"kind": "verdict", "news_id": c["news_id"],
                            "verdict": "no_breakout",
                            "t_settled": _iso(until)})
            seen.pop(c["news_id"], None)  # each candidate is verdicted once
    state["last_hindsight"] = _iso(now)
    return out
