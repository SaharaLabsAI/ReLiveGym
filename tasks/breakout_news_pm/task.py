"""Breakout-news task on the self-owned CC-NEWS + Polymarket minute world.

A notification at sim time tau is the claim: "market m will START a breakout
in direction d within (tau, tau + W]; this article is my evidence."

Ground truth: the 1,525 hindsight-labeled breakpoints of the 350 Sample-v1
markets, minute-localized (data/build.py). Gold-citable articles per
breakpoint: cited articles of groups with confidence >= attr_threshold whose
pub_ts lies in [t_move_start - W, t_move_start) — i.e. articles on which an
immediate, honest W-hour claim would have been true. Filters are applied at
load from run config; data/built/attributions.jsonl is unfiltered.

Scoring (metrics, not dollars; all closes at t_move_start;
tau >= t_move_start earns nothing):
- covered_news   -- earliest eligible notification (tau < t_start <= tau+W,
  direction matches) citing a gold article. Time credit decays linearly
  from the cited article's publish (credit 1) to the move start (0):
  react fast to the right story.
- covered_timing -- eligible notification citing a non-gold article:
  flat timing_credit. Credit for beating the breakout on evidence the
  labeler didn't cite; never a false alarm.
- miss           -- no eligible notification: credit 0 (winnable is a
  reporting/oracle annotation; settlement never branches on it).
- one standing claim per market: a new alert is rejected (free 400) while
  a previous alert on that market is pending — until it covers a
  breakpoint or expires at tau + W as false_alarm (tagged wrong_direction
  when a breakout matched on market+timing but not direction; a
  wrong_direction alert whose paired breakpoint was missed is excused —
  excluded from precision — so attempting a direction never ranks worse
  than silence). Same-breakout spam and up/down hedging are structurally
  impossible.

Primary metric (v2): cov_f1 — harmonic mean of precision (covering
alerts / non-excused resolved alerts) and cov_recall (covered
breakpoints / ALL closed breakpoints, binary credit). The v1 TC-F1
(gold-cite decay credit over winnable breakpoints only) is retained as
a reporting-only secondary: its news decay could score a causal cite
below the flat timing credit, and its winnable split rests on a single
hindsight labeling pass. Visibility: news searchable from pub_ts (date_to clamped
to now); a price point at t is visible from t + price_delay_minutes.
Event-driven waiting is authored gatekeeper code (harness/authored.py);
there is no declarative condition grammar.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import as_utc, iso, parse_iso

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TASK_DIR / "data" / "built"
DEFAULT_INDEX_DIR = TASK_DIR / "news" / "tantivy_index_v3"  # v3 clock
#   (news/build_index.py): visibility = corroborated self-report, else
#   crawl − 2 h.

DIRECTIONS = ("up", "down")


def _ts(dt: datetime) -> float:
    return dt.timestamp()


def _iso_ts(t: float) -> str:
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))


# -- config ---------------------------------------------------------------------------


class MarketWindow(BaseModel):
    market_id: str
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return as_utc(v)

    @model_validator(mode="after")
    def _order(self) -> "MarketWindow":
        if self.end <= self.start:
            raise ValueError(f"market {self.market_id}: end must be after start")
        return self


class BreakoutPMCostConfig(BaseModel):
    price_call: float = 0.0  # market data is free in reality
    # commercial news-API anchor (~$2 per 1k requests, basis: approximate)
    news_search_call: float = 0.002
    article_call: float | None = None  # defaults to news_search_call

    @model_validator(mode="after")
    def _defaults(self) -> "BreakoutPMCostConfig":
        if self.article_call is None:
            self.article_call = self.news_search_call
        return self


class BreakoutNewsPMConfig(BaseModel):
    markets: list[MarketWindow]
    breakout_window_hours: float = 24.0  # W: claim horizon == gold window
    attr_threshold: float = 0.6  # gold = groups with confidence >= this
    timing_credit: float = 0.7  # covered_timing flat credit (metric const)
    price_delay_minutes: float = 10.0  # price point at t visible from t+delay
    search_top_k: int = 10
    data_dir: Path | None = None  # default: tasks/breakout_news_pm/data/built
    # oracle source: which hindsight labeler's attributions feed gold /
    # sig=oracle feedback. Default data_dir/attributions.jsonl (gpt-5.6-sol);
    # alternates built by data/build.py --labels (e.g. attributions_terra.jsonl)
    attributions_path: Path | None = None
    news_index_dir: Path | None = None  # default: .../news/tantivy_index_v3
    cost: BreakoutPMCostConfig = BreakoutPMCostConfig()
    # advertised + enforced limits (INSTRUCTION.md):
    # NO rate limits on the news and price APIs — {"window": "none"} is
    # always allowed and advertised as "none (unlimited)". The former
    # defaults (news 60/min, prices 600/min, fixed window) can be
    # restored per run from the task section.
    news_rate_limit: dict = {"window": "none"}
    price_rate_limit: dict = {"window": "none"}

    @model_validator(mode="after")
    def _windows(self) -> "BreakoutNewsPMConfig":
        if not self.markets:
            raise ValueError("task.markets must list at least one market window")
        by_mid: dict[str, list[MarketWindow]] = {}
        for w in self.markets:
            by_mid.setdefault(w.market_id, []).append(w)
        for mid, ws in by_mid.items():
            ws.sort(key=lambda w: w.start)
            for a, b in zip(ws, ws[1:]):
                if b.start < a.end:
                    raise ValueError(f"market {mid}: overlapping monitoring windows")
        return self

    def _resolve(self, path: Path | None, default: Path,
                 base_dir: Path | None) -> Path:
        if path is None:
            return default
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path

    def resolve_data_dir(self, base_dir: Path | None = None) -> Path:
        return self._resolve(self.data_dir, DEFAULT_DATA_DIR, base_dir)

    def resolve_index_dir(self, base_dir: Path | None = None) -> Path:
        return self._resolve(self.news_index_dir, DEFAULT_INDEX_DIR, base_dir)

    def resolve_attributions_path(self, base_dir: Path | None = None) -> Path:
        return self._resolve(self.attributions_path,
                             self.resolve_data_dir(base_dir) / "attributions.jsonl",
                             base_dir)


# -- stores ---------------------------------------------------------------------------


class PriceStore:
    """Minute change-series per monitored market (sparse, forward-fill:
    between two stored points the price is the earlier point's value)."""

    def __init__(self, data_dir: Path, market_ids: list[str], delay_minutes: float):
        self._delay_s = delay_minutes * 60
        self._series: dict[str, list[list[float]]] = {}
        for mid in market_ids:
            path = data_dir / "prices" / f"{mid}.json"
            if not path.exists():
                raise ValueError(f"no price data for market {mid!r} in {data_dir}")
            self._series[mid] = json.loads(path.read_text())["points"]

    def query(self, market_id: str, start: datetime, end: datetime,
              now: datetime, grid_minutes: float = 1) -> dict:
        if isinstance(grid_minutes, bool) \
                or not isinstance(grid_minutes, (int, float)) \
                or grid_minutes < 1:
            raise ValueError(
                "grid_minutes must be a number >= 1 (coarser replies are "
                "smaller; finer than the stored 1-minute grid is not "
                "offered)")
        pts = self._series[market_id]
        lo = _ts(start)
        hi = min(_ts(end), _ts(now) - self._delay_s)
        level = None
        for t, p in pts:
            if t > min(lo, hi):
                break
            level = p
        changes = [[t, p] for t, p in pts if lo < t <= hi]
        if grid_minutes > 1:
            # last visible change per N-minute bucket anchored at `start`,
            # kept at its actual timestamp; consecutive equal values under
            # forward-fill collapse (dedupe seeded with level_at_start)
            step = grid_minutes * 60
            last: dict[int, list[float]] = {}
            for tt, p in changes:  # sorted, so last in bucket wins
                last[int((tt - lo) // step)] = [tt, p]
            changes, prev = [], level
            for k in sorted(last):
                tt, p = last[k]
                if p != prev:
                    changes.append([tt, p])
                    prev = p
        return {
            "market_id": market_id,
            "grid_minutes": grid_minutes,
            "sparse": "changes",  # forward-fill between points
            "level_at_start": level if hi >= lo else None,
            "changes": {"time": [_iso_ts(t) for t, _ in changes],
                        "p": [p for _, p in changes]},
            "visible_until": _iso_ts(hi) if hi >= lo else None,
        }


class NewsStore:
    """Visibility-clamped wrapper over the shared BM25 module
    (news/search.py): date filters run on pub_ts, date_to is clamped to
    `now`, and get_article never returns an unpublished article or the
    internal provenance fields."""

    def __init__(self, index_dir: Path):
        from tasks.breakout_news_pm.news.search import NewsSearch

        self._engine = NewsSearch(str(index_dir))

    def search(self, query: str, date_from: str | None, date_to: str | None,
               now: datetime, top_k: int, offset: int) -> list[dict]:
        now_iso = datetime.fromtimestamp(_ts(now), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S")
        clamped = now_iso if date_to is None else min(str(date_to), now_iso)
        return self._engine.search(query, date_from, clamped, top_k, offset)

    def get_article(self, news_id: str, now: datetime) -> dict | None:
        d = self._engine.get_article(news_id)
        if d is None or d["pub_ts"] > _ts(now):
            return None
        return {"news_id": d["id"], "title": d["title"],
                "domain": d["domain"], "published": d["pub_date"],
                "description": d.get("description") or "",
                "text": d.get("text") or ""}

    def earliest_match(self, query: str, lo_ts: float,
                       hi_ts: float) -> int | None:
        """Earliest pub_ts in (lo_ts, hi_ts] matching `query` (search()
        syntax) — the canonical engine, so a standing wake condition
        matches exactly what polling search_news would."""
        return self._engine.earliest_match(query, lo_ts, hi_ts)

    def published_ts(self, news_id: str) -> float | None:
        d = self._engine.get_article(news_id)
        return None if d is None else float(d["pub_ts"])


# -- ground truth ---------------------------------------------------------------------


@dataclass
class Breakpoint:
    market_id: str
    date: str
    t_start: float  # closes here: no reward at or after the move start
    t_end: float
    direction: str  # "up" | "down"
    dp: float
    winnable: bool  # reporting/oracle annotation ONLY (plan D3)
    gold: dict[str, float]  # gold-citable news_id -> pub_ts
    gold_groups: list[dict]  # oracle payload (story, confidence, articles)
    no_attribution: bool  # labeler found nothing (vs gold filtered out)
    status: str = "open"  # open | covered_news | covered_timing | miss
    covered_by: "Alert | None" = None
    credit: float = 0.0  # time credit (TC-recall numerator)

    def to_dict(self) -> dict:
        d = {
            "market_id": self.market_id, "date": self.date,
            "t_move_start": _iso_ts(self.t_start), "direction": self.direction,
            "status": self.status, "credit": round(self.credit, 6),
            "notified_at": _iso_ts(self.covered_by.tau) if self.covered_by else None,
            "news_id": self.covered_by.news_id if self.covered_by else None,
        }
        if self.covered_by is not None:
            d["notification_latency_hours"] = round(
                (self.t_start - self.covered_by.tau) / 3600, 4)
        return d


@dataclass
class Alert:
    market_id: str
    news_id: str
    direction: str
    tau: float
    pub: float  # cited article pub_ts
    resolve_t: float  # tau + W
    status: str = "pending"  # pending | covering_news | covering_timing |
    #                          stale | wrong_direction | false_alarm
    excused: bool = False  # pair rule: wrong_direction on a missed breakout
    resolved_t: float | None = None

    def to_dict(self) -> dict:
        d = {"market_id": self.market_id, "news_id": self.news_id,
             "direction": self.direction, "at": _iso_ts(self.tau),
             "status": self.status}
        if self.excused:
            d["excused"] = True
        return d


def load_breakpoints(data_dir: Path, tcfg: BreakoutNewsPMConfig,
                     attributions_path: Path | None = None) -> list[Breakpoint]:
    """Join built breakpoints with attributions; apply the confidence floor
    and the W-hour gold-citability window from config (the built
    attributions are unfiltered so thresholds stay live). `attributions_path`
    selects the oracle source (default data_dir/attributions.jsonl)."""
    windows: dict[str, list[tuple[float, float]]] = {}
    for w in tcfg.markets:
        windows.setdefault(w.market_id, []).append((_ts(w.start), _ts(w.end)))
    attrs = {}
    with open(attributions_path or data_dir / "attributions.jsonl") as f:
        for line in f:
            e = json.loads(line)
            attrs[(e["market_id"], e["date"])] = e
    w_s = tcfg.breakout_window_hours * 3600
    out = []
    with open(data_dir / "breakpoints.jsonl") as f:
        for line in f:
            b = json.loads(line)
            mid = b["market_id"]
            if not any(lo <= b["t_move_start"] <= hi
                       for lo, hi in windows.get(mid, ())):
                continue
            e = attrs[(mid, b["date"])]
            gold: dict[str, float] = {}
            groups = []
            for g in e["groups"]:
                if g["confidence"] < tcfg.attr_threshold:
                    continue
                arts = [a for a in g["articles"]
                        if b["t_move_start"] - w_s <= a["pub_ts"]
                        < b["t_move_start"]]
                if not arts:
                    continue
                for a in arts:
                    gold[a["news_id"]] = float(a["pub_ts"])
                groups.append({"story": g["story"],
                               "confidence": g["confidence"],
                               "articles": [
                                   {"news_id": a["news_id"],
                                    "published_at": _iso_ts(a["pub_ts"])}
                                   for a in arts]})
            out.append(Breakpoint(
                market_id=mid, date=b["date"], t_start=b["t_move_start"],
                t_end=b["t_move_end"],
                direction="up" if b["dp"] > 0 else "down", dp=b["dp"],
                winnable=bool(gold), gold=gold, gold_groups=groups,
                no_attribution=e["no_attribution"]))
    out.sort(key=lambda b: (b.t_start, b.market_id, b.date))
    return out


# -- scorer ---------------------------------------------------------------------------


class Scorer:
    """Event-ordered settlement: breakpoint closes at
    t_move_start and claims eligible alerts; unclaimed alerts resolve at
    tau + W, after every breakpoint they could still claim has closed."""

    def __init__(self, tcfg: BreakoutNewsPMConfig, news: NewsStore,
                 breakpoints: list[Breakpoint]):
        self._tcfg = tcfg
        self._news = news
        self._w_s = tcfg.breakout_window_hours * 3600
        self.breakpoints = breakpoints
        self.alerts: list[Alert] = []
        self._events: list[OutcomeEvent] = []
        self._next_bp = 0

    # -- recording ----------------------------------------------------------------

    def record_notification(self, sim_time: datetime, market_id: str,
                            news_id: str, direction: str) -> None:
        now = _ts(sim_time)
        if direction not in DIRECTIONS:
            raise NotificationError(
                f"direction must be 'up' or 'down', got {direction!r}")
        if not any(w.market_id == market_id and _ts(w.start) <= now <= _ts(w.end)
                   for w in self._tcfg.markets):
            raise NotificationError(
                f"market {market_id!r} is not monitored at {iso(sim_time)}")
        pub = self._news.published_ts(news_id)
        if pub is None:
            raise NotificationError(f"unknown news_id {news_id!r}")
        if pub > now:
            raise NotificationError(f"news {news_id!r} is not yet published")
        # one standing claim per market: the lockout is the
        # claim's own lifetime — no constant, no duplicate machinery
        pending = next((a for a in self.alerts
                        if a.market_id == market_id and a.status == "pending"),
                       None)
        if pending is not None:
            raise NotificationError(
                f"market {market_id!r} already has a pending prediction "
                f"(made {_iso_ts(pending.tau)}); it resolves by "
                f"{_iso_ts(pending.resolve_t)} — one standing claim per "
                f"market")
        self.alerts.append(Alert(market_id=market_id, news_id=news_id,
                                 direction=direction, tau=now, pub=pub,
                                 resolve_t=now + self._w_s))

    # -- settlement ---------------------------------------------------------------

    def _eligible(self, a: Alert, bp: Breakpoint) -> bool:
        return (a.market_id == bp.market_id and a.tau < bp.t_start
                and bp.t_start <= a.tau + self._w_s
                and a.direction == bp.direction)

    def _close_breakpoint(self, bp: Breakpoint) -> None:
        # the one-claim rule makes at most one alert eligible per market
        eligible = [a for a in self.alerts
                    if a.status == "pending" and self._eligible(a, bp)]
        eligible.sort(key=lambda a: (a.tau, a.news_id))
        if eligible:
            first = eligible[0]
            first.resolved_t = bp.t_start
            bp.covered_by = first
            if first.news_id in bp.gold:
                first.status = "covering_news"
                bp.status = "covered_news"
                lead_s = bp.t_start - first.pub
                frac = 1.0 if lead_s <= 0 else min(
                    1.0, max(0.0, (first.tau - first.pub) / lead_s))
                bp.credit = 1.0 - frac
            else:
                first.status = "covering_timing"
                bp.status = "covered_timing"
                bp.credit = self._tcfg.timing_credit
        else:
            bp.status = "miss"  # uniform: winnable never branches (plan D3)
            bp.credit = 0.0
        self._events.append(OutcomeEvent(
            f"{bp.market_id}@{_iso_ts(bp.t_start)}", bp.status,
            {"credit": round(bp.credit, 6)}))

    def _resolve_alert(self, a: Alert) -> None:
        stale = any(b.status != "open" and b.market_id == a.market_id
                    and a.news_id in b.gold and b.t_start <= a.tau
                    for b in self.breakpoints)
        if stale:
            a.status = "stale"
        else:
            paired = [b for b in self.breakpoints
                      if b.market_id == a.market_id
                      and a.tau < b.t_start <= a.tau + self._w_s
                      and b.direction != a.direction]
            if paired:
                a.status = "wrong_direction"
                # Pair rule: the opposite-direction breakpoint, if missed,
                # already scores credit 0 in full — attempting a direction
                # must never rank worse than silence, so the alert is
                # excused (excluded from the precision denominator).
                a.excused = any(b.status == "miss" for b in paired)
            else:
                a.status = "false_alarm"
        a.resolved_t = a.resolve_t
        self._events.append(OutcomeEvent(
            f"{a.market_id}/{a.news_id}", a.status,
            {"excused": True} if a.excused else {}))

    def _settle(self, upto: float) -> list[OutcomeEvent]:
        # closes first on ties so a breakpoint claims its alert before the
        # alert can expire
        while True:
            bp = (self.breakpoints[self._next_bp]
                  if self._next_bp < len(self.breakpoints) else None)
            pend = [a for a in self.alerts if a.status == "pending"
                    and a.resolve_t <= upto]
            pend.sort(key=lambda a: (a.resolve_t, a.market_id, a.news_id))
            al = pend[0] if pend else None
            if bp is not None and bp.t_start <= upto and (
                    al is None or bp.t_start <= al.resolve_t):
                self._close_breakpoint(bp)
                self._next_bp += 1
            elif al is not None:
                self._resolve_alert(al)
            else:
                break
        out, self._events = self._events, []
        return out

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self._settle(_ts(now))

    def close_all(self) -> list[OutcomeEvent]:
        return self._settle(float("inf"))

    # -- reporting ----------------------------------------------------------------

    def feedback(self) -> list[dict]:
        """Closed breakpoints and resolved alerts, oldest first — outcome
        statuses only, no gold article sets (gold internals are
        oracle-only)."""
        out = [{"kind": "breakpoint", "t": b.t_start, **b.to_dict()}
               for b in self.breakpoints if b.status != "open"]
        out += [{"kind": "alert", "t": a.tau, **a.to_dict()}
                for a in self.alerts if a.status != "pending"]
        out.sort(key=lambda d: (d["t"], d["kind"]))
        for d in out:
            del d["t"]
        return out

    def oracle_outcomes(self, since_ts: float | None, now_ts: float) -> list[dict]:
        lo = since_ts if since_ts is not None else float("-inf")
        out = []
        for b in self.breakpoints:
            if b.status != "open" and lo < b.t_start <= now_ts:
                rec = {"kind": "breakpoint", "t_settled": _iso_ts(b.t_start),
                       **b.to_dict(), "winnable": b.winnable}
                if not b.winnable:
                    rec["no_attributable_news"] = True
                    rec["reason"] = ("no_attribution" if b.no_attribution
                                     else "gold_filtered_out")
                else:
                    rec["gold_groups"] = b.gold_groups
                out.append(rec)
        for a in self.alerts:
            if a.resolved_t is not None and lo < a.resolved_t <= now_ts:
                out.append({"kind": "alert", "t_settled": _iso_ts(a.resolved_t),
                            **a.to_dict()})
        out.sort(key=lambda d: (d["t_settled"], d["kind"],
                                d.get("news_id") or ""))
        return out

    def metrics(self) -> dict:
        closed = [b for b in self.breakpoints if b.status != "open"]
        # v2 primary: binary coverage credit, no gold-cite distinction,
        # recall denominated over ALL closed breakpoints — the winnable
        # split comes from one hindsight labeling pass, too weak a basis
        # to drop breakpoints from the metric.
        all_covered = [b for b in closed if b.status.startswith("covered")]
        cov_recall = len(all_covered) / len(closed) if closed else None
        resolved = [a for a in self.alerts if a.status != "pending"]
        good = [a for a in resolved
                if a.status in ("covering_news", "covering_timing")]
        denom = [a for a in resolved if not a.excused]
        precision = len(good) / len(denom) if denom else None

        def _f1(p: float | None, r: float | None) -> float | None:
            if p is None or r is None:
                return None
            return 0.0 if p + r == 0 else 2 * p * r / (p + r)

        cov_f1 = _f1(precision, cov_recall)
        # secondary (TC-F1): gold-cite decay credit over winnable only
        winnable = [b for b in closed if b.winnable]
        tc_recall = (sum(b.credit for b in winnable) / len(winnable)
                     if winnable else None)
        tc_f1 = _f1(precision, tc_recall)
        covered = [b for b in winnable if b.status.startswith("covered")]
        dir_right = sum(1 for a in self.alerts if a.status in
                        ("covering_news", "covering_timing"))
        dir_wrong = sum(1 for a in self.alerts
                        if a.status == "wrong_direction")

        def _r(x: float | None) -> float | None:
            return round(x, 4) if x is not None else None

        return {
            "primary": {"name": "cov_f1", "value": _r(cov_f1),
                        "direction": "max"},
            "cov_recall": _r(cov_recall),
            "precision": _r(precision),
            "breakpoints_covered": len(all_covered),
            "breakpoints_closed": len(closed),
            "alerts_resolved": len(resolved),
            "false_alarms": sum(1 for a in resolved
                                if a.status == "false_alarm"),
            "direction_accuracy": _r(dir_right / (dir_right + dir_wrong)
                                     if dir_right + dir_wrong else None),
            "tc_f1": _r(tc_f1),
            "tc_recall": _r(tc_recall),
            "coverage_rate": (round(len(covered) / len(winnable), 4)
                              if winnable else None),
            "winnable_breakpoints": len(winnable),
        }

    def report(self) -> dict:
        closed = [b for b in self.breakpoints if b.status != "open"]

        def metric_block(bps: list[Breakpoint]) -> dict:
            n = len(bps)
            covered = [b for b in bps if b.status.startswith("covered")]
            lat = sorted((b.t_start - b.covered_by.tau) / 3600 for b in covered)
            block = {
                "breakpoints": n,
                "covered": len(covered),
                "covered_news": sum(1 for b in bps
                                    if b.status == "covered_news"),
                "covered_timing": sum(1 for b in bps
                                      if b.status == "covered_timing"),
                "miss": sum(1 for b in bps if b.status == "miss"),
                "coverage_rate": round(len(covered) / n, 4) if n else None,
            }
            if lat:
                block["notification_latency_hours"] = {
                    "median": round(statistics.median(lat), 3),
                    "p10": round(lat[int(0.1 * len(lat))], 3),
                    "p90": round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 3),
                }
            return block

        winnable = [b for b in closed if b.winnable]
        unwinnable = [b for b in closed if not b.winnable]

        # direction accuracy: among timing-matched alerts, how many called
        # the direction right (covering + duplicate matched; wrong_direction
        # matched timing but not direction)
        dir_right = sum(1 for a in self.alerts if a.status in
                        ("covering_news", "covering_timing"))
        dir_wrong = sum(1 for a in self.alerts
                        if a.status == "wrong_direction")

        n_notifs = len(self.alerts)
        n_news_correct = sum(1 for a in self.alerts
                             if a.status == "covering_news")
        n_win_covered_news = sum(1 for b in winnable
                                 if b.status == "covered_news")
        precision = n_news_correct / n_notifs if n_notifs else None
        recall = (n_win_covered_news / len(winnable)) if winnable else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        pub_lat = sorted((a.tau - a.pub) / 3600 for a in self.alerts
                         if a.status == "covering_news")

        per_market: dict[str, dict] = {}
        monthly: dict[str, dict] = {}

        def bucket(d: dict, key: str) -> dict:
            return d.setdefault(key, {
                "breakpoints": 0, "covered_news": 0, "covered_timing": 0,
                "miss": 0, "false_alarm": 0, "wrong_direction": 0, "stale": 0,
                "credit": 0.0})

        for b in closed:
            for d, key in ((per_market, b.market_id),
                           (monthly, _iso_ts(b.t_start)[:7])):
                bk = bucket(d, key)
                bk["breakpoints"] += 1
                if b.status in ("covered_news", "covered_timing", "miss"):
                    bk[b.status if b.status != "miss" else "miss"] += 1
                bk["credit"] = round(bk["credit"] + b.credit, 6)
        for a in self.alerts:
            if a.status in ("stale", "false_alarm", "wrong_direction"):
                for d, key in ((per_market, a.market_id),
                               (monthly, _iso_ts(a.tau)[:7])):
                    bk = bucket(d, key)
                    bk[a.status] += 1

        return {
            "price_centric": {
                "all": metric_block(closed),
                "winnable": metric_block(winnable),
                "unwinnable": metric_block(unwinnable),
                "direction_accuracy": (round(dir_right / (dir_right + dir_wrong), 4)
                                       if dir_right + dir_wrong else None),
            },
            "news_centric": {
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "publish_latency_hours": {
                    "median": round(statistics.median(pub_lat), 3),
                    "p90": round(pub_lat[min(len(pub_lat) - 1,
                                             int(0.9 * len(pub_lat)))], 3),
                } if pub_lat else None,
            },
            "per_market": dict(sorted(per_market.items())),
            "monthly_outcomes": dict(sorted(monthly.items())),
            # closed rows only: an open row carries a future t_move_start,
            # and a report can be written before sim_end (a paused run's
            # results_partial.json, harness/checkpoint.py) — for a finished
            # run close_all has closed everything, so nothing changes there
            "breakpoints": [{**b.to_dict(), "winnable": b.winnable}
                            for b in self.breakpoints if b.status != "open"],
            "alerts": [a.to_dict() for a in self.alerts],
        }


# -- task -----------------------------------------------------------------------------


class BreakoutNewsPMTask(Task):
    name = "breakout_news_pm"

    def __init__(self, tcfg: BreakoutNewsPMConfig, markets_meta: dict[str, dict],
                 prices: PriceStore, news: NewsStore,
                 scorer: Scorer):
        self.tcfg = tcfg
        self.markets_meta = markets_meta
        self.prices = prices
        self.news = news
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig, repo_root: Path) -> "BreakoutNewsPMTask":
        tcfg = BreakoutNewsPMConfig(**cfg.task_params)
        for w in tcfg.markets:
            if w.start < cfg.sim_start or w.end > cfg.sim_end:
                raise ValueError(
                    f"market {w.market_id}: monitoring window must lie within "
                    f"[sim_start, sim_end]")
        data_dir = tcfg.resolve_data_dir(repo_root)
        meta = {}
        with open(data_dir / "markets.jsonl") as f:
            for line in f:
                m = json.loads(line)
                meta[m["market_id"]] = m
        mids = sorted({w.market_id for w in tcfg.markets})
        unknown = [m for m in mids if m not in meta]
        if unknown:
            raise ValueError(f"unknown market_id(s): {unknown}")
        prices = PriceStore(data_dir, mids, tcfg.price_delay_minutes)
        news = NewsStore(tcfg.resolve_index_dir(repo_root))
        scorer = Scorer(tcfg, news, load_breakpoints(
            data_dir, tcfg, tcfg.resolve_attributions_path(repo_root)))
        return cls(tcfg, {m: meta[m] for m in mids}, prices, news, scorer)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.breakout_news_pm.env.apps import MarketsApp

        return [MarketsApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        market_id = payload.get("market_id")
        news_id = payload.get("news_id")
        direction = payload.get("direction")
        if not (isinstance(market_id, str) and isinstance(news_id, str)
                and isinstance(direction, str)):
            raise NotificationError(
                "payload needs string 'market_id', 'news_id' and 'direction' "
                f"('up'|'down'), got: {payload!r}")
        self.scorer.record_notification(sim_time, market_id, news_id, direction)

    # -- authored wait programs (TM-B) ---------------------------------------------------

    def authored_example(self) -> str | None:
        return (Path(__file__).resolve().parent / "agent" /
                "example_gatekeeper.py").read_text(encoding="utf-8")

    # -- scoring ---------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.scorer.close_due(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.scorer.close_all()

    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        return self.scorer.oracle_outcomes(
            _ts(since) if since is not None else None, _ts(now))

    def metrics(self) -> dict:
        return self.scorer.metrics()

    def report(self) -> dict:
        return self.scorer.report()

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        from harness.limits import RateLimiter

        cost = self.tcfg.cost
        rows = []
        for w in self.tcfg.markets:
            m = self.markets_meta[w.market_id]
            rows.append(f"| `{w.market_id}` | {m['question']} "
                        f"| {iso(w.start)} | {iso(w.end)} |")
        return {
            "markets_table": "\n".join(rows),
            "n_markets": len(self.tcfg.markets),
            "breakout_window_hours": f"{self.tcfg.breakout_window_hours:g}",
            "price_delay_minutes": f"{self.tcfg.price_delay_minutes:g}",
            "search_top_k": self.tcfg.search_top_k,
            "news_search_call": f"${cost.news_search_call:g}",
            "article_call": f"${cost.article_call:g}",
            "timing_credit": f"{self.tcfg.timing_credit:g}",
            "news_rate_limit": RateLimiter(
                "news", self.tcfg.news_rate_limit).doc(),
            "price_rate_limit": RateLimiter(
                "prices", self.tcfg.price_rate_limit).doc(),
        }


TASK = BreakoutNewsPMTask
