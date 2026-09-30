"""Daily Reddit digest task: once a day, deliver the AI-subreddit posts of the
last 24 h that end up popular. A rung BELOW reddit_ai_popularity on the same
world — nothing to predict, only timing discipline and cost.

World: byte-identical to reddit_ai_popularity (stores, visibility rule,
fees, rate limit imported from tasks/reddit_ai_popularity). Every post view
carries `num_comments` = comments posted by now; the final label
`descendants` (comments within horizon_hours of the root) is hidden until
reveal = post + horizon_hours. Since the prefix count is monotone, a post
with `num_comments >= popular_min_desc` at delivery time is a sure hit — the
agent never has to guess.

Calendar. `a` = digest_hour_utc. A scored day d is every UTC calendar day
whose whole candidate window lies inside the run:

    window(d)   = [a:00(d) - lookback_hours, a:00(d))      (posts eligible)
    delivery(d) = [a:00(d), a:00(d) + delivery_window_hours) (digest accepted)
    settle(d)   = a:00(d) + horizon_hours                    (labels frozen)

with a:00(d) - lookback >= sim_start and a:00(d) < sim_end.

Scoring. One accepted digest(root_ids) per day (the first inside the
delivery window). picks = the first digest_size DISTINCT ids in list order;
hits = picks that are in window(d) with final descendants >= popular_min_desc;
score(d) = hits / digest_size, or 0 for a day without a digest. Primary
metric digest_score = mean of score(d) over scored days, direction max.

Free rejections (NotificationError; every rule is a function of the clock
or the payload shape, never of the world): malformed payload, call outside
every delivery window, second digest in the same window. Ids are NOT
validated at call time, so the action's response carries no information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import iso
from tasks.reddit_ai_popularity.task import (
    DEFAULT_DATA_DIR,
    CascadeStore,
    RedditCostConfig,
    RootStore,
    _iso_ts,
    _root_ids_in_window,
    _ts,
)

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent


class DailyRedditDigestConfig(BaseModel):
    history_days: int = 7  # revealed lookback before sim_start
    horizon_hours: float = 24.0  # reveal delay == label window
    digest_hour_utc: int = 12  # `a`: the pinned daily hour
    delivery_window_hours: float = 1.0  # digest accepted in [a, a + this)
    lookback_hours: float = 24.0  # candidate window = [a - this, a)
    digest_size: int = 10  # ids scored per digest (= the denominator)
    popular_min_desc: int = 50  # popular: final descendants >= this
    page_size: int = 50
    data_dir: Path | None = None  # default: the sibling's built_min0
    cost: RedditCostConfig = RedditCostConfig()
    # same documented Reddit OAuth-client limit as the sibling
    rate_limit: dict = {"window": "fixed_window", "window_seconds": 600,
                        "budget": 1000}

    @model_validator(mode="after")
    def _consistent(self) -> "DailyRedditDigestConfig":
        if not 0 <= self.digest_hour_utc < 24:
            raise ValueError("digest_hour_utc must be in 0..23")
        if self.horizon_hours <= 0 or self.page_size < 1:
            raise ValueError("horizon_hours and page_size must be positive")
        if self.delivery_window_hours <= 0 or self.lookback_hours <= 0:
            raise ValueError("delivery_window_hours and lookback_hours must "
                             "be positive")
        if self.lookback_hours > self.horizon_hours:
            raise ValueError("lookback_hours must be <= horizon_hours (every "
                             "candidate must be revealed by settlement)")
        if self.digest_size < 1 or self.popular_min_desc < 1:
            raise ValueError("digest_size and popular_min_desc must be "
                             "positive")
        return self

    def resolve_data_dir(self, base_dir: Path) -> Path:
        path = self.data_dir or DEFAULT_DATA_DIR
        if not path.is_absolute():
            path = base_dir / path
        return path


# -- calendar -------------------------------------------------------------------------


def _at(d: date, hour: int) -> float:
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc).timestamp()


@dataclass
class Digest:
    day: date
    delivered_at: float
    root_ids: list[str]
    picks: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | ok
    hits: int | None = None
    score: float | None = None
    per_id: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "day": self.day.isoformat(),
            "delivered_at": _iso_ts(self.delivered_at),
            "root_ids": list(self.root_ids),
            "picks": list(self.picks),
            "hits": self.hits,
            "score": self.score,
            "status": self.status,
            "per_id": list(self.per_id),
        }


class DigestScorer:
    """Per-day settlement at a:00 + horizon: the day's digest (if any) is
    scored against the frozen labels of its candidate window."""

    def __init__(self, tcfg: DailyRedditDigestConfig, roots: RootStore,
                 sim_start: datetime, sim_end: datetime):
        self._tcfg = tcfg
        self._roots = roots
        self._a = tcfg.digest_hour_utc
        self._look = tcfg.lookback_hours * 3600
        self._deliver = tcfg.delivery_window_hours * 3600
        self._h = tcfg.horizon_hours * 3600
        self._sim_start = _ts(sim_start)
        self._sim_end = _ts(sim_end)
        self.days: list[date] = self._scored_days(sim_start, sim_end)
        self.digests: dict[date, Digest] = {}
        self.rejections: dict[str, int] = {}
        self._settled: dict[date, dict] = {}  # day -> outcome detail

    def _scored_days(self, sim_start: datetime, sim_end: datetime) -> list[date]:
        out, d = [], sim_start.date()
        while True:
            a = _at(d, self._a)
            if a >= self._sim_end:
                break
            if a - self._look >= self._sim_start:
                out.append(d)
            d += timedelta(days=1)
        return out

    def window_for(self, d: date) -> tuple[float, float]:
        a = _at(d, self._a)
        return a - self._look, a

    def delivery_for(self, d: date) -> tuple[float, float]:
        a = _at(d, self._a)
        return a, a + self._deliver

    def settle_at(self, d: date) -> float:
        return _at(d, self._a) + self._h

    def day_for(self, now_ts: float) -> date | None:
        """The scored day whose delivery window contains now, if any."""
        today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date()
        for d in (today, today - timedelta(days=1)):
            lo, hi = self.delivery_for(d)
            if d in self.days and lo <= now_ts < hi:
                return d
        return None

    def next_delivery(self, now_ts: float) -> tuple[float, float] | None:
        for d in self.days:
            lo, hi = self.delivery_for(d)
            if now_ts < hi:
                return lo, hi
        return None

    # -- recording ----------------------------------------------------------------

    def _reject(self, reason: str, msg: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        raise NotificationError(msg)

    def record_notification(self, sim_time: datetime, root_ids) -> None:
        now = _ts(sim_time)
        if (not isinstance(root_ids, list) or not root_ids
                or not all(isinstance(r, str) and r for r in root_ids)):
            self._reject("malformed", "payload needs 'root_ids': a non-empty "
                         "list of non-empty strings")
        d = self.day_for(now)
        if d is None:
            nxt = self.next_delivery(now)
            hint = (f"; next window {_iso_ts(nxt[0])}–{_iso_ts(nxt[1])}"
                    if nxt else "; no delivery window remains in this run")
            self._reject("outside_window",
                         "not in a delivery window" + hint)
        if d in self.digests:
            self._reject("duplicate_day",
                         f"already delivered for {d.isoformat()}")
        picks: list[str] = []
        for r in root_ids:
            if r not in picks:
                picks.append(r)
            if len(picks) == self._tcfg.digest_size:
                break
        self.digests[d] = Digest(day=d, delivered_at=now,
                                 root_ids=list(root_ids), picks=picks)

    # -- settlement -----------------------------------------------------------------

    def _settle_day(self, d: date) -> OutcomeEvent:
        lo, hi = self.window_for(d)
        g = self.digests.get(d)
        if g is None:
            detail = {"score": 0.0, "hits": 0, "delivered_at": None}
            self._settled[d] = detail
            return OutcomeEvent(d.isoformat(), "miss", dict(detail))
        hits = 0
        for rid in g.picks:
            r = self._roots.get(rid)
            in_window = r is not None and lo <= r["created_utc"] < hi
            desc = r["_label"] if r is not None else None
            popular = (in_window and desc is not None
                       and desc >= self._tcfg.popular_min_desc)
            hits += popular
            g.per_id.append({"root_id": rid, "in_window": in_window,
                             "descendants": desc if in_window else None,
                             "hit": bool(popular)})
        g.hits = hits
        g.score = round(hits / self._tcfg.digest_size, 4)
        g.status = "ok"
        detail = {"score": g.score, "hits": hits,
                  "delivered_at": _iso_ts(g.delivered_at),
                  "picks": list(g.picks)}
        self._settled[d] = detail
        return OutcomeEvent(d.isoformat(), "ok", dict(detail))

    def _settle(self, upto: float) -> list[OutcomeEvent]:
        return [self._settle_day(d) for d in self.days
                if d not in self._settled and self.settle_at(d) <= upto]

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self._settle(_ts(now))

    def close_all(self) -> list[OutcomeEvent]:
        return self._settle(float("inf"))

    # -- reporting ----------------------------------------------------------------

    def oracle_outcomes(self, since_ts: float | None, now_ts: float) -> list[dict]:
        lo = since_ts if since_ts is not None else float("-inf")
        out = []
        for d, detail in self._settled.items():
            t_settled = self.settle_at(d)
            if lo < t_settled <= now_ts:
                g = self.digests.get(d)
                out.append({"kind": "digest", "day": d.isoformat(),
                            "t_settled": _iso_ts(t_settled),
                            "status": "ok" if g else "miss",
                            **(g.to_dict() if g else {"score": 0.0,
                                                      "hits": 0})})
        out.sort(key=lambda o: o["t_settled"])
        return out

    def metrics(self) -> dict:
        settled = [self._settled[d] for d in self.days if d in self._settled]
        scores = [s["score"] for s in settled]
        delivered = [self.digests[d] for d in self.days if d in self.digests]
        latency = [(g.delivered_at - _at(g.day, self._a)) / 60
                   for g in delivered]
        mean = (round(sum(scores) / len(scores), 4) if scores else None)
        return {
            "primary": {"name": "digest_score", "value": mean,
                        "direction": "max"},
            "days": len(self.days),
            "days_settled": len(settled),
            "days_delivered": len(delivered),
            "days_full": sum(1 for s in scores if s >= 1.0),
            "mean_hits": (round(sum(s["hits"] for s in settled) / len(settled), 3)
                          if settled else None),
            "delivery_latency_min_mean": (round(sum(latency) / len(latency), 2)
                                          if latency else None),
            "rejections": sum(self.rejections.values()),
        }

    def report(self) -> dict:
        return {
            "days": [d.isoformat() for d in self.days],
            "pending_days": [d.isoformat() for d in self.days
                             if d not in self._settled],
            "daily_outcomes": {d.isoformat(): self._settled[d]
                               for d in self.days if d in self._settled},
            "digests": [self.digests[d].to_dict() for d in self.days
                        if d in self.digests],
            "rejections": dict(sorted(self.rejections.items())),
        }


# -- task -----------------------------------------------------------------------------


class DailyRedditDigestTask(Task):
    name = "daily_reddit_digest"

    def __init__(self, tcfg: DailyRedditDigestConfig, history_start: datetime,
                 roots: RootStore, cascades: CascadeStore, scorer: DigestScorer):
        self.tcfg = tcfg
        self.history_start = history_start
        self.roots = roots
        self.cascades = cascades
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig, repo_root: Path) -> "DailyRedditDigestTask":
        tcfg = DailyRedditDigestConfig(**cfg.task_params)
        data_dir = tcfg.resolve_data_dir(repo_root)
        stats_path = data_dir / "build_stats.json"
        if not stats_path.exists():
            raise ValueError(f"no built data at {data_dir} (run "
                             "tasks/reddit_ai_popularity/data/build.py)")
        stats = json.loads(stats_path.read_text())
        history_start = cfg.sim_start - timedelta(days=tcfg.history_days)
        cov_lo = float(stats["window"]["after"])
        cov_hi = float(stats["window"]["before"])
        if _ts(history_start) < cov_lo or _ts(cfg.sim_end) > cov_hi:
            raise ValueError(
                f"built data covers roots [{stats['window']['after_h']}, "
                f"{stats['window']['before_h']}] but the run needs "
                f"[{iso(history_start)}, {iso(cfg.sim_end)}]; rebuild wider")
        collected_h = float(stats.get("horizon_days_collected", 7)) * 24
        if tcfg.horizon_hours > collected_h:
            raise ValueError(
                f"horizon_hours={tcfg.horizon_hours} exceeds the collected tree "
                f"depth ({collected_h} h); rebuild with a longer horizon")
        horizon_s = tcfg.horizon_hours * 3600
        lo_ts, hi_ts = _ts(history_start), _ts(cfg.sim_end)
        keep = _root_ids_in_window(data_dir / "roots.jsonl", lo_ts, hi_ts)
        cascades = CascadeStore(data_dir / "cascades.jsonl", keep)
        roots = RootStore(data_dir / "roots.jsonl", lo_ts, hi_ts, horizon_s,
                          cascades)
        scorer = DigestScorer(tcfg, roots, cfg.sim_start, cfg.sim_end)
        if not scorer.days:
            raise ValueError("no scored day fits inside the run window "
                             "(need a:00 - lookback >= sim_start and a:00 < "
                             "sim_end for at least one day)")
        return cls(tcfg, history_start, roots, cascades, scorer)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.daily_reddit_digest.env.apps import DigestApp
        from tasks.reddit_ai_popularity.env.apps import RedditReadApp

        return [RedditReadApp(sim, self), DigestApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise NotificationError(f"payload must be an object, got {payload!r}")
        self.scorer.record_notification(sim_time, payload.get("root_ids"))

    # -- authored wait programs (TM-B) ---------------------------------------------------

    def authored_example(self) -> str | None:
        return (TASK_DIR / "agent" / "example_gatekeeper.py").read_text(
            encoding="utf-8")

    # -- scoring ---------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.scorer.close_due(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.scorer.close_all()

    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        since_ts = _ts(since) if since is not None else None
        return self.scorer.oracle_outcomes(since_ts, _ts(now))

    def metrics(self) -> dict:
        return self.scorer.metrics()

    def report(self) -> dict:
        return self.scorer.report()

    def instruction_context(self) -> dict[str, object]:
        from harness.limits import RateLimiter

        sc = self.scorer
        first, last = sc.days[0], sc.days[-1]
        lo, hi = sc.window_for(first)
        dlo, dhi = sc.delivery_for(first)
        return {
            "api_call": f"${self.tcfg.cost.api_call:g}",
            "rate_limit": RateLimiter("reddit", self.tcfg.rate_limit).doc(),
            "digest_hour": f"{self.tcfg.digest_hour_utc:02d}",
            "lookback_hours": f"{self.tcfg.lookback_hours:g}",
            "delivery_window_hours": f"{self.tcfg.delivery_window_hours:g}",
            "digest_size": self.tcfg.digest_size,
            "popular_min_desc": self.tcfg.popular_min_desc,
            "horizon_hours": f"{self.tcfg.horizon_hours:g}",
            "page_size": self.tcfg.page_size,
            "history_start": iso(self.history_start),
            "n_days": len(sc.days),
            "first_day": first.isoformat(),
            "last_day": last.isoformat(),
            "example_window_lo": _iso_ts(lo),
            "example_window_hi": _iso_ts(hi),
            "example_deliver_lo": _iso_ts(dlo),
            "example_deliver_hi": _iso_ts(dhi),
            "example_settle": _iso_ts(sc.settle_at(first)),
        }


TASK = DailyRedditDigestTask
