"""Reddit AI-subreddit popularity task: recommend discussion threads that
hindsight proves popular, promptly. The
distinguishing feature is the *timestamped comment cascade* — the agent watches
the reply tree grow in simulated time and calls a trend early.

World data (data/build.py, Arctic Shift): roots.jsonl (one root submission per
row) + cascades.jsonl ({root_id, nodes[]} raw reply tree, per-node created_utc).

Popularity is the log-rescaled final cascade size: pop = log2(max(d, 1)) + 1,
where d = `descendants` = number of comment nodes created within
horizon_hours of the root. There is NO score anywhere: `score` (root and
comment) is an ingestion-time snapshot and is never loaded or served.

Visibility. A root's static metadata (id, subreddit, author, title, selftext,
url, ...) is visible from its post time. The label `descendants` stays hidden
until reveal = post + horizon_hours (24 h). The comment cascade is revealed as
a GROWING PREFIX: get_cascade returns only nodes with created_utc <= now, with
each node's `score` stripped (a final snapshot would leak the outcome). History
before sim_start (history_days) is fully revealed. Time filters clip to now.
Every post view carries `num_comments` = comments posted by now (the
real-time cascade size, as the real Reddit listing does); list_posts can sort
by it (`sort=comments`) — a ranking on the CURRENT snapshot, never on the
label — so nothing about the future can move a post in or out of a page.

Scoring (metrics, not dollars). The
positive class P is the tail: posts whose final `descendants >=
tail_min_desc`. Each accepted recommendation r = (root s, sim time tau)
settles at reveal(s) = post(s) + H with a timeliness weight over
D = decay_hours (default: D = H):

    weight(r) = max(0, 1 - (tau - post(s)) / D)

Primary metric: TWR@cap (time-weighted tail recall under the rolling
daily_cap) = sum over p in P of weight(first rec of p) / |P|. Junk
recommendations self-punish by wasting cap slots; precision, plain
recall, and latency are secondaries. The API bills at Reddit's real
commercial rate (api_call) behind the documented OAuth-client rate
limit; the binding resource is the run's budget_usd (LLM tokens).

Recommendation validation (free rejection, no leakage — every rule is a
function of public info): unknown or not-yet-posted root_id (one
indistinguishable message), already revealed (label public then), posted
before sim_start (history), duplicate root_id, cap reached. Posts
revealing after sim_end still settle via close_all().
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import iso

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TASK_DIR / "data" / "built_min0"

# Root fields served in a RootPostView. NB: the source `score`, `num_comments`,
# and `retrieved_on` are deliberately absent — all three are ingestion-time
# final snapshots that would leak the outcome. The `num_comments` a view DOES
# carry is recomputed from the cascade prefix at now (see visible_view).
ROOT_VIEW_FIELDS = ("id", "name", "subreddit", "author", "title", "selftext",
                    "url", "is_self", "over_18", "link_flair_text", "domain")


def pop_of(descendants: int) -> float:
    return math.log2(max(descendants, 1)) + 1


def _ts(dt: datetime) -> float:
    return dt.timestamp()


def _iso_ts(t: float) -> str:
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))


# -- config ---------------------------------------------------------------------------


class RedditCostConfig(BaseModel):
    # Reddit's real commercial Data API rate ($0.24 per 1k calls, basis:
    # documented 2023 pricing), uniform across endpoints like the real API
    api_call: float = 0.00024


class RedditPopularityConfig(BaseModel):
    history_days: int = 30  # revealed lookback before sim_start
    horizon_hours: float = 24.0  # H: reveal delay == label window
    # Reward-decay window, decoupled from the label horizon: decay =
    # max(0, 1 - latency/decay_hours), so a recommendation later than
    # decay_hours after posting earns nothing while the post stays
    # recommendable (and its label hidden) until horizon_hours. None falls
    # back to horizon_hours so pre-existing runs and tests replay identically.
    decay_hours: float | None = None
    daily_cap: int = 10  # accepted recommendations per rolling 24 h
    tail_min_desc: int = 50  # positive class: final descendants >= this
    page_size: int = 50  # results per list_posts / nodes per get_cascade page
    data_dir: Path | None = None  # default: tasks/<task>/data/built_min0
    cost: RedditCostConfig = RedditCostConfig()
    # Reddit OAuth-client limit (basis: documented — 100 QPM averaged over
    # a 10-minute window). Advertised in INSTRUCTION.md, enforced with 429,
    # never metered for the agent.
    rate_limit: dict = {"window": "fixed_window", "window_seconds": 600,
                        "budget": 1000}

    @model_validator(mode="after")
    def _positive(self) -> "RedditPopularityConfig":
        if self.horizon_hours <= 0 or self.daily_cap < 1 or self.page_size < 1:
            raise ValueError("horizon_hours, daily_cap, page_size must be positive")
        if self.tail_min_desc < 1:
            raise ValueError("tail_min_desc must be positive")
        if self.decay_hours is not None and not (
                0 < self.decay_hours <= self.horizon_hours):
            raise ValueError("decay_hours must satisfy 0 < decay_hours "
                             "<= horizon_hours")
        return self

    @property
    def effective_decay_hours(self) -> float:
        return self.decay_hours if self.decay_hours is not None \
            else self.horizon_hours

    def resolve_data_dir(self, base_dir: Path | None = None) -> Path:
        path = self.data_dir
        if path is None:
            return DEFAULT_DATA_DIR
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path


# -- stores ---------------------------------------------------------------------------


class CascadeStore:
    """Reply trees keyed by root_id. Each node kept as (created_utc, id, kind,
    parent_id, author, body) — NO score. Nodes per root are sorted ascending by
    created_utc so a revealed prefix (created_utc <= now) is a bisect slice and,
    because replies are causal (parent precedes child), always a connected
    subtree rooted at the post."""

    def __init__(self, path: Path, keep_ids: set[str]):
        self.nodes: dict[str, list[dict]] = {}
        self._times: dict[str, list[float]] = {}
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                rid = c["root_id"]
                if rid not in keep_ids:
                    continue
                ns = [{"id": n["id"], "kind": n["kind"],
                       "parent_id": n["parent_id"], "author": n["author"],
                       "created_utc": n["created_utc"], "body": n["body"]}
                      for n in c["nodes"]]
                ns.sort(key=lambda n: (n["created_utc"], n["id"]))
                self.nodes[rid] = ns
                self._times[rid] = [n["created_utc"] for n in ns]

    def descendants_upto(self, root_id: str, upto_ts: float) -> int:
        """Number of COMMENT nodes with created_utc <= upto_ts (root excluded)."""
        ns = self.nodes.get(root_id)
        if not ns:
            return 0
        k = bisect_right(self._times[root_id], upto_ts)
        # subtract the root if it falls in the slice (it always does for
        # upto_ts >= post, but count only comments)
        return sum(1 for n in ns[:k] if n["kind"] == "comment")

    def prefix(self, root_id: str, now_ts: float, offset: int,
               page_size: int) -> dict:
        """Revealed prefix: nodes with created_utc <= now, score-stripped,
        paged. Includes running counts of the current observable prefix."""
        ns = self.nodes.get(root_id, [])
        k = bisect_right(self._times.get(root_id, []), now_ts)
        revealed = ns[:k]
        n_comments = sum(1 for n in revealed if n["kind"] == "comment")
        page = revealed[offset:offset + page_size]
        return {
            "root_id": root_id,
            "now": _iso_ts(now_ts),
            "n_nodes": len(revealed),
            "n_comments": n_comments,
            "nodes": [dict(n) for n in page],
            "offset": offset,
            "has_more": offset + page_size < len(revealed),
        }


class RootStore:
    """Root submissions sorted by (created_utc, id); the `descendants` label is
    visible only from post + horizon (the reveal)."""

    def __init__(self, path: Path, lo_ts: float, hi_ts: float, horizon_s: float,
                 cascades: CascadeStore):
        self._h = horizon_s
        self._cascades = cascades
        self.rows: list[dict] = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if lo_ts <= r["created_utc"] <= hi_ts:
                    row = {k: r.get(k) for k in ROOT_VIEW_FIELDS}
                    row["created_utc"] = r["created_utc"]
                    # final label: comments within H of the root (fixed per root)
                    row["_label"] = cascades.descendants_upto(
                        r["id"], r["created_utc"] + horizon_s)
                    self.rows.append(row)
        self.rows.sort(key=lambda r: (r["created_utc"], r["id"]))
        self._times = [r["created_utc"] for r in self.rows]
        self.by_id = {r["id"]: r for r in self.rows}

    def get(self, root_id: str) -> dict | None:
        return self.by_id.get(root_id)

    def rows_between(self, lo_ts: float, hi_ts: float) -> list[dict]:
        return self.rows[bisect_left(self._times, lo_ts):
                         bisect_right(self._times, hi_ts)]

    def visible_view(self, r: dict, now_ts: float, include_text: bool = True) -> dict:
        revealed = r["created_utc"] + self._h <= now_ts
        v = {k: r[k] for k in ROOT_VIEW_FIELDS}
        if not include_text:
            v.pop("selftext", None)
        v["created_utc"] = r["created_utc"]
        v["posted_at"] = _iso_ts(r["created_utc"])
        v["reveal_at"] = _iso_ts(r["created_utc"] + self._h)
        v["revealed"] = revealed
        # real-time cascade size: comments posted by now (the listing field
        # the real Reddit API exposes); equals the label once revealed
        v["num_comments"] = self._cascades.descendants_upto(r["id"], now_ts)
        v["descendants"] = r["_label"] if revealed else None
        return v

    def query(self, since: datetime | None, until: datetime | None,
              subreddit: str | None, order: str, offset: int, page_size: int,
              now: datetime, sort: str = "new") -> dict:
        now_ts = _ts(now)
        lo = _ts(since) if since else (self._times[0] if self._times else 0.0)
        hi = min(_ts(until), now_ts) if until else now_ts
        cands = self.rows_between(lo, hi)
        if subreddit is not None:
            sub = subreddit.lower()
            cands = [r for r in cands if (r["subreddit"] or "").lower() == sub]
        # `new` = time order (store order). `comments` = current cascade size
        # at now — a snapshot ranking, NOT the label, so no outcome leaks;
        # ties fall back to time. `order` flips either.
        if sort == "comments":
            cands = sorted(cands, key=lambda r: (
                self._cascades.descendants_upto(r["id"], now_ts),
                r["created_utc"], r["id"]))
        if order == "desc":
            cands = cands[::-1]
        page = cands[offset:offset + page_size]
        return {
            "sort": sort,
            "total_hits": len(cands),
            "offset": offset,
            "posts": [self.visible_view(r, now_ts, include_text=False)
                      for r in page],
            "clipped_until": _iso_ts(hi),
            "has_more": offset + page_size < len(cands),
        }


# -- scorer ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    root_id: str
    title: str
    subreddit: str
    tau: float  # recommendation sim time
    post: float  # root post time
    reveal_t: float  # post + horizon
    status: str = "pending"  # pending | tail | nontail
    descendants: int | None = None
    pop: float | None = None
    weight: float | None = None  # timeliness weight at settlement

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "title": self.title,
            "subreddit": self.subreddit,
            "recommended_at": _iso_ts(self.tau),
            "posted_at": _iso_ts(self.post),
            "delay_hours": round((self.tau - self.post) / 3600, 3),
            "descendants": self.descendants,
            "pop": round(self.pop, 4) if self.pop is not None else None,
            "weight": round(self.weight, 4) if self.weight is not None else None,
            "status": self.status,
        }


class Scorer:
    """Per-root settlement at the reveal: each accepted recommendation
    settles as tail/nontail with its timeliness weight (cost-free
    outcome events; the metric is TWR@cap)."""

    def __init__(self, tcfg: RedditPopularityConfig, roots: RootStore,
                 sim_start: datetime, sim_end: datetime):
        self._tcfg = tcfg
        self._roots = roots
        self._h = tcfg.horizon_hours * 3600
        self._decay_s = tcfg.effective_decay_hours * 3600
        self._sim_start = _ts(sim_start)
        self._sim_end = _ts(sim_end)
        self.recs: list[Recommendation] = []
        self._rec_ids: set[str] = set()
        self._settled: list[Recommendation] = []  # settle order

    # -- recording ----------------------------------------------------------------

    def used_last_24h(self, now_ts: float) -> int:
        return sum(1 for r in self.recs if r.tau > now_ts - 86400)

    def record_notification(self, sim_time: datetime, root_id: str) -> None:
        now = _ts(sim_time)
        s = self._roots.get(root_id)
        # one message for nonexistent and future ids: a rejection must not
        # reveal that an unposted id exists
        if s is None or s["created_utc"] > now:
            raise NotificationError(
                f"unknown or not yet posted root_id {root_id!r}")
        post = s["created_utc"]
        if now >= post + self._h:
            raise NotificationError(
                f"post {root_id!r} is already revealed; too late to recommend")
        if post < self._sim_start:
            raise NotificationError(
                f"post {root_id!r} predates the run; history posts cannot "
                f"be recommended")
        if root_id in self._rec_ids:
            raise NotificationError(f"post {root_id!r} was already recommended")
        if self.used_last_24h(now) >= self._tcfg.daily_cap:
            raise NotificationError(
                f"recommendation cap reached: {self._tcfg.daily_cap} per "
                f"rolling 24h window")
        self.recs.append(Recommendation(
            root_id=root_id, title=s["title"], subreddit=s["subreddit"],
            tau=now, post=post, reveal_t=post + self._h))
        self._rec_ids.add(root_id)

    # -- settlement -----------------------------------------------------------------

    def _settle(self, upto: float) -> list[OutcomeEvent]:
        due = [r for r in self.recs if r.status == "pending" and r.reveal_t <= upto]
        due.sort(key=lambda r: (r.reveal_t, r.root_id))
        events = []
        for r in due:
            r.descendants = self._roots.get(r.root_id)["_label"]
            r.pop = pop_of(r.descendants)
            r.weight = max(0.0, 1 - (r.tau - r.post) / self._decay_s)
            r.status = ("tail" if r.descendants >= self._tcfg.tail_min_desc
                        else "nontail")
            self._settled.append(r)
            events.append(OutcomeEvent(
                r.root_id, r.status,
                {"weight": round(r.weight, 4),
                 "descendants": r.descendants}))
        return events

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self._settle(_ts(now))

    def close_all(self) -> list[OutcomeEvent]:
        return self._settle(float("inf"))

    # -- reporting ----------------------------------------------------------------

    def feedback(self) -> list[dict]:
        """Settled recommendations only (labels are revealed there), in
        settle order."""
        return [r.to_dict() for r in self._settled]

    def oracle_outcomes(self, since_ts: float | None, now_ts: float) -> list[dict]:
        lo = since_ts if since_ts is not None else float("-inf")
        out = [{"kind": "recommendation", "t_settled": _iso_ts(r.reveal_t),
                **r.to_dict()}
               for r in self._settled if lo < r.reveal_t <= now_ts]
        out.sort(key=lambda d: (d["t_settled"], d["root_id"]))
        return out

    def _tail_universe(self) -> list[dict]:
        """The positive class P: window posts with a tail-sized final
        label. Posts created inside [sim_start, sim_end)."""
        return [r for r in self._roots.rows_between(self._sim_start,
                                                    self._sim_end - 1e-6)
                if r["_label"] >= self._tcfg.tail_min_desc]

    def metrics(self) -> dict:
        universe = self._tail_universe()
        tail_recs = [r for r in self._settled if r.status == "tail"]
        twr = (sum(r.weight for r in tail_recs) / len(universe)
               if universe else None)
        precision = (len(tail_recs) / len(self._settled)
                     if self._settled else None)
        recall = (len(tail_recs) / len(universe) if universe else None)
        delays = sorted((r.tau - r.post) / 3600 for r in self._settled)
        # weight-1 idealization of the cap: the best conceivable TWR
        by_day: dict[str, int] = {}
        for r in universe:
            d = _iso_ts(r["created_utc"])[:10]
            by_day[d] = by_day.get(d, 0) + 1
        ceiling = (sum(min(self._tcfg.daily_cap, n) for n in by_day.values())
                   / len(universe) if universe else None)
        return {
            "primary": {"name": "twr_at_cap",
                        "value": round(twr, 4) if twr is not None else None,
                        "direction": "max"},
            "precision": (round(precision, 4)
                          if precision is not None else None),
            "recall": round(recall, 4) if recall is not None else None,
            "tail_posts": len(universe),
            "recommendations": len(self.recs),
            "settled": len(self._settled),
            "tail_hits": len(tail_recs),
            "median_delay_hours": (round(delays[len(delays) // 2], 3)
                                   if delays else None),
            "twr_ceiling_at_cap": (round(ceiling, 4)
                                   if ceiling is not None else None),
        }

    def report(self) -> dict:
        daily: dict[str, dict] = {}
        for r in self._settled:
            bk = daily.setdefault(_iso_ts(r.tau)[:10], {
                "recommendations": 0, "tail": 0, "nontail": 0,
                "weight_sum": 0.0})
            bk["recommendations"] += 1
            bk[r.status] += 1
            if r.status == "tail":
                bk["weight_sum"] = round(bk["weight_sum"] + r.weight, 4)
        return {
            "pending": len(self.recs) - len(self._settled),
            "daily_outcomes": dict(sorted(daily.items())),
            "recommendations": [r.to_dict() for r in self.recs],
        }


# -- task -----------------------------------------------------------------------------


class RedditPopularityTask(Task):
    name = "reddit_ai_popularity"

    def __init__(self, tcfg: RedditPopularityConfig, history_start: datetime,
                 roots: RootStore, cascades: CascadeStore, scorer: Scorer):
        self.tcfg = tcfg
        self.history_start = history_start
        self.roots = roots
        self.cascades = cascades
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig, repo_root: Path) -> "RedditPopularityTask":
        tcfg = RedditPopularityConfig(**cfg.task_params)
        data_dir = tcfg.resolve_data_dir(repo_root)
        stats_path = data_dir / "build_stats.json"
        if not stats_path.exists():
            raise ValueError(f"no built data at {data_dir} (run data/build.py)")
        stats = json.loads(stats_path.read_text())
        history_start = cfg.sim_start - timedelta(days=tcfg.history_days)
        # coverage: build_stats records the ROOT window as unix `after`/`before`
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
        # load cascades only for roots in the window (bounds memory)
        keep = _root_ids_in_window(data_dir / "roots.jsonl", lo_ts, hi_ts)
        cascades = CascadeStore(data_dir / "cascades.jsonl", keep)
        roots = RootStore(data_dir / "roots.jsonl", lo_ts, hi_ts, horizon_s,
                          cascades)
        scorer = Scorer(tcfg, roots, cfg.sim_start, cfg.sim_end)
        return cls(tcfg, history_start, roots, cascades, scorer)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.reddit_ai_popularity.env.apps import RedditApp, RedditOracleApp

        apps = [RedditApp(sim, self)]
        if sim.cfg.cell.sig == "oracle":
            apps.append(RedditOracleApp(sim, self))
        return apps

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        root_id = payload.get("root_id")
        if not isinstance(root_id, str) or not root_id:
            raise NotificationError(
                f"payload needs a non-empty string 'root_id', got: {payload!r}")
        self.scorer.record_notification(sim_time, root_id)

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
        """Own recommendation settlements (every one, either class) PLUS
        the tail posts the agent did NOT recommend — the recall failures.
        Non-recommended nontail posts stay out of the feed: at ~97% of
        settlements they would drown the signal, and their labels are
        public in listings anyway. Every record carries a `growth`
        trajectory (settled = revealed; no leakage). With no cursor the
        stream begins at sim_start."""
        now_ts = _ts(now)
        since_ts = _ts(since) if since is not None else self.scorer._sim_start
        out = self.scorer.oracle_outcomes(since_ts, now_ts)
        for o in out:
            o["growth"] = self._growth(o["root_id"])
        h = self.tcfg.horizon_hours * 3600
        for r in self.roots.rows_between(since_ts - h + 1e-6, now_ts - h):
            if (r["id"] in self.scorer._rec_ids
                    or r["_label"] < self.tcfg.tail_min_desc):
                continue
            out.append({
                "kind": "post", "t_settled": _iso_ts(r["created_utc"] + h),
                "root_id": r["id"], "subreddit": r["subreddit"],
                "title": r["title"], "posted_at": _iso_ts(r["created_utc"]),
                "descendants": r["_label"],
                "pop": round(pop_of(r["_label"]), 4),
                # judged env-side (tail_min_desc is a task param the
                # agent-side records module must not hard-code)
                "status": "tail",
                "growth": self._growth(r["id"]),
            })
        out.sort(key=lambda d: (d["t_settled"], d.get("root_id", "")))
        return out

    def _growth(self, root_id: str) -> dict:
        """Comment counts at fixed ages after posting — how the cascade
        evolved inside the label horizon."""
        post = self.roots.get(root_id)["created_utc"]
        return {f"{hh:g}h": self.cascades.descendants_upto(
                    root_id, post + hh * 3600)
                for hh in (0.5, 1, 2, 3, 6, 12, 24)
                if hh <= self.tcfg.horizon_hours}

    def metrics(self) -> dict:
        return self.scorer.metrics()

    def report(self) -> dict:
        return self.scorer.report()

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        from harness.limits import RateLimiter

        d = self.tcfg.effective_decay_hours
        pts = [h for h in (0.5, 1, 2, 3, 4, 5, 8, 12, 18) if h < d]
        examples = "; ".join(f"{h:g} h late → {1 - h / d:.2f}"
                             for h in pts) + f"; ≥{d:g} h late → 0"
        return {
            "api_call": f"${self.tcfg.cost.api_call:g}",
            "rate_limit": RateLimiter("reddit", self.tcfg.rate_limit).doc(),
            "tail_min_desc": self.tcfg.tail_min_desc,
            "daily_cap": self.tcfg.daily_cap,
            "horizon_hours": f"{self.tcfg.horizon_hours:g}",
            "decay_hours": f"{self.tcfg.effective_decay_hours:g}",
            "decay_examples": examples,
            "page_size": self.tcfg.page_size,
            "history_start": iso(self.history_start),
        }


def _root_ids_in_window(path: Path, lo_ts: float, hi_ts: float) -> set[str]:
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if lo_ts <= r["created_utc"] <= hi_ts:
                ids.add(r["id"])
    return ids


TASK = RedditPopularityTask
