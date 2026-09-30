"""Standing-forecast maintenance over Polymarket questions.

The agent maintains a probability distribution per question
(`submit_forecast`); the standing forecast is piecewise-constant between
submissions and is scored CONTINUOUSLY against the eventual resolution:

    BSS(t) = 1 - sum_{o in outcomes} (p_o(t) - 1[o = y])^2
    TA_q   = (1 / |life_q|) * integral of BSS_q(t) dt

Before the first submission the UNIFORM distribution stands in (1/n per
outcome, +0.5 on this dataset's all-binary questions), so the metric
reads as informativeness above uninformed rather than as coverage.

life_q = [activation, t_res], activation = max(open_date, sim_start),
t_res = min(closedTime, scheduled_end) — UMA settlement lag earns nothing.
Both the forecast and the (scorer-only) market price series are step
functions, so the integral is computed exactly per segment; no grid.

Questions are visible only from their real open date (staggered arrivals);
a resolution becomes an observable world fact at closedTime, for everyone —
that is observability, not a sig treatment. Market prices are withheld from
the agent everywhere and drive only the `ta_bss_market` anchor and the
`easy` reporting flag (both recomputed at run time over the run's actual
life, never a settlement branch).

Scored set of a run: questions with a clean recorded resolution whose t_res
lies inside (activation, sim_end]. Everything else in the portfolio is live
attention load — the agent cannot tell which is which. Primary metric:
`ta_bss` = mean TA_q over the run-scored questions, direction max.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import as_utc, iso, parse_iso

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TASK_DIR / "data" / "built"
# shared substrate: the bnpm CC-NEWS world, end to end — on the v3
# ground-truth clock (crawl-corroborated
# publish times: self-report iff crawl within 2 h, else crawl - 2 h,
# outage-window self-reports trusted). A min(self-report, crawl)
# rule would let backdated articles serve before the crawler proved
# they existed, which under a time-averaged score converts every leaked
# hour straight into ta_bss.
DEFAULT_INDEX_DIR = (TASK_DIR.parent / "breakout_news_pm" / "news"
                     / "tantivy_index_v3")

WORLD_START = datetime(2026, 3, 1, tzinfo=timezone.utc)
WORLD_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

SUM_TOL = 1e-9


def _ts(t: datetime) -> float:
    return t.timestamp()


def _iso_ts(t: float) -> str:
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))


# -- config ---------------------------------------------------------------------------


class ForecastCostConfig(BaseModel):
    # commercial news-API anchor, inherited from bnpm unchanged
    news_search_call: float = 0.002
    article_call: float | None = None  # defaults to news_search_call
    submit_call: float = 0.0  # proper scoring already disincentivizes thrash

    @model_validator(mode="after")
    def _defaults(self) -> "ForecastCostConfig":
        if self.article_call is None:
            self.article_call = self.news_search_call
        return self


class ForecastPortfolioConfig(BaseModel):
    questions: list[str]  # question_ids; portfolio scope IS this list
    search_top_k: int = 10
    easy_band: float = 0.95  # `easy` = price never left [band, 1] or [0, 1-band]
    data_dir: Path | None = None  # default: tasks/forecast_portfolio/data/built
    news_index_dir: Path | None = None  # default: bnpm tantivy_index_v3
    cost: ForecastCostConfig = ForecastCostConfig()
    news_rate_limit: dict = {"window": "fixed_window", "window_seconds": 60,
                             "budget": 60}

    @model_validator(mode="after")
    def _nonempty(self) -> "ForecastPortfolioConfig":
        if not self.questions:
            raise ValueError("task.questions must list at least one question")
        if len(set(self.questions)) != len(self.questions):
            raise ValueError("task.questions has duplicates")
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


# -- questions ------------------------------------------------------------------------


@dataclass
class Question:
    qid: str
    question: str
    description: str  # verbatim resolution criteria
    outcomes: list[str]
    open_ts: float
    sched_end_ts: float
    activation: float  # max(open_ts, sim_start)
    # ground truth (scorer-only)
    run_scored: bool  # clean resolution with t_res in (activation, sim_end]
    t_res: float | None
    closed_ts: float | None  # public-resolution moment (feedback visibility)
    answer: str | None
    provenance: dict = field(default_factory=dict)  # report slices
    # runtime state — trace entries are (t, dist, news_id): the citation
    # is RECORDED, NEVER SCORED (an evidence-provenance instrument, so a
    # run can be audited for what a belief actually rested on)
    trace: list[tuple[float, dict[str, float], str | None]] = field(
        default_factory=list)
    settled: bool = False
    ta: float | None = None
    ta_market: float | None = None
    easy: bool | None = None
    abstain_share: float | None = None

    def dist_at(self, t: float) -> dict[str, float] | None:
        cur = None
        for tt, d, _ in self.trace:
            if tt > t:
                break
            cur = d
        return cur

    def resolved_public(self, now_ts: float) -> bool:
        return self.closed_ts is not None and self.closed_ts <= now_ts


def _submission(t: float, dist: dict[str, float],
                news_id: str | None) -> dict:
    """One own-submission echo, shared by get_forecasts and the oracle."""
    rec = {"at": _iso_ts(t), "forecast": dist}
    if news_id is not None:
        rec["news_id"] = news_id
    return rec


def _bss(dist: dict[str, float] | None, outcomes: list[str], y: str) -> float:
    """Brier skill of the distribution in force. NO STANDING FORECAST
    SCORES AS THE UNIFORM ONE: an unattended question
    earns 1/n on every outcome, i.e. +0.5 on the binary questions that
    are all this dataset holds — not 0. So the metric measures
    informativeness above uninformed, and an arm cannot be ranked by how
    fast its scaffold got a placeholder in. Note this is the default only
    for the ABSENT forecast; once a distribution is submitted it is
    scored literally, and an outcome the agent omits is a stated zero,
    not a fallback to 1/n (INSTRUCTION says so in as many words)."""
    if not dist:
        dist = {o: 1.0 / len(outcomes) for o in outcomes}
    return 1.0 - sum((dist.get(o, 0.0) - (1.0 if o == y else 0.0)) ** 2
                     for o in outcomes)


def _segments(trace: list[tuple[float, dict[str, float], str | None]],
              lo: float, hi: float):
    """(t_from, t_to, dist-in-force) covering [lo, hi] exactly."""
    cur: dict[str, float] | None = None
    cuts = [lo]
    dists = []
    for tt, d, _ in trace:
        if tt <= lo:
            cur = d
        elif tt < hi:
            dists.append(cur)
            cuts.append(tt)
            cur = d
    dists.append(cur)
    cuts.append(hi)
    return [(a, b, d) for a, b, d in zip(cuts, cuts[1:], dists) if b > a]


def integrate_ta(trace: list[tuple[float, dict[str, float], str | None]],
                 outcomes: list[str], y: str, lo: float,
                 hi: float) -> tuple[float, float]:
    """(TA, abstention share) over [lo, hi] — exact, piecewise-constant."""
    total = 0.0
    absent = 0.0
    for a, b, d in _segments(trace, lo, hi):
        total += _bss(d, outcomes, y) * (b - a)
        if not d:
            absent += b - a
    return total / (hi - lo), absent / (hi - lo)


def market_trace(points: list[list[float]]) -> list[tuple[float, dict, None]]:
    """Price change-points as a forecast trace: Yes-price p read as the
    full-mass distribution {Yes: p, No: 1-p}. Quoteless lead-in (before
    the first point) stays abstention, mirroring the agent's no-forecast
    rule."""
    return [(float(t), {"Yes": float(p), "No": round(1.0 - float(p), 10)},
             None) for t, p in points]


def load_questions(data_dir: Path, tcfg: ForecastPortfolioConfig,
                   sim_start: datetime, sim_end: datetime) -> dict[str, Question]:
    rows = {}
    with open(data_dir / "questions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["question_id"]] = r
    unknown = [q for q in tcfg.questions if q not in rows]
    if unknown:
        raise ValueError(f"unknown question_id(s): {unknown}")
    t0, t1 = _ts(sim_start), _ts(sim_end)
    out: dict[str, Question] = {}
    for qid in tcfg.questions:
        r = rows[qid]
        a, s = r["agent"], r["scorer"]
        open_ts = _ts(as_utc(parse_iso(a["open_date"])))
        sched_end_ts = _ts(as_utc(parse_iso(a["scheduled_end"])))
        closed_ts = (_ts(as_utc(parse_iso(s["closed_time"])))
                     if s["closed_time"] else None)
        if closed_ts is not None and closed_ts < t0:
            raise ValueError(
                f"question {qid} resolved before sim_start — drop it from "
                f"the config (it would be dead weight the agent can see "
                f"through)")
        if open_ts >= t1:
            raise ValueError(f"question {qid} opens at or after sim_end")
        activation = max(open_ts, t0)
        t_res = s["t_res"]
        run_scored = bool(s["scored"] and t_res is not None
                          and activation < t_res <= t1)
        out[qid] = Question(
            qid=qid, question=a["question"], description=a["description"],
            outcomes=list(a["outcomes"]), open_ts=open_ts,
            sched_end_ts=sched_end_ts, activation=activation,
            run_scored=run_scored,
            t_res=float(t_res) if run_scored else None,
            closed_ts=closed_ts,
            answer=s["resolution_answer"],
            provenance={"family": s["family"], "volume_usd": s["volume_usd"],
                        "event_id": s["event_id"], "tags": s["tags"]})
    return out


# -- scorer ---------------------------------------------------------------------------


class Scorer:
    def __init__(self, tcfg: ForecastPortfolioConfig, data_dir: Path,
                 questions: dict[str, Question], news=None):
        self._tcfg = tcfg
        self._data_dir = data_dir
        self.questions = questions
        self._news = news  # published-article check for the citation param
        self._events: list[OutcomeEvent] = []

    # -- recording ----------------------------------------------------------------

    def record_forecast(self, sim_time: datetime, question_id: str,
                        forecast: dict, news_id=None) -> None:
        now = _ts(sim_time)
        q = self.questions.get(question_id)
        # unknown and not-yet-visible ids must be indistinguishable
        if q is None or q.activation > now:
            raise NotificationError(f"unknown question_id {question_id!r}")
        if q.resolved_public(now):
            raise NotificationError(
                f"question {question_id!r} is already resolved")
        if not isinstance(forecast, dict) or not forecast:
            raise NotificationError(
                "forecast must be a non-empty {outcome: probability} object")
        bad = [o for o in forecast if o not in q.outcomes]
        if bad:
            raise NotificationError(
                f"unknown outcome(s) {bad} — this question's outcomes are "
                f"{q.outcomes}")
        dist = {}
        for o, p in forecast.items():
            if isinstance(p, bool) or not isinstance(p, (int, float)) or p < 0:
                raise NotificationError(
                    f"probability for {o!r} must be a number >= 0, got {p!r}")
            dist[o] = float(p)
        if sum(dist.values()) > 1.0 + SUM_TOL:
            raise NotificationError(
                f"probabilities sum to {sum(dist.values()):g} > 1 "
                f"(a short sum is allowed — it scores the outcomes you "
                f"left out as zero; an excess is not)")
        if news_id is not None:
            if not isinstance(news_id, str) or not news_id:
                raise NotificationError("news_id must be a non-empty string")
            pub = self._news.published_ts(news_id) if self._news else None
            if self._news is not None and (pub is None or pub > now):
                raise NotificationError(
                    f"unknown or not-yet-published news_id {news_id!r}")
        q.trace.append((now, dist, news_id))

    # -- settlement ---------------------------------------------------------------

    def _prices(self, qid: str) -> list[list[float]]:
        return json.loads(
            (self._data_dir / "prices" / f"{qid}.json").read_text())["points"]

    def _settle(self, q: Question) -> None:
        q.ta, q.abstain_share = integrate_ta(
            q.trace, q.outcomes, q.answer, q.activation, q.t_res)
        pts = self._prices(q.qid)
        q.ta_market, _ = integrate_ta(
            market_trace(pts), q.outcomes, q.answer, q.activation, q.t_res)
        band = self._tcfg.easy_band
        levels = [p for t, p in pts if q.activation < t <= q.t_res]
        lead = [p for t, p in pts if t <= q.activation]
        if lead:
            levels.insert(0, lead[-1])
        q.easy = bool(levels) and (min(levels) >= band
                                   or max(levels) <= 1.0 - band)
        q.settled = True
        n_updates = len(q.trace)
        self._events.append(OutcomeEvent(
            ref=q.qid, status="resolved",
            detail={"outcome": q.answer, "ta_bss": round(q.ta, 6),
                    "ta_bss_market": round(q.ta_market, 6),
                    "n_updates": n_updates}))

    def _due(self, pending_filter) -> list[OutcomeEvent]:
        due = [q for q in self.questions.values()
               if q.run_scored and not q.settled and pending_filter(q)]
        due.sort(key=lambda q: (q.closed_ts, q.qid))
        for q in due:
            self._settle(q)
        out, self._events = self._events, []
        return out

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        now_ts = _ts(now)
        return self._due(lambda q: q.resolved_public(now_ts))

    def close_all(self) -> list[OutcomeEvent]:
        return self._due(lambda q: True)

    # -- reporting ----------------------------------------------------------------

    def _settled(self) -> list[Question]:
        return [q for q in self.questions.values() if q.settled]

    def oracle_outcomes(self, since_ts: float | None,
                        now_ts: float) -> list[dict]:
        # sig-oracle = graded hindsight on questions ALREADY publicly
        # resolved: the outcome, the agent's own trace, and its own score
        # decomposition — so it can carry lessons to the questions still
        # open. Nothing about an unresolved question, and no
        # market-derived number: `ta_bss_market` is computed from the
        # price series the agent never sees anywhere , so
        # handing the learning arms a per-question market-quality
        # readout would be a price leak dressed as feedback.
        lo = since_ts if since_ts is not None else float("-inf")
        out = []
        for q in self._settled():
            if lo < q.closed_ts <= now_ts:
                out.append({
                    "kind": "question", "question_id": q.qid,
                    "t_settled": _iso_ts(q.closed_ts), "outcome": q.answer,
                    "ta_bss": round(q.ta, 6),
                    "abstention_share": round(q.abstain_share, 6),
                    "your_forecasts": [_submission(t, d, n)
                                       for t, d, n in q.trace],
                })
        out.sort(key=lambda d: (d["t_settled"], d["question_id"]))
        return out

    def metrics(self) -> dict:
        done = self._settled()
        nontrivial = [q for q in done if not q.easy]

        def mean(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals), 4) if vals else None

        ta = mean([q.ta for q in done])
        ta_market = mean([q.ta_market for q in done])
        return {
            "primary": {"name": "ta_bss", "value": ta, "direction": "max"},
            "ta_bss_nontrivial": mean([q.ta for q in nontrivial]),
            "ta_bss_market": ta_market,
            "skill_vs_market": (round(ta - ta_market, 4)
                                if ta is not None else None),
            "abstention_share": mean([q.abstain_share for q in done]),
            "questions_settled": len(done),
            "questions_easy": sum(1 for q in done if q.easy),
            "mean_updates": mean([float(len(q.trace)) for q in done]),
        }

    def report(self) -> dict:
        done = sorted(self._settled(), key=lambda q: q.closed_ts)

        def row(q: Question) -> dict:
            return {
                "question_id": q.qid, "question": q.question,
                "outcome": q.answer, "resolved_at": _iso_ts(q.closed_ts),
                "life_hours": round((q.t_res - q.activation) / 3600, 2),
                "ta_bss": round(q.ta, 6),
                "ta_bss_market": round(q.ta_market, 6),
                "abstention_share": round(q.abstain_share, 6),
                "n_updates": len(q.trace),
                # recorded-unscored evidence trail: how many of this
                # question's submissions named the article they rested on
                "n_cited": sum(1 for _, _, n in q.trace if n),
                "easy": q.easy,
                "family": q.provenance["family"],
            }

        def slice_by(key) -> dict:
            groups: dict[str, list[Question]] = {}
            for q in done:
                groups.setdefault(key(q), []).append(q)
            return {k: {"n": len(v),
                        "ta_bss": round(sum(q.ta for q in v) / len(v), 4),
                        "ta_bss_market": round(
                            sum(q.ta_market for q in v) / len(v), 4)}
                    for k, v in sorted(groups.items())}

        def life_band(q: Question) -> str:
            days = (q.t_res - q.activation) / 86400
            return ("<7d" if days < 7 else "7-30d" if days < 30
                    else "30-90d" if days < 90 else ">=90d")

        open_qs = [q for q in self.questions.values() if not q.settled]
        return {
            "questions": [row(q) for q in done],
            "by_family": slice_by(lambda q: q.provenance["family"]),
            "by_resolution_month": slice_by(
                lambda q: _iso_ts(q.closed_ts)[:7]),
            "by_life_band": slice_by(life_band),
            "by_easy": slice_by(lambda q: "easy" if q.easy else "nontrivial"),
            "unsettled": [
                {"question_id": q.qid,
                 "n_updates": len(q.trace),
                 "last_forecast": (q.trace[-1][1] if q.trace else None)}
                for q in sorted(open_qs, key=lambda q: q.qid)],
        }


# -- task -----------------------------------------------------------------------------


class ForecastPortfolioTask(Task):
    name = "forecast_portfolio"

    def __init__(self, tcfg: ForecastPortfolioConfig, data_dir: Path,
                 news, scorer: Scorer):
        self.tcfg = tcfg
        self.data_dir = data_dir
        self.news = news
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig,
                        repo_root: Path) -> "ForecastPortfolioTask":
        from tasks.breakout_news_pm.task import NewsStore  # shared substrate

        tcfg = ForecastPortfolioConfig(**cfg.task_params)
        if as_utc(cfg.sim_start) < WORLD_START or as_utc(cfg.sim_end) > WORLD_END:
            raise ValueError(
                f"sim window must lie within the data world "
                f"[{iso(WORLD_START)}, {iso(WORLD_END)}]")
        data_dir = tcfg.resolve_data_dir(repo_root)
        questions = load_questions(data_dir, tcfg, cfg.sim_start, cfg.sim_end)
        news = NewsStore(tcfg.resolve_index_dir(repo_root))
        scorer = Scorer(tcfg, data_dir, questions, news=news)
        return cls(tcfg, data_dir, news, scorer)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.forecast_portfolio.env.apps import ForecastApp

        return [ForecastApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        qid = payload.get("question_id")
        forecast = payload.get("forecast")
        if not isinstance(qid, str):
            raise NotificationError(
                f"payload needs string 'question_id' and object 'forecast' "
                f"({{outcome: probability}}, + optional 'news_id'), got: "
                f"{payload!r}")
        self.scorer.record_forecast(sim_time, qid, forecast,
                                    payload.get("news_id"))

    # -- question visibility (consumed by env/apps.py) --------------------------------

    def question_index(self, now: datetime, status: str | None = None,
                       added_after: datetime | None = None) -> list[dict]:
        now_ts = _ts(now)
        out = []
        for q in self.scorer.questions.values():
            if q.activation > now_ts:
                continue
            if added_after is not None and q.activation <= _ts(added_after):
                continue
            resolved = q.resolved_public(now_ts)
            if status == "open" and resolved:
                continue
            if status == "resolved" and not resolved:
                continue
            rec = {"question_id": q.qid, "question": q.question,
                   "added_at": _iso_ts(q.activation),
                   "scheduled_close": _iso_ts(q.sched_end_ts),
                   "status": "resolved" if resolved else "open"}
            if resolved:
                rec["outcome"] = q.answer
                rec["resolved_at"] = _iso_ts(q.closed_ts)
            out.append(rec)
        out.sort(key=lambda d: (d["added_at"], d["question_id"]))
        return out

    def question_detail(self, qid: str, now: datetime) -> dict:
        q = self.scorer.questions.get(qid)
        if q is None or q.activation > _ts(now):
            # identical for unknown and not-yet-visible
            raise NotificationError(f"unknown question_id {qid!r}")
        rec = next(r for r in self.question_index(now)
                   if r["question_id"] == qid)
        rec["description"] = q.description
        rec["outcomes"] = q.outcomes
        return rec

    def forecast_history(self, qid: str | None, now: datetime) -> list[dict]:
        now_ts = _ts(now)
        qs = self.scorer.questions.values()
        if qid is not None:
            q = self.scorer.questions.get(qid)
            if q is None or q.activation > now_ts:
                raise NotificationError(f"unknown question_id {qid!r}")
            qs = [q]
        out = []
        for q in qs:
            if not q.trace:
                continue
            out.append({"question_id": q.qid,
                        "current": q.trace[-1][1],
                        "history": [_submission(t, d, n)
                                    for t, d, n in q.trace]})
        out.sort(key=lambda d: d["question_id"])
        return out

    # -- authored wait programs (TM-B) -----------------------------------------------

    def authored_example(self) -> str | None:
        return (TASK_DIR / "agent" / "example_gatekeeper.py").read_text(
            encoding="utf-8")

    # -- scoring ---------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.scorer.close_due(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.scorer.close_all()

    def oracle_outcomes(self, since: datetime | None,
                        now: datetime) -> list[dict]:
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
        return {
            "search_top_k": self.tcfg.search_top_k,
            "news_search_call": f"${cost.news_search_call:g}",
            "article_call": f"${cost.article_call:g}",
            "news_rate_limit": RateLimiter(
                "news", self.tcfg.news_rate_limit).doc(),
        }


TASK = ForecastPortfolioTask
