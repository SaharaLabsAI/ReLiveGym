"""Market-resolution detection over Polymarket questions.

The agent watches a roster of standing questions and, per question, may
commit ONE claim ever (`mark_outcome`): "this question's outcome is
already decided in the world — it is X". No retraction. A claim settles
when the question resolves:

  covered       outcome = y, tau >= t_det
                credit = exp(-(tau - t_det) / LATE_DECAY_S)
  early         outcome = y, tau < t_det, question winnable
                credit = exp(-(t_det - tau) / EARLY_DECAY_S)
  fa_premature  the question has no in-window determination (incl. every
                never-resolving one, settled at close_all)
  fa_wrong      outcome != y
  miss          winnable question, no claim

  tc_f1 = harmonic mean of  recall = sum(credits) / n_winnable
                       and  precision = sum(credits) / n_claims_settled

Both decay clocks are FIXED (independent of the question's gap) and
asymmetric — early costs 4x per hour vs late (the oracle optimum stays
at mark@0.99, perfect keeps a +0.34 margin, blanket-No 0.016). cov_f1
stays the undecayed binary companion (covered only).

t_det (determination time) and gap are BUILD-TIME ground truth pinned in
data/built/questions.jsonl under the frozen settlement constants
(theta 0.99 / exit band 0.95 / grace 0 / gap-normalized decay); this
module never touches prices, and no price surface exists anywhere.

Questions are visible only from their real open date (staggered
arrivals); a resolution becomes an observable world fact at closedTime,
for everyone. Claims are acknowledged when accepted and settle silently —
their category is only knowable at resolution. Claims standing on
questions that never resolve in-run settle as fa_premature at close_all:
that is the quiet-majority pressure that makes claim discipline the
measured skill.
"""

from __future__ import annotations

import json
import math
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
# shared substrate: the bnpm CC-NEWS world with the v3 ground-truth clock
# (crawl-corroborated publish times: self-report iff crawl within 2 h,
# else crawl - 2 h, outage-window self-reports trusted — baked in at
# index build, news/build_index.py --ts-v3)
DEFAULT_INDEX_DIR = (TASK_DIR.parent / "breakout_news_pm" / "news"
                     / "tantivy_index_v3")

WORLD_START = datetime(2026, 3, 1, tzinfo=timezone.utc)
WORLD_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

# soft-metric decay clocks (frozen task constants)
EARLY_DECAY_S = 12 * 3600
LATE_DECAY_S = 48 * 3600


def _ts(t: datetime) -> float:
    return t.timestamp()


def _iso_ts(t: float) -> str:
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))


# -- config ---------------------------------------------------------------------------


class DetectCostConfig(BaseModel):
    # commercial news-API anchor, inherited from bnpm unchanged
    news_search_call: float = 0.002
    article_call: float | None = None  # defaults to news_search_call
    mark_call: float = 0.0  # one-shot commitment is the economics, not a fee

    @model_validator(mode="after")
    def _defaults(self) -> "DetectCostConfig":
        if self.article_call is None:
            self.article_call = self.news_search_call
        return self


class ResolutionDetectConfig(BaseModel):
    questions: list[str]  # question_ids; roster scope IS this list
    search_top_k: int = 10
    data_dir: Path | None = None  # default: tasks/resolution_detect/data/built
    news_index_dir: Path | None = None  # default: bnpm tantivy_index_v3
    cost: DetectCostConfig = DetectCostConfig()
    news_rate_limit: dict = {"window": "fixed_window", "window_seconds": 60,
                             "budget": 60}

    @model_validator(mode="after")
    def _nonempty(self) -> "ResolutionDetectConfig":
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
class Claim:
    at: float
    outcome: str
    news_id: str | None


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
    t_det: float | None  # frozen predicate, build-time; None = unwinnable
    gap_s: float | None
    winnable: bool  # run_scored and t_det exists
    provenance: dict = field(default_factory=dict)  # report slices
    # runtime state
    claim: Claim | None = None
    settled: bool = False
    category: str | None = None  # covered|fa_premature|fa_wrong|miss|silent
    credit: float = 0.0
    delay_s: float | None = None  # tau - t_det for covered claims

    def resolved_public(self, now_ts: float) -> bool:
        return self.closed_ts is not None and self.closed_ts <= now_ts


def load_questions(data_dir: Path, tcfg: ResolutionDetectConfig,
                   sim_start: datetime,
                   sim_end: datetime) -> dict[str, Question]:
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
                f"the config")
        if open_ts >= t1:
            raise ValueError(f"question {qid} opens at or after sim_end")
        activation = max(open_ts, t0)
        t_res = s["t_res"]
        run_scored = bool(s["scored"] and t_res is not None
                          and activation < t_res <= t1)
        winnable = bool(run_scored and s["winnable"]
                        and s["t_det"] is not None)
        out[qid] = Question(
            qid=qid, question=a["question"], description=a["description"],
            outcomes=list(a["outcomes"]), open_ts=open_ts,
            sched_end_ts=sched_end_ts, activation=activation,
            run_scored=run_scored,
            t_res=float(t_res) if run_scored else None,
            closed_ts=closed_ts,
            answer=s["resolution_answer"],
            t_det=(float(s["t_det"]) if winnable else None),
            gap_s=(float(s["gap_s"]) if winnable else None),
            winnable=winnable,
            provenance={"family": s["family"], "volume_usd": s["volume_usd"],
                        "event_id": s["event_id"], "tags": s["tags"],
                        "trap": s["trap"],
                        "news_lead_flag": s["news_lead_flag"]})
    return out


# -- scorer ---------------------------------------------------------------------------


class Scorer:
    def __init__(self, tcfg: ResolutionDetectConfig,
                 questions: dict[str, Question], news=None):
        self._tcfg = tcfg
        self.questions = questions
        self._news = news  # published-article check for the citation param
        self._events: list[OutcomeEvent] = []

    # -- recording ----------------------------------------------------------------

    def record_mark(self, sim_time: datetime, question_id: str,
                    outcome, news_id=None) -> None:
        now = _ts(sim_time)
        q = self.questions.get(question_id)
        # unknown and not-yet-visible ids must be indistinguishable (a
        # distinct error would leak future arrivals)
        if q is None or q.activation > now:
            raise NotificationError(f"unknown question_id {question_id!r}")
        # No already-resolved rejection: resolutions
        # are invisible to the agent, and a distinct reject here would be
        # a free probe channel for exactly that bit. Post-resolution
        # claims are accepted like any other and priced by the fixed
        # decay clock (credit exp(-(tau - t_det)/48h) is already ~0 by
        # the time post-resolution coverage is abundant) — copying pays
        # nothing extra over claiming just before the close, which was
        # always possible.
        if q.claim is not None:
            raise NotificationError(
                f"question {question_id!r} already has your claim "
                f"({q.claim.outcome!r} at {_iso_ts(q.claim.at)}) — one claim "
                f"per question, it cannot be changed")
        if not isinstance(outcome, str) or outcome not in q.outcomes:
            raise NotificationError(
                f"outcome must be one of {q.outcomes}, got {outcome!r}")
        if news_id is not None:
            if not isinstance(news_id, str) or not news_id:
                raise NotificationError("news_id must be a non-empty string")
            pub = self._news.published_ts(news_id) if self._news else None
            if self._news is not None and (pub is None or pub > now):
                raise NotificationError(
                    f"unknown or not-yet-published news_id {news_id!r}")
        q.claim = Claim(at=now, outcome=outcome, news_id=news_id)

    # -- settlement ---------------------------------------------------------------

    def _settle_resolved(self, q: Question) -> None:
        """Category + credit for a run-scored question at its resolution."""
        c = q.claim
        if c is None:
            q.category = "miss" if q.winnable else "silent"
        elif c.outcome != q.answer:
            q.category = "fa_wrong"
        elif not q.winnable:
            q.category = "fa_premature"  # no determination ever existed
        elif c.at < q.t_det:
            q.category = "early"
            q.credit = math.exp(-(q.t_det - c.at) / EARLY_DECAY_S)
            q.delay_s = c.at - q.t_det  # negative
        else:
            q.category = "covered"
            q.credit = math.exp(-(c.at - q.t_det) / LATE_DECAY_S)
            q.delay_s = c.at - q.t_det
        q.settled = True
        detail = {"outcome": q.answer, "category": q.category,
                  "credit": round(q.credit, 6)}
        if c is not None:
            detail["claim"] = {"at": _iso_ts(c.at), "outcome": c.outcome,
                               "news_id": c.news_id}
        self._events.append(OutcomeEvent(
            ref=q.qid, status="resolved", detail=detail))

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        # only CLAIMED questions settle incrementally: an unclaimed
        # resolved question must stay open to a late claim (claims after
        # the invisible resolution are legal), so its
        # miss is booked at close_all
        now_ts = _ts(now)
        due = [q for q in self.questions.values()
               if q.run_scored and not q.settled and q.claim is not None
               and q.resolved_public(now_ts)]
        due.sort(key=lambda q: (q.closed_ts, q.qid))
        for q in due:
            self._settle_resolved(q)
        out, self._events = self._events, []
        return out

    def close_all(self) -> list[OutcomeEvent]:
        due = [q for q in self.questions.values()
               if q.run_scored and not q.settled]
        due.sort(key=lambda q: (q.closed_ts, q.qid))
        for q in due:
            self._settle_resolved(q)
        # claims standing on questions that never resolved in-run: the
        # world had not settled them — premature by definition (the
        # quiet-majority pressure)
        standing = [q for q in self.questions.values()
                    if not q.run_scored and q.claim is not None
                    and not q.settled]
        standing.sort(key=lambda q: q.qid)
        for q in standing:
            q.category = "fa_premature"
            q.settled = True
            self._events.append(OutcomeEvent(
                ref=q.qid, status="unresolved",
                detail={"category": "fa_premature",
                        "claim": {"at": _iso_ts(q.claim.at),
                                  "outcome": q.claim.outcome,
                                  "news_id": q.claim.news_id}}))
        out, self._events = self._events, []
        return out

    # -- reporting ----------------------------------------------------------------

    def _settled(self) -> list[Question]:
        return [q for q in self.questions.values() if q.settled]

    def oracle_outcomes(self, since_ts: float | None,
                        now_ts: float) -> list[dict]:
        lo = since_ts if since_ts is not None else float("-inf")
        out = []
        for q in self._settled():
            if q.closed_ts is None or not (lo < q.closed_ts <= now_ts):
                continue
            rec = {"kind": "question", "question_id": q.qid,
                   "t_settled": _iso_ts(q.closed_ts), "outcome": q.answer,
                   "your_claim": None,
                   "category": q.category,
                   "credit": round(q.credit, 6)}
            if q.claim is not None:
                rec["your_claim"] = {"at": _iso_ts(q.claim.at),
                                     "outcome": q.claim.outcome,
                                     "news_id": q.claim.news_id}
            out.append(rec)
        out.sort(key=lambda d: (d["t_settled"], d["question_id"]))
        return out

    def metrics(self) -> dict:
        done = self._settled()
        winnable = [q for q in done if q.winnable]
        claims = [q for q in done if q.claim is not None]
        covered = [q for q in claims if q.category == "covered"]
        credited = [q for q in claims if q.category in ("covered", "early")]
        n_win, n_claims = len(winnable), len(claims)
        credit_sum = sum(q.credit for q in credited)

        def f1(rec: float, prec: float) -> float | None:
            if rec is None or prec is None:
                return None
            return (round(2 * prec * rec / (prec + rec), 4)
                    if prec + rec > 0 else 0.0)

        recall = round(credit_sum / n_win, 4) if n_win else None
        precision = (round(credit_sum / n_claims, 4)
                     if n_claims else (0.0 if n_win else None))
        cov_recall = round(len(covered) / n_win, 4) if n_win else None
        cov_precision = (round(len(covered) / n_claims, 4)
                         if n_claims else (0.0 if n_win else None))
        delays = sorted(q.delay_s for q in credited)  # signed: early < 0
        return {
            "primary": {"name": "tc_f1", "value": f1(recall, precision),
                        "direction": "max"},
            "cov_f1": f1(cov_recall, cov_precision),
            "precision": precision,
            "recall": recall,
            "n_winnable": n_win,
            "n_claims_settled": n_claims,
            "n_covered": len(covered),
            "n_early": sum(1 for q in done if q.category == "early"),
            "n_fa_premature": sum(1 for q in done
                                  if q.category == "fa_premature"),
            "n_fa_wrong": sum(1 for q in done if q.category == "fa_wrong"),
            "n_miss": sum(1 for q in done if q.category == "miss"),
            "median_delay_h": (round(delays[len(delays) // 2] / 3600, 2)
                               if delays else None),
        }

    def report(self) -> dict:
        done = sorted(self._settled(),
                      key=lambda q: (q.closed_ts or float("inf"), q.qid))

        def row(q: Question) -> dict:
            return {
                "question_id": q.qid, "question": q.question,
                "outcome": q.answer, "category": q.category,
                "credit": round(q.credit, 6),
                "winnable": q.winnable,
                "gap_hours": (round(q.gap_s / 3600, 2)
                              if q.gap_s is not None else None),
                "delay_hours": (round(q.delay_s / 3600, 2)
                                if q.delay_s is not None else None),
                "claim": ({"at": _iso_ts(q.claim.at),
                           "outcome": q.claim.outcome,
                           "news_id": q.claim.news_id}
                          if q.claim else None),
                "trap": q.provenance["trap"],
                "news_lead_flag": q.provenance["news_lead_flag"],
                "family": q.provenance["family"],
            }

        def slice_by(key) -> dict:
            groups: dict[str, list[Question]] = {}
            for q in done:
                groups.setdefault(key(q), []).append(q)
            return {k: {"n": len(v),
                        "covered": sum(1 for q in v
                                       if q.category == "covered"),
                        "early": sum(1 for q in v if q.category == "early"),
                        "credit": round(sum(q.credit for q in v), 4)}
                    for k, v in sorted(groups.items())}

        def gap_band(q: Question) -> str:
            if q.gap_s is None:
                return "unwinnable"
            h = q.gap_s / 3600
            return ("<6h" if h < 6 else "6-24h" if h < 24
                    else "24-96h" if h < 96 else ">=96h")

        fa = [q for q in done if q.category == "fa_premature"]
        open_qs = [q for q in self.questions.values() if not q.settled]
        return {
            "questions": [row(q) for q in done],
            "by_family": slice_by(lambda q: q.provenance["family"]),
            "by_gap_band": slice_by(gap_band),
            "by_category": slice_by(lambda q: q.category),
            "fa_decomposition": {
                "quiet_question": sum(1 for q in fa if not q.run_scored),
                "unwinnable_question": sum(1 for q in fa
                                           if q.run_scored and not q.winnable),
            },
            "unclaimed_open": [
                {"question_id": q.qid}
                for q in sorted(open_qs, key=lambda q: q.qid)
                if q.claim is None],
            "standing_claims_open": [
                {"question_id": q.qid, "at": _iso_ts(q.claim.at),
                 "outcome": q.claim.outcome}
                for q in sorted(open_qs, key=lambda q: q.qid)
                if q.claim is not None],
        }


# -- task -----------------------------------------------------------------------------


class ResolutionDetectTask(Task):
    name = "resolution_detect"

    def __init__(self, tcfg: ResolutionDetectConfig, data_dir: Path,
                 news, scorer: Scorer):
        self.tcfg = tcfg
        self.data_dir = data_dir
        self.news = news
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig,
                        repo_root: Path) -> "ResolutionDetectTask":
        from tasks.breakout_news_pm.task import NewsStore  # shared substrate

        tcfg = ResolutionDetectConfig(**cfg.task_params)
        if as_utc(cfg.sim_start) < WORLD_START or as_utc(cfg.sim_end) > WORLD_END:
            raise ValueError(
                f"sim window must lie within the data world "
                f"[{iso(WORLD_START)}, {iso(WORLD_END)}]")
        data_dir = tcfg.resolve_data_dir(repo_root)
        questions = load_questions(data_dir, tcfg, cfg.sim_start, cfg.sim_end)
        news = NewsStore(tcfg.resolve_index_dir(repo_root))
        scorer = Scorer(tcfg, questions, news=news)
        return cls(tcfg, data_dir, news, scorer)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.resolution_detect.env.apps import DetectApp

        return [DetectApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        qid = payload.get("question_id")
        if not isinstance(qid, str):
            raise NotificationError(
                f"payload needs string 'question_id' and string 'outcome' "
                f"(+ optional 'news_id'), got: {payload!r}")
        self.scorer.record_mark(sim_time, qid, payload.get("outcome"),
                                payload.get("news_id"))

    # -- question visibility (consumed by env/apps.py) --------------------------------

    def question_index(self, now: datetime,
                       added_after: datetime | None = None) -> list[dict]:
        # NO resolution signal of any kind: a
        # deployed resolution-detection agent has no oracle telling it a
        # question has settled — that determination is the task itself.
        # Resolved questions stay listed, indistinguishable from open
        # ones; outcome/close time/status all scorer-only. (Earlier that
        # day only outcome+resolved_at were dropped; the status flip
        # was still free supervision and went too.)
        now_ts = _ts(now)
        out = []
        for q in self.scorer.questions.values():
            if q.activation > now_ts:
                continue
            if added_after is not None and q.activation <= _ts(added_after):
                continue
            out.append({"question_id": q.qid, "question": q.question,
                        "added_at": _iso_ts(q.activation)})
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

    def mark_history(self, qid: str | None, now: datetime) -> list[dict]:
        """Own-claim echo: the standing claim per question. No categories,
        no scores — nothing the agent didn't itself send."""
        now_ts = _ts(now)
        qs = self.scorer.questions.values()
        if qid is not None:
            q = self.scorer.questions.get(qid)
            if q is None or q.activation > now_ts:
                raise NotificationError(f"unknown question_id {qid!r}")
            qs = [q]
        out = []
        for q in qs:
            if q.claim is None:
                continue
            rec = {"question_id": q.qid, "at": _iso_ts(q.claim.at),
                   "outcome": q.claim.outcome}
            if q.claim.news_id is not None:
                rec["news_id"] = q.claim.news_id
            out.append(rec)
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


TASK = ResolutionDetectTask
