"""Synthetic forecast_portfolio world with exact-math lifetimes.

Sim window for tests: 2026-03-01 .. 2026-03-11 (10 days). Questions
(times in whole days from t0 so integrals are exact fractions):

  q_res    opens day -2, sched_end day 4, closes day 4    -> Yes  (scored)
  q_lag    opens day 0,  sched_end day 6, closes day 6.5  -> No   (scored;
           UMA lag: life ends at sched_end, public half a day later)
  q_mid    opens day 2,  sched_end day 8, closes day 8    -> Yes  (scored,
           mid-window arrival)
  q_open   opens day -1, sched_end day 40, never closes          (load)
  q_late   opens day 1,  sched_end day 9, closes day 30   -> Yes (resolves
           after sim_end: run-unscored load, answer never public in-run)

Prices: q_res flat 0.9 (easy under band 0.85... default band 0.95 ->
0.9 is NOT easy); q_lag steps 0.4 -> 0.2 at day 3 (market TA computable
by hand); q_mid has no quotes until day 5 (quoteless lead-in = market
abstention).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tasks.forecast_portfolio.task import (
    ForecastPortfolioConfig, ForecastPortfolioTask, Scorer, load_questions,
)

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
SIM_END = T0 + timedelta(days=10)


def t(days: float) -> datetime:
    return T0 + timedelta(days=days)


def ts(days: float) -> float:
    return t(days).timestamp()


def iso_d(days: float) -> str:
    return t(days).strftime("%Y-%m-%dT%H:%M:%SZ")


QUESTIONS = [
    # qid, open_d, sched_end_d, closed_d, answer, scored, family
    ("q_res", -2, 4, 4, "Yes", True, "geopolitics"),
    ("q_lag", 0, 6, 6.5, "No", True, "econ"),
    ("q_mid", 2, 8, 8, "Yes", True, "politics"),
    ("q_open", -1, 40, None, None, False, "politics"),
    ("q_late", 1, 9, 30, "Yes", False, "culture"),
]

PRICES = {
    "q_res": [[ts(-2), 0.9]],
    "q_lag": [[ts(0), 0.4], [ts(3), 0.2]],
    "q_mid": [[ts(5), 0.8]],
    "q_open": [[ts(-1), 0.5]],
    "q_late": [[ts(1), 0.6]],
}


def write_world(root: Path) -> Path:
    built = root / "built"
    (built / "prices").mkdir(parents=True)
    with (built / "questions.jsonl").open("w") as f:
        for qid, od, ed, cd, ans, scored, fam in QUESTIONS:
            f.write(json.dumps({
                "question_id": qid,
                "agent": {
                    "question": f"Will {qid} happen?",
                    "description": f"Resolution criteria for {qid}.",
                    "outcomes": ["Yes", "No"],
                    "open_date": iso_d(od),
                    "scheduled_end": iso_d(ed),
                },
                "scorer": {
                    "scored": scored,
                    "t_res": min(ts(cd), ts(ed)) if scored else None,
                    "closed_time": iso_d(cd) if cd is not None else None,
                    "resolution_answer": ans,
                    "answer_status": "ok" if ans else "open",
                    "resolved_in_window": scored,
                    "easy": None, "ta_bss_market": None,
                    "family": fam, "sched_end_bucket": "03",
                    "volume_usd": 200_000, "event_id": f"ev_{qid}",
                    "event_title": qid, "tags": [],
                },
            }) + "\n")
    for qid, pts in PRICES.items():
        (built / "prices" / f"{qid}.json").write_text(
            json.dumps({"grid_hours": 1, "points": pts}))
    return built


def make_tcfg(built: Path, **overrides) -> ForecastPortfolioConfig:
    params = dict(questions=[q[0] for q in QUESTIONS], data_dir=built)
    params.update(overrides)
    return ForecastPortfolioConfig(**params)


class StubNews:
    """Minimal stand-in for the bnpm NewsStore. n1 is published at t0,
    n2 on day 5 — so a day-0 forecast citing n2 is not-yet-published."""

    PUB = {"n1": 0.0, "n2": 5.0}

    def search(self, q, date_from, date_to, now, top_k, offset):
        return [{"news_id": "n1", "title": "stub", "domain": "x",
                 "published": "2026-03-01"}]

    def get_article(self, news_id, now):
        if news_id not in self.PUB:
            return None
        return {"news_id": news_id, "title": "stub", "domain": "x",
                "published": iso_d(self.PUB[news_id]), "description": "",
                "text": "body"}

    def published_ts(self, news_id):
        d = self.PUB.get(news_id)
        return None if d is None else ts(d)


@pytest.fixture
def built(tmp_path: Path) -> Path:
    return write_world(tmp_path)


@pytest.fixture
def tcfg(built: Path) -> ForecastPortfolioConfig:
    return make_tcfg(built)


@pytest.fixture
def news() -> StubNews:
    return StubNews()


@pytest.fixture
def scorer(built: Path, tcfg: ForecastPortfolioConfig,
           news: StubNews) -> Scorer:
    return Scorer(tcfg, built, load_questions(built, tcfg, T0, SIM_END),
                  news=news)


@pytest.fixture
def task(built: Path, tcfg: ForecastPortfolioConfig, news: StubNews,
         scorer: Scorer) -> ForecastPortfolioTask:
    return ForecastPortfolioTask(tcfg, built, news, scorer)
