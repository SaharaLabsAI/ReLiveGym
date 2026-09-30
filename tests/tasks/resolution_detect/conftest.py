"""Synthetic resolution_detect world with exact-math determination times.

Sim window for tests: 2026-03-01 .. 2026-03-11 (10 days). Questions
(times in whole days from t0 so credit fractions are exact):

  q_win    opens day -2, sched_end day 60, t_det day 4, t_res/closes day 6
           -> Yes  (winnable, gap 2 days)
  q_lag    opens day 0, sched_end day 60, t_det day 3, t_res day 6,
           closes day 6.5 -> No  (winnable; official close lags t_res —
           credit window ends at t_res, settlement at closed_time)
  q_mid    opens day 2, sched_end day 60, t_det day 7, t_res/closes day 9
           -> Yes  (winnable, gap 2 days, mid-window arrival)
  q_unwin  opens day -1, closes day 5 -> No  (scored, NO determination
           crossing: unwinnable — claims on it are fa_premature)
  q_open   opens day -1, never closes                    (quiet load)
  q_late   opens day 1, closes day 30 -> Yes  (resolves after sim_end:
           run-unscored load, answer never public in-run)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tasks.resolution_detect.task import (
    ResolutionDetectConfig, ResolutionDetectTask, Scorer, load_questions,
)

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
SIM_END = T0 + timedelta(days=10)


def t(days: float) -> datetime:
    return T0 + timedelta(days=days)


def ts(days: float) -> float:
    return t(days).timestamp()


def iso_d(days: float) -> str:
    return t(days).strftime("%Y-%m-%dT%H:%M:%SZ")


# qid, open_d, sched_end_d, closed_d, t_res_d, t_det_d, answer, scored, family
QUESTIONS = [
    ("q_win", -2, 60, 6, 6, 4, "Yes", True, "geopolitics"),
    ("q_lag", 0, 60, 6.5, 6, 3, "No", True, "econ"),
    ("q_mid", 2, 60, 9, 9, 7, "Yes", True, "politics"),
    ("q_unwin", -1, 60, 5, 5, None, "No", True, "crypto"),
    ("q_open", -1, 60, None, None, None, None, False, "politics"),
    ("q_late", 1, 60, 30, None, None, "Yes", False, "culture"),
]


def write_world(root: Path) -> Path:
    built = root / "built"
    built.mkdir(parents=True)
    with (built / "questions.jsonl").open("w") as f:
        for (qid, od, ed, cd, rd, dd, ans, scored, fam) in QUESTIONS:
            t_det = ts(dd) if dd is not None else None
            t_res = ts(rd) if rd is not None else None
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
                    "t_res": t_res,
                    "closed_time": iso_d(cd) if cd is not None else None,
                    "resolution_answer": ans,
                    "answer_status": "ok" if ans else "open",
                    "resolved_in_window": scored,
                    "t_det": t_det,
                    "gap_s": (t_res - t_det if t_det is not None else None),
                    "winnable": t_det is not None,
                    "trap": False,
                    "news_lead_flag": False,
                    "family": fam, "open_bucket": "pre03",
                    "volume_usd": 200_000, "event_id": f"ev_{qid}",
                    "event_title": qid, "tags": [],
                },
            }) + "\n")
    return built


def make_tcfg(built: Path, **overrides) -> ResolutionDetectConfig:
    params = dict(questions=[q[0] for q in QUESTIONS], data_dir=built)
    params.update(overrides)
    return ResolutionDetectConfig(**params)


class StubNews:
    """Minimal stand-in for the bnpm NewsStore."""

    def search(self, q, date_from, date_to, now, top_k, offset):
        return [{"news_id": "n1", "title": "stub", "domain": "x",
                 "published": "2026-03-01"}]

    def get_article(self, news_id, now):
        if news_id != "n1":
            return None
        return {"news_id": "n1", "title": "stub", "domain": "x",
                "published": "2026-03-01", "description": "", "text": "body"}

    def published_ts(self, news_id):
        return ts(0.5) if news_id == "n1" else None


@pytest.fixture
def built(tmp_path: Path) -> Path:
    return write_world(tmp_path)


@pytest.fixture
def tcfg(built: Path) -> ResolutionDetectConfig:
    return make_tcfg(built)


@pytest.fixture
def scorer(built: Path, tcfg: ResolutionDetectConfig) -> Scorer:
    return Scorer(tcfg, load_questions(built, tcfg, T0, SIM_END),
                  news=StubNews())


@pytest.fixture
def task(built: Path, tcfg: ResolutionDetectConfig,
         scorer: Scorer) -> ResolutionDetectTask:
    return ResolutionDetectTask(tcfg, built, StubNews(), scorer)
