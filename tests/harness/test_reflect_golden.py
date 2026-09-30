"""Golden render pin: moving code between scaffolds/runtime and the
task's records.py must not change one byte of any rendered
reflection prompt — compiled workspaces are the experiment substrate.

The fixture exercises every moved path: id enrichment (news_title /
market_question / gold-article seen_by_you), the per-market ledger,
the digest (gold story, lead, unwinnable, sig-self verdict), the
interval split, market-description truncation, and the standing-wait
billing note in the cost report.

Regenerate (ONLY for an intentional render change):
    REGEN_GOLDEN=1 pytest tests/harness/test_reflect_golden.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime import memory, reflect  # noqa: E402
from runtime.env_client import EnvError  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "bnpm_reflect_render_v1.txt"
TASK_AGENT = REPO_ROOT / "tasks" / "breakout_news_pm" / "agent"


def _load_bnpm_records():
    spec = importlib.util.spec_from_file_location(
        "bnpm_records_golden", TASK_AGENT / "records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BNPM_RECORDS = _load_bnpm_records()


class StubEnv:
    def __init__(self, responses: dict):
        self.responses = responses

    def call(self, name: str, **kwargs):
        if name not in self.responses:
            raise EnvError(f"unknown tool {name!r}")
        return self.responses[name]

    def now(self):
        return datetime(2026, 3, 8, 0, 30, tzinfo=timezone.utc)


LONG_DESC = ("Resolves YES if the Federal Reserve makes no cut to the "
             "target rate at any 2026 meeting. " * 12)  # > 600 chars

MARKETS = [
    {"market_id": "616902",
     "question": "Will no Fed rate cuts happen in 2026?",
     "description": LONG_DESC},
    {"market_id": "582717",
     "question": "Russia x Ukraine ceasefire before 2027?",
     "description": ""},
]

RECORDS = [
    # own alert, unlinked (is_own via outcome.kind)
    {"t": "2026-03-03T00:30:00Z", "src": "oracle",
     "outcome": {"kind": "alert", "market_id": "616902",
                 "news_id": "abc123", "direction": "up",
                 "status": "false_alarm", "penalty": 5.0},
     "action": None, "note": None},
    # own breakpoint outcome, linked action
    {"t": "2026-03-04T00:30:00Z", "src": "oracle",
     "outcome": {"kind": "breakpoint", "market_id": "616902",
                 "status": "covered_news", "penalty": 12.5,
                 "news_id": "abc123",
                 "t_move_start": "2026-03-03T13:00:00Z"},
     "action": {"did": "alerted", "at": "2026-03-03T02:00:00Z",
                "market_id": "616902", "direction": "up",
                "title": "Fed holds rates"},
     "note": None},
    # missed breakpoint with two gold groups (confidence pick, lead, seen)
    {"t": "2026-03-05T00:30:00Z", "src": "oracle",
     "outcome": {"kind": "breakpoint", "market_id": "616902",
                 "status": "miss", "penalty": 50.0, "winnable": True,
                 "direction": "down",
                 "t_move_start": "2026-03-04T13:41:24Z",
                 "gold_groups": [
                     {"story": "Minor echo coverage.", "confidence": 0.62,
                      "articles": [{"news_id": "unseen1",
                                    "published_at": "2026-03-04T13:00:00Z"}]},
                     {"story": "Jobs report shocked markets into pricing "
                               "an emergency cut.", "confidence": 0.95,
                      "articles": [{"news_id": "abc123",
                                    "published_at": "2026-03-04T08:33:16Z"}]},
                 ]},
     "action": None, "note": None},
    # -- interval boundary: state["last_reflection"] sits here --
    # unwinnable miss
    {"t": "2026-03-06T00:30:00Z", "src": "oracle",
     "outcome": {"kind": "breakpoint", "market_id": "582717",
                 "status": "miss", "penalty": 50.0,
                 "direction": "up", "t_move_start": "2026-03-05T09:00:00Z",
                 "no_attributable_news": True},
     "action": None, "note": None},
    # sig-self hindsight verdict
    {"t": "2026-03-07T00:30:00Z", "src": "self",
     "outcome": {"verdict": "no_breakout", "market_id": "582717",
                 "news_id": "def456"},
     "action": None, "note": None},
    # own stale alert (t seeds the exemplar rng — keep it last)
    {"t": "2026-03-07T12:00:00Z", "src": "oracle",
     "outcome": {"kind": "alert", "market_id": "582717",
                 "news_id": "def456", "direction": "up",
                 "status": "stale", "penalty": 1.0},
     "action": None, "note": None},
]


def render_prompt(tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    Path("prompts").mkdir()
    shutil.copyfile(TASK_AGENT / "prompts" / "reflect.md",
                    "prompts/reflect.md")
    Path("INSTRUCTION.md").write_text(
        "# Task\nAlert on the news driving abrupt market moves; "
        "score = penalties plus spend, lower is better.\n",
        encoding="utf-8")
    for r in RECORDS:
        memory.append_record(**r)
    env = StubEnv({
        "get_markets": MARKETS,
        "get_costs": {"spend_by_type": {"news_search": 2.4, "llm": 6.05,
                                        "news_article": 0.3},
                      "spend_total": 8.75},
        "get_crontab": [{"id": "learn", "cron_expr": "30 0 * * *"}],
    })
    state = {
        "last_reflection": "2026-03-05T00:30:00Z",
        "reflection_count": 2,
        "last_costs": {"news_search": 1.0, "llm": 2.5},
        "last_wait": {"until": "2026-03-08T12:00:00Z"},
        "registered": {"abc123": {"title": "Fed holds rates",
                                  "published": "2026-03-04T08:33:16Z"},
                       "def456": {"title": "Ceasefire talks stall",
                                  "published": "2026-03-05T08:00:00Z"}},
    }
    captured = {}
    monkeypatch.setattr(
        reflect.llm_client, "chat",
        lambda e, m: captured.setdefault("p", m[0]["content"]) and "S")
    reflect.reflect(env, state, BNPM_RECORDS)
    return captured["p"]


def test_bnpm_reflection_render_is_byte_stable(tmp_path, monkeypatch):
    prompt = render_prompt(tmp_path, monkeypatch)
    if os.environ.get("REGEN_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(prompt, encoding="utf-8")
        pytest.skip(f"golden regenerated: {GOLDEN}")
    assert GOLDEN.exists(), "golden missing — run with REGEN_GOLDEN=1"
    assert prompt == GOLDEN.read_text(encoding="utf-8")
