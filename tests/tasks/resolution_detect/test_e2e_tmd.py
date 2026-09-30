"""End-to-end smoke of TM-D on
resolution_detect with a mocked LLM upstream, both shapes:

- task:cron_react under tm=D: the agent creates a one-time schedule with
  a note on its first waking; the env fires it through the ordinary
  trigger loop, the note arrives in the wake marker, the base daily cron
  keeps firing, and the schedule is gone afterwards.
- task:per_question_cron under tm=D: each question agent's schedule CRUD
  is pinned to its own question — a schedule created by one agent fires
  that agent only, and another agent's delete is refused by the wrapper.

Window 2026-03-01..03-04 with two never-claimed questions (the scripted
agents never claim), so every firing is observable.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import harness.llm_proxy as llm_proxy

REPO = Path(__file__).resolve().parents[3]
BUILT = REPO / "tasks" / "resolution_detect" / "data" / "built"
INDEX = REPO / "tasks" / "breakout_news_pm" / "news" / "tantivy_index_v3"

pytestmark = pytest.mark.skipif(
    not (BUILT / "questions.jsonl").exists() or not INDEX.exists(),
    reason="built world or news index absent")


def _resp(text: str, model: str) -> tuple[dict, float]:
    return ({"choices": [{"message": {"content": text}}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
            0.0)


def _transcript(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _wake_markers(path: Path) -> list[str]:
    return [m["content"] for m in _transcript(path)
            if m["role"] == "user" and m["content"].startswith("(woke at ")]


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in
            (run_dir / "ledger.jsonl").read_text().splitlines()]


def _cfg(run_id: str, scaffold: str):
    from harness.config import RunConfig

    return RunConfig(
        run_id=run_id,
        task={"name": "resolution_detect",
              "questions": ["891191", "665472"]},
        sim_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        sim_end=datetime(2026, 3, 4, tzinfo=timezone.utc),
        budget_usd=5.0,
        cell={"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"},
        agent={"scaffold": scaffold, "model": "mock-luna"},
        watchdog_seconds=60.0,
    )


def test_cron_react_tmd_agent_schedule_fires_with_note(tmp_path, monkeypatch):
    from harness.run import run_experiment

    async def fake_upstream(path, body):
        msgs = body["messages"]
        woke = [m for m in msgs[1:] if m["role"] == "user"
                and m["content"].startswith("(woke at ")]
        created = any('"tool":"create_schedule"' in m["content"]
                      or '"tool": "create_schedule"' in m["content"]
                      for m in msgs[1:])
        if len(woke) == 1 and not created:
            reply = {"tool": "create_schedule",
                     "args": {"id": "recheck", "at": "2026-03-01T06:00:00Z",
                              "note": "recheck 891191 after the morning"},
                     "thought": "come back later"}
        else:
            reply = {"tool": "done", "args": {}, "thought": "nothing now"}
        return _resp(json.dumps(reply), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    results = asyncio.run(run_experiment(
        _cfg("rd-e2e-tmd-cronreact", "task:cron_react"),
        repo_root=REPO, run_dir=tmp_path / "run"))
    assert "failed-degenerate" not in results["flags"]

    ws = tmp_path / "run" / "workspace"
    markers = _wake_markers(ws / "logs" / "transcript_agent.jsonl")
    # bootstrap, the agent's own 06:00 schedule, then the daily base cron
    assert "trigger __bootstrap__/at" in markers[0]
    assert "trigger recheck/at" in markers[1]
    assert "Your note for this schedule: recheck 891191 after the morning" \
        in markers[1]
    assert all("trigger act/cron" in m for m in markers[2:])
    assert len(markers) == 1 + 1 + 2  # 03-02 and 03-03 midnight base fires
    # the instruction the agent read names the schedule API
    text = (ws / "instructions" / "agent.md").read_text()
    assert "list_schedules / create_schedule" in text
    assert "you may change or remove it" in text

    ev = _events(tmp_path / "run")
    fired = [(e["id"], e.get("owner")) for e in ev if e["type"] == "trigger"]
    assert fired == [("__bootstrap__", None), ("recheck", "agent"),
                     ("act", "agent"), ("act", "agent")]  # default = agent row
    assert [e["id"] for e in ev if e["type"] == "schedule_seed"] == ["act"]
    created = [e for e in ev if e["type"] == "schedule_create"]
    assert len(created) == 1 and created[0]["cost"] == 0.0
    assert created[0]["note"] == "recheck 891191 after the morning"
    # one-time: gone after firing
    sched = json.loads((tmp_path / "run" / "schedule.json").read_text())
    assert [r["id"] for r in sched["agent"]] == ["act"]
    assert sched["crontab"] == [] and sched["seeded"] == ["act"]


def test_per_question_cron_tmd_schedules_are_pinned_to_their_question(
        tmp_path, monkeypatch):
    from harness.run import run_experiment

    async def fake_upstream(path, body):
        msgs = body["messages"]
        sys_prompt = msgs[0]["content"]
        mine = "question_id: 891191" in sys_prompt
        woke = [m for m in msgs[1:] if m["role"] == "user"
                and m["content"].startswith("(woke at ")]
        acted = any('create_schedule' in m["content"]
                    or 'delete_schedule' in m["content"]
                    for m in msgs[1:] if m["role"] == "assistant")
        # roster order sweeps 665472 first, so the cross-delete waits for
        # the second waking (the schedule then exists and is recurring)
        if mine and len(woke) == 1 and not acted:
            reply = {"tool": "create_schedule",
                     "args": {"id": "noon-look", "cron_expr": "0 12 * * *",
                              "note": "noon look"},
                     "thought": "later"}
        elif not mine and len(woke) == 2 and not acted:
            reply = {"tool": "delete_schedule",
                     "args": {"id": "noon-look"}, "thought": "not mine"}
        else:
            reply = {"tool": "done", "args": {}, "thought": "nothing now"}
        return _resp(json.dumps(reply), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    results = asyncio.run(run_experiment(
        _cfg("rd-e2e-tmd-perquestioncron", "task:per_question_cron"),
        repo_root=REPO, run_dir=tmp_path / "run"))
    assert "failed-degenerate" not in results["flags"]

    ws = tmp_path / "run" / "workspace"
    own = _wake_markers(ws / "logs" / "transcript_q-891191.jsonl")
    other = _wake_markers(ws / "logs" / "transcript_q-665472.jsonl")
    # the schedule woke its owner only: owner = bootstrap + 3 noons + 2
    # daily, the other agent = bootstrap + 2 daily
    assert sum("trigger noon-look/cron" in m for m in own) == 3
    assert any("Your note for this schedule: noon look" in m for m in own)
    assert not any("noon-look" in m for m in other)
    assert len(own) == 6 and len(other) == 3
    # the cross-delete was refused by the wrapper, never reached the env
    assert "belongs to another agent" in \
        (ws / "logs" / "transcript_q-665472.jsonl").read_text()
    ev = _events(tmp_path / "run")
    assert [e["type"] for e in ev if e["type"].startswith("schedule_")] == \
        ["schedule_create"]
    created = [e for e in ev if e["type"] == "schedule_create"][0]
    assert created["target"] == "q-891191"
    fired = [(e["id"], e.get("target")) for e in ev
             if e["type"] == "trigger" and e.get("owner") == "agent"]
    assert fired == [("noon-look", "q-891191")] * 3
