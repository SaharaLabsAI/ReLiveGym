"""Wake-segment compaction in the react loop (agent.py context policy).

Regression: wake markers were only
appended at process start, but the v3 react loop waits *in-process* — the
whole run was one wake-segment, the compactor's keep-the-current-segment
guard made it a permanent no-op, and transcripts grew past 190k tokens on
real runs. A wake is a sim-time event: every clock-advancing wait-tool
return now appends a segment marker.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime import agent as agent_mod  # noqa: E402
from runtime.tokens import count_tokens  # noqa: E402


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class ClockEnv:
    def __init__(self):
        self.t = datetime(2026, 3, 1, tzinfo=timezone.utc)

    def now(self):
        return self.t


def make_agent(env, replies, filler_tokens=400, context_tokens=2_000):
    """An agent whose LLM is a scripted reply queue: each sleep advances
    the clock an hour and each look returns a large payload."""
    replies = iter(replies)

    def chat_json(_env, _messages):
        return next(replies)

    def sleep_fn(args):
        env.t += timedelta(hours=1)
        return {"now": env.t.isoformat(), "woke_for": "timeout"}

    tools = {
        "sleep": {"doc": "sleep", "fn": sleep_fn},
        "look": {"doc": "look", "fn": lambda args: {"data": "x " * filler_tokens}},
    }
    monkey_target = agent_mod.llm_client
    orig = monkey_target.chat_json
    monkey_target.chat_json = chat_json
    a = agent_mod.Agent(env, "actor", tools,
                        transcript="logs/react_transcript.jsonl",
                        wait_tool="sleep", context_tokens=context_tokens)
    a.wake({"id": None, "kind": "at"})
    return a, lambda: setattr(monkey_target, "chat_json", orig)


def run_turns(a, n):
    for _ in range(n):
        a.turn()


def transcript_tokens(a):
    return sum(count_tokens(m["content"]) for m in a.messages)


def wake_markers(a):
    return [m for m in a.messages
            if m["content"].startswith("(woke at ")]


def test_wait_return_appends_segment_marker():
    env = ClockEnv()
    a, undo = make_agent(env, [{"tool": "sleep", "args": {}}])
    try:
        a.turn()
    finally:
        undo()
    marks = wake_markers(a)
    assert len(marks) == 2  # process wake + the sleep return
    assert "from sleep" in marks[-1]["content"]
    assert env.t.isoformat() in marks[-1]["content"]


def test_transcript_compacts_across_wakes():
    # look+sleep cycles: each cycle adds ~400 tokens of payload against a
    # 2k budget — without per-wake markers this grows unboundedly; with them
    # it stays bounded.
    env = ClockEnv()
    script = [{"tool": "look", "args": {}}, {"tool": "sleep", "args": {}}] * 12
    a, undo = make_agent(env, script)
    try:
        run_turns(a, len(script))
    finally:
        undo()
    assert transcript_tokens(a) <= a.context_tokens
    # compaction dropped the oldest segments but kept the newest wake
    assert len(wake_markers(a)) >= 1
    assert env.t.isoformat() in wake_markers(a)[-1]["content"]
    # the on-disk transcript matches the in-memory one (rewritten on drop)
    lines = [json.loads(x) for x in
             Path("logs/react_transcript.jsonl").read_text().splitlines()]
    assert lines == a.messages


def test_compaction_never_drops_current_segment():
    # one enormous payload inside the current (only) segment: over budget
    # but nothing older to drop — the guard keeps the segment whole
    env = ClockEnv()
    a, undo = make_agent(env, [{"tool": "look", "args": {}}],
                         filler_tokens=5_000)
    try:
        a.turn()
    finally:
        undo()
    assert transcript_tokens(a) > a.context_tokens
    assert any("x x" in m["content"] for m in a.messages)


# -- daily cost brief render -----------------------

COSTS = {"day": 2, "of": 5, "budget_usd": 10.0, "spent_usd": 1.0,
         "remaining_usd": 9.0,
         "llm": {"budget_usd": 4.0, "spent_usd": 1.0, "remaining_usd": 3.0},
         "spend_by_type": {"llm": 1.0}}


def test_wake_and_wait_markers_carry_the_brief_and_stay_segment_markers():
    env = ClockEnv()
    a, undo = make_agent(env, [{"tool": "sleep", "args": {}}])
    try:
        a.wake({"id": "act", "kind": "cron", "costs": COSTS})
        a.tools["sleep"]["fn"] = lambda args: {
            "now": (env.t + timedelta(hours=1)).isoformat(),
            "woke_for": "sleep", "costs": COSTS}
        a.turn()
    finally:
        undo()
    marks = wake_markers(a)
    assert len(marks) == 3  # make_agent's wake, the briefed wake, the sleep
    for m in marks[1:]:
        assert m["content"].startswith("(woke at ")
        assert "\nDay 2 of 5. Budget: $1.00 spent of $10.00 ($9.00 left). " \
               "LLM budget: $1.00 spent of $4.00 ($3.00 left). " \
               "By type: llm $1.00." in m["content"]
    assert "Day" not in marks[0]["content"]  # no costs -> no line
