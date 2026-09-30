"""Task-owned agent-side code: scan handler, record
semantics, and the sig-self stack against the sparse minute price payload.

Modules are loaded the way a provisioned workspace lays them out — flat,
with agent/ and agent/sig_self/ on the path — under fresh module names so
same-named modules elsewhere can't leak in either direction."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT = REPO_ROOT / "tasks" / "breakout_news_pm" / "agent"
UTC = timezone.utc

_MODS = ("breakout_stats", "feedback_fn", "pull_feedback", "records",
         "question_keywords", "scan")


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scaffolds"))
    monkeypatch.syspath_prepend(str(AGENT))
    monkeypatch.syspath_prepend(str(AGENT / "sig_self"))
    for m in _MODS:
        sys.modules.pop(m, None)
    out = {m: importlib.import_module(m) for m in _MODS}
    yield out
    for m in _MODS:
        sys.modules.pop(m, None)


def t(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2021, 6, day, hour, minute, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sparse_payload(jump_day: int | None) -> dict:
    """Sparse minute change-series: flat 0.5 from May 1, one +0.2 change at
    14:23 on jump_day (June), visible through June 10."""
    changes = []
    if jump_day is not None:
        changes = [(_iso(t(jump_day, 14).replace(minute=23)), 0.7)]
    return {"market_id": "m1", "grid_minutes": 1, "sparse": "changes",
            "level_at_start": 0.5,
            "changes": {"time": [c[0] for c in changes],
                        "p": [c[1] for c in changes]},
            "visible_until": _iso(t(10))}


MAY1 = datetime(2021, 5, 1, tzinfo=UTC)


# -- breakout_stats: sparse-payload adapter + detector --------------------------------


def test_daily_closes_densifies_sparse_series(mods):
    bs = mods["breakout_stats"]
    closes = bs.daily_closes(sparse_payload(jump_day=5), MAY1)
    by_day = {d: (p, ct) for d, p, ct in closes}
    # every calendar day of the visible span is present (quiet days too)
    assert len(closes) == 41  # May 1 .. June 10
    assert by_day["2021-05-20"][0] == 0.5
    assert by_day["2021-06-04"][0] == 0.5
    assert by_day["2021-06-05"] == (0.7, "2021-06-05T14:23:00Z")
    assert by_day["2021-06-06"][0] == 0.7  # forward-filled
    assert closes[0][0] == "2021-05-01" and closes[-1][0] == "2021-06-10"


def test_detector_finds_jump_day_with_direction(mods):
    bs = mods["breakout_stats"]
    closes = bs.daily_closes(sparse_payload(jump_day=5), MAY1)
    bps = bs.detect_breakpoints(closes, t(1), t(10))
    assert [b["day"] for b in bps] == ["2021-06-05"]
    assert bps[0]["direction"] == "up"
    assert bps[0]["t_move"] == "2021-06-05T14:23:00Z"  # minute-informed
    assert bps[0]["z"] >= bs.Z_THRESHOLD
    quiet = bs.daily_closes(sparse_payload(jump_day=None), MAY1)
    assert bs.detect_breakpoints(quiet, t(1), t(10)) == []


# -- feedback_fn ----------------------------------------------------------------------


class FakeEnv:
    """Speaks the uniform call() surface of runtime.env_client.Env."""

    def __init__(self, now: datetime, payload: dict, markets=()):
        self._now = now
        self._payload = payload
        self._markets = list(markets)
        self.price_calls = 0
        self.llm_calls = 0
        self.llm_reply: dict | None = {}

    def now(self) -> datetime:
        return self._now

    def call(self, name, **args):
        if name == "get_markets":
            return self._markets
        assert name == "get_prices"
        self.price_calls += 1
        return self._payload


def stub_llm(monkeypatch, mods, env):
    def fake_chat_json(_env, messages, **kw):
        env.llm_calls += 1
        return env.llm_reply
    monkeypatch.setattr(mods["feedback_fn"].llm_client, "chat_json",
                        fake_chat_json)


def test_quiet_period_all_negative_zero_llm(mods, monkeypatch):
    fb = mods["feedback_fn"]
    env = FakeEnv(now=t(12), payload=sparse_payload(jump_day=None))
    stub_llm(monkeypatch, mods, env)
    out = fb.hindsight_verdicts(
        env, "m1", t(1), t(10),
        [{"news_id": "n1", "title": "x", "published": _iso(t(4))}])
    assert out["breakpoints"] == []
    assert out["verdicts"] == [{"news_id": "n1", "verdict": "no_breakout"}]
    assert env.llm_calls == 0 and env.price_calls == 1


def test_attributed_verdict_carries_direction(mods, monkeypatch):
    fb = mods["feedback_fn"]
    env = FakeEnv(now=t(12), payload=sparse_payload(jump_day=5))
    env.llm_reply = {"n-hit": 0.9, "n-nope": 0.1}
    stub_llm(monkeypatch, mods, env)
    out = fb.hindsight_verdicts(
        env, "m1", t(1), t(10),
        [{"news_id": "n-hit", "title": "driver", "published": _iso(t(4, 18))},
         {"news_id": "n-nope", "title": "noise", "published": _iso(t(4, 19))}],
        question="Will X happen?")
    by_id = {v["news_id"]: v for v in out["verdicts"]}
    assert by_id["n-hit"]["verdict"] == "attributed"
    assert by_id["n-hit"]["direction"] == "up"
    assert by_id["n-hit"]["t_move"] == "2021-06-05T14:23:00Z"
    assert by_id["n-nope"]["verdict"] == "no_breakout"
    assert env.llm_calls == 1


def test_open_period_refused_before_spending(mods, monkeypatch):
    fb = mods["feedback_fn"]
    env = FakeEnv(now=t(10, 12), payload=sparse_payload(jump_day=5))
    stub_llm(monkeypatch, mods, env)
    with pytest.raises(ValueError, match="not closed"):
        fb.hindsight_verdicts(env, "m1", t(1), t(10), [])
    assert env.price_calls == 0 and env.llm_calls == 0


# -- pull_feedback: global candidate sweep --------------------------------------------


def test_compile_sweeps_markets_and_verdicts_once(mods, monkeypatch):
    pf = mods["pull_feedback"]
    env = FakeEnv(now=t(12), payload=sparse_payload(jump_day=9),
                  markets=[{"market_id": "m1", "question": "Will X?"},
                           {"market_id": "m2", "question": "Will Y?"}])
    env.llm_reply = {"n-hit": 0.8}
    stub_llm(monkeypatch, mods, env)
    state = {"registered": {
        "n-hit": {"title": "driver", "published": _iso(t(9, 1))},
        "n-other": {"title": "noise", "published": _iso(t(9, 2))},
        "n-out": {"title": "outside window", "published": _iso(t(1))},
    }}
    out = pf.compile(env, state)
    # attributed on both markets (same fake payload); one no_breakout for
    # the unattributed candidate; the out-of-window one stays registered
    hits = [r for r in out if r["verdict"] == "attributed"]
    assert {r["market_id"] for r in hits} == {"m1", "m2"}
    assert all(r["news_id"] == "n-hit" and r["direction"] == "up"
               for r in hits)
    misses = [r for r in out if r["verdict"] == "no_breakout"]
    assert [r["news_id"] for r in misses] == ["n-other"]
    assert "market_id" not in misses[0]
    assert set(state["registered"]) == {"n-out"}  # swept candidates popped
    assert state["last_hindsight"] == _iso(t(12))
    # throttle: immediate second compile is a no-op
    assert pf.compile(env, state) == []


def test_compile_waits_for_closed_period_without_stamping(mods, monkeypatch):
    pf = mods["pull_feedback"]
    # now is too close: until = now - lag, until + grace > now always when
    # grace > lag is false here (lag 24, grace 24) -> boundary: equality
    # passes; make now land mid-grace via a fresher registered item only
    env = FakeEnv(now=t(2, 1), payload=sparse_payload(jump_day=None),
                  markets=[{"market_id": "m1", "question": "Q"}])
    stub_llm(monkeypatch, mods, env)
    state = {"registered": {"n1": {"title": "x", "published": _iso(t(1))}}}
    monkeypatch.setattr(pf, "HINDSIGHT_LAG_HOURS", 1)  # until+grace > now
    assert pf.compile(env, state) == []
    assert "last_hindsight" not in state  # retries next firing
    assert "n1" in state["registered"]


# -- records --------------------------------------------------------------------------


def test_action_for_links_and_pops_once(mods):
    rec = mods["records"]
    state = {"actions": {"n1": {"did": "alerted", "direction": "up"}}}
    assert rec.action_for({"news_id": "n1"}, state)["direction"] == "up"
    assert rec.action_for({"news_id": "n1"}, state) is None
    assert rec.action_for({"kind": "breakpoint"}, state) is None


def test_stratum_of_new_vocabulary(mods):
    rec = mods["records"]
    assert rec.stratum_of({"outcome": {"status": "wrong_direction"}}) == \
        "wrong_direction"
    assert rec.stratum_of({"outcome": {"verdict": "attributed"}}) == \
        "attributed"
    assert rec.stratum_of({"outcome": {"kind": "breakpoint"}}) == "breakpoint"
    assert rec.stratum_of({}) == "unknown"


def test_register_candidates_global_dedup(mods):
    rec = mods["records"]
    state: dict = {}
    result = {"results": [
        {"news_id": "n1", "title": "T1", "published": _iso(t(1)),
         "snippet": "s", "domain": "d"},
        {"news_id": "n2", "title": "T2", "published": _iso(t(2)),
         "snippet": "s", "domain": "d"}]}
    rec.register_candidates(state, result)
    rec.register_candidates(state, {"results": [
        {"news_id": "n1", "title": "changed", "published": _iso(t(3))}]})
    assert set(state["registered"]) == {"n1", "n2"}
    assert state["registered"]["n1"]["title"] == "T1"  # first sighting wins


# -- scan -----------------------------------------------------------------------------


class ScanEnv:
    def __init__(self, now: datetime, results: list[dict],
                 reject: bool = False):
        self._now = now
        self._results = results
        self._reject = reject
        self.searches: list[dict] = []
        self.notified: list[dict] = []

    def now(self) -> datetime:
        return self._now

    def call(self, name, **args):
        if name == "get_markets":
            return [{"market_id": "m1", "question": "Will the Fed cut rates?",
                     "start": "2021-06-01T00:00:00Z",
                     "end": "2021-06-30T00:00:00Z"}]
        if name == "search_news":
            self.searches.append(args)
            return {"query": args["q"], "offset": 0, "results": self._results}
        assert name == "notify"
        if self._reject:
            raise RuntimeError("400 not-yet-published")
        self.notified.append(args)
        return {"status": "accepted"}


def hit(nid: str, when: datetime) -> dict:
    return {"news_id": nid, "title": f"story {nid}", "domain": "x.com",
            "published": _iso(when), "snippet": "words"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "decide.md").write_text(
        (AGENT / "prompts" / "decide.md").read_text())
    (tmp_path / "INSTRUCTION.md").write_text("(instruction)")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def scan_llm(monkeypatch, mods, reply):
    calls = []

    def fake_chat_json(_env, messages, **kw):
        calls.append(messages)
        return reply
    monkeypatch.setattr(mods["scan"].llm_client, "chat_json", fake_chat_json)
    return calls


def test_scan_notifies_with_direction(mods, workspace, monkeypatch):
    scan = mods["scan"]
    env = ScanEnv(t(2, 6), [hit("n1", t(2, 5, 30)), hit("n2", t(2, 5, 30))])
    calls = scan_llm(monkeypatch, mods, {"alerts": [
        {"news_id": "n1", "direction": "down"}]})
    state: dict = {}
    scan.scan(env, state, block="")
    assert env.searches[0]["date_from"] == _iso(t(2, 5))  # 1 h cursor default
    assert env.notified == [{"market_id": "m1", "news_id": "n1",
                             "direction": "down"}]
    assert state["actions"]["n1"]["direction"] == "down"
    assert set(state["registered"]) == {"n1", "n2"}  # global candidates
    assert state["scan_cursor"] == _iso(t(2, 6))
    assert len(calls) == 1 and "(instruction)" in calls[0][0]["content"]


def test_scan_drops_malformed_alerts(mods, workspace, monkeypatch):
    scan = mods["scan"]
    env = ScanEnv(t(2, 6), [hit("n1", t(2, 5, 30)), hit("n2", t(2, 5, 30)),
                            hit("n3", t(2, 5, 30))])
    scan_llm(monkeypatch, mods, {"alerts": [
        "n1",                                      # bare id: no direction
        {"news_id": "n2", "direction": "sideways"},
        {"news_id": "n3", "direction": "up"}]})
    scan.scan(env, {}, block="")
    assert [n["news_id"] for n in env.notified] == ["n3"]
    notes = [json.loads(l) for l in
             open("logs/trace.jsonl", encoding="utf-8")]
    assert sum(1 for n in notes
               if n.get("what") == "malformed_alert") == 2


def test_scan_idempotent_and_rejection_logged(mods, workspace, monkeypatch):
    scan = mods["scan"]
    reply = {"alerts": [{"news_id": "n1", "direction": "up"}]}
    # first scan alerts n1
    env = ScanEnv(t(2, 6), [hit("n1", t(2, 5, 30))])
    scan_llm(monkeypatch, mods, reply)
    state: dict = {}
    scan.scan(env, state, block="")
    assert len(env.notified) == 1
    # second scan re-surfaces the same item: never re-sent
    env2 = ScanEnv(t(2, 7), [hit("n1", t(2, 6, 30))])
    scan_llm(monkeypatch, mods, reply)
    scan.scan(env2, state, block="")
    assert env2.notified == []
    # a rejected notify is logged, not raised, and not marked alerted
    env3 = ScanEnv(t(2, 8), [hit("n9", t(2, 7, 30))], reject=True)
    scan_llm(monkeypatch, mods, {"alerts": [
        {"news_id": "n9", "direction": "up"}]})
    scan.scan(env3, state, block="")
    assert "n9" not in state["alerted"]["m1"]
    notes = [json.loads(l) for l in
             open("logs/trace.jsonl", encoding="utf-8")]
    assert any(n.get("what") == "notify_rejected" for n in notes)
