"""Reddit learning cells:
records.py semantics, the reflection template's slot set, the
constructor mounts for tmB-tlrndaily-sigoracle-algskills, and one
mocked e2e where the actor recommends, the learn cron drains the oracle
into records, and reflection writes the skill block."""

from __future__ import annotations

import pytest

import asyncio
import importlib.util
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import harness.llm_proxy as llm_proxy
from harness.run import run_experiment
from scaffolds.compose import render_main, workspace_files
from tests.tasks.reddit_ai_popularity.conftest import make_reddit_config, t

AGENT_DIR = (Path(__file__).resolve().parents[3] / "tasks"
             / "reddit_ai_popularity" / "agent")

LEARNING_CELL = dict(tlrn="daily", sig="oracle", alg="skills")


def _records_mod():
    spec = importlib.util.spec_from_file_location(
        "reddit_records", AGENT_DIR / "records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTCOME = {"kind": "recommendation", "t_settled": "2026-03-03T06:00:00Z",
           "root_id": "big", "title": "Post big", "subreddit": "OpenAI",
           "recommended_at": "2026-03-02T06:30:00Z",
           "posted_at": "2026-03-02T06:00:00Z", "delay_hours": 0.5,
           "descendants": 256, "pop": 9.0, "weight": 0.9792,
           "status": "tail"}


# -- records.py semantics --------------------------------------------------------------


POST_OUTCOME = {"kind": "post", "t_settled": "2026-03-03T09:00:00Z",
                "root_id": "miss", "subreddit": "grok", "title": "Post miss",
                "posted_at": "2026-03-02T09:00:00Z", "descendants": 88,
                "pop": 7.46, "status": "tail",
                "growth": {"1h": 5, "6h": 40, "24h": 88}}


def test_records_semantics():
    r = _records_mod()
    rec = {"t": "2026-03-03T06:00:00Z", "src": "oracle", "outcome": OUTCOME,
           "action": None}
    assert r.action_for(OUTCOME, {}) is None  # rec outcomes ARE own actions
    assert r.is_own(rec)
    assert r.stratum_of(rec) == "tail"
    assert r.entity_of(rec) == "OpenAI"
    post = {**rec, "outcome": POST_OUTCOME}
    assert not r.is_own(post)  # missed-tail settlements are not own actions
    assert r.stratum_of(post) == "tail"
    assert r.entity_of(post) == "grok"
    line = r.digest_line(post)
    assert "r/grok" in line and "88 comments" in line
    assert "growth 1h:5 6h:40 24h:88" in line


def test_render_context_quota_slot():
    r = _records_mod()

    class FakeEnv:
        def call(self, name, **kw):
            assert name == "quota"
            return {"daily_cap": 10, "used_last_24h": 4, "remaining": 6}

    ctx = r.render_context(FakeEnv(), {})
    slot = ctx["extra_slots"]["quota"]
    assert "used 4 of 10" in slot and "6 free" in slot
    assert "24 h after" in slot  # the return mechanics, in words


def test_reflect_template_slots_all_resolvable():
    """Every ${...} slot in the template is either a runtime-provided
    slot or the task's own extra slot — nothing renders unresolved."""
    known = {"run_header", "instruction", "skills", "cum_counts",
             "entity_ledger", "own_history", "outcome_digest",
             "distribution", "own_actions", "examples", "costs",
             "quota"}
    text = (AGENT_DIR / "prompts" / "reflect.md").read_text()
    used = set(re.findall(r"\$\{(\w+)\}", text))
    assert used <= known
    assert "quota" in used  # the task-defined slot is actually rendered


# -- constructor mounts ----------------------------------------------------------------


def _cfg(built, tm, **overrides):
    overrides.setdefault("run_id", f"learn-{tm}")
    return make_reddit_config(
        built, cell=dict(tm=tm, **LEARNING_CELL),
        agent=dict(scaffold="react", model="mock-luna"), **overrides)


def test_learning_cell_mounts_records_and_reflect_prompt(built):
    files = workspace_files(_cfg(built, "B"))
    assert "records.py" in files and files["records.py"].is_file()
    assert "prompts/reflect.md" in files
    assert files["prompts/reflect.md"].is_file()


def test_ab_learning_programs_differ_only_in_wait_tool(built):
    a = render_main(_cfg(built, "A"))
    b = render_main(_cfg(built, "B"))
    assert "pull_oracle" in a and "reflect.reflect" in a
    assert a != b
    assert b.replace('"run_program"', '"sleep"') == a


def test_algmemory_cell_formatted_block_no_reflection(built):
    """alg=memory: outcomes drain into records and render straight into
    the actor's context block — no reflection LLM, no curation prompt."""
    cfg = make_reddit_config(
        built, run_id="learn-mem",
        cell=dict(tm="B", tlrn="daily", sig="oracle", alg="memory"),
        agent=dict(scaffold="react", model="mock-luna"))
    files = workspace_files(cfg)
    assert "records.py" in files  # outcome linking still task-owned
    assert "prompts/reflect.md" not in files  # no curation step
    main = render_main(cfg)
    assert "pull_oracle" in main
    assert "block_fn=memory.render_block" in main
    assert "reflect" not in main
    # the formatted render is bound with the task semantics (the legend
    # now rides in via memory.bind; memory_formatted_render_v1)
    assert "memory.bind(records, env, state)" in main
    assert "memory.LEGEND" not in main


def test_block_legend_is_factual_feed_contract():
    """The legend states the two things raw records can't say: what the
    kinds mean and that the stream is tail-filtered (base-rate trap)."""
    r = _records_mod()
    leg = r.BLOCK_LEGEND
    assert '"recommendation"' in leg and '"post"' in leg
    assert "did not recommend" in leg
    assert "do not reflect the tail base rate" in leg
    assert "growth" in leg


# -- mocked e2e ------------------------------------------------------------------------


def _resp(text: str, model: str) -> tuple[dict, float]:
    return ({"choices": [{"message": {"content": text}}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10},
             "model": model}, 0.0)


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tmb_learning_cell_end_to_end(built, tmp_path, monkeypatch):
    """tmB-tlrndaily-sigoracle-algskills on the synthetic world: the
    actor waits by program, recommends 'big' shortly after posting, the
    daily learn cron drains the settled outcome into records.jsonl, and
    reflection overwrites memory/skills.md — all without a sleep tool."""
    SKILL = "Rule: pace the rolling cap; require velocity confirmation."
    state = {"recommended": False}

    async def fake_upstream(path, body):
        prompt = body["messages"][-1]["content"]
        if "WRITE THE NEW SKILL BLOCK" in prompt:  # reflection call
            return _resp(SKILL, body["model"])
        if '"bytes"' in prompt:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "wait.py"},
                 "thought": "wake time lives in the code"}), body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        if not state["recommended"] and now >= t(2, 6, 30):
            state["recommended"] = True
            return _resp(json.dumps(
                {"tool": "recommend", "args": {"root_id": "big"},
                 "thought": "6:00 post is cascading"}), body["model"])
        until = (t(2, 6, 30) if now < t(2, 6, 30)
                 else now + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "write_file",
             "args": {"path": "wait.py",
                      "content": f'import envkit\nenvkit.wait("{until}")\n'},
             "thought": "park until the next decision point"}),
            body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    # tail bar lowered to 4 so the synthetic world has unrecommended
    # tails (mid: 8, self: 4) settling inside the window
    cfg = _cfg(built, "B", run_id="reddit-tmb-learn",
               sim_start=t(2), sim_end=t(5), task=dict(tail_min_desc=4))
    results = asyncio.run(run_experiment(
        cfg, repo_root=built.parent, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    manifest = json.loads(
        (tmp_path / "run" / "workspace_manifest.json").read_text())
    names = {tool["name"] for tool in manifest["tools"]}
    assert "sleep" not in names
    assert {"run_program", "get_feedback", "get_post_popularity"} <= names

    ws = tmp_path / "run" / "workspace"
    # the learn cron drained the feed into records: the own
    # recommendation plus the unrecommended tails, never nontail noise
    records = read_jsonl(ws / "memory" / "records.jsonl")
    by_kind: dict = {}
    for r in records:
        assert r["src"] == "oracle"
        by_kind.setdefault(r["outcome"]["kind"], []).append(r["outcome"])
    own = by_kind.get("recommendation", [])
    assert [o["root_id"] for o in own] == ["big"]
    assert own[0]["status"] == "tail" and "growth" in own[0]
    posts = {o["root_id"]: o for o in by_kind.get("post", [])}
    assert {"mid", "self"} <= set(posts)  # missed tails stream
    assert "dud" not in posts  # nontail non-recommended: silent
    assert all(o["status"] == "tail" and o["growth"]
               for o in posts.values())
    # reflection wrote the skill block through the task's reflect.md
    assert (ws / "memory" / "skills.md").read_text().strip().endswith(SKILL)
    # and the outcome scored: one settled tail recommendation
    assert results["performance"]["tail_hits"] == 1
    assert results["performance"]["primary"]["value"] > 0


@pytest.mark.slow
def test_tmc_cron_react_cell_end_to_end(built, tmp_path, monkeypatch):
    """tmC-tlrndaily-sigoracle-algskills via task:cron_react on the
    synthetic world: the hourly act cron fires the single stream
    agent, which recommends 'big' by its own choice inside a firing and
    ends every firing with done; the transcript persists across firings;
    the learn cron drains the settled outcome into records.jsonl and
    reflection writes memory/skills.md. The agent holds no wait,
    schedule, or feedback tools."""
    SKILL = "Rule: distilled from oracle outcomes."
    state = {"recommended": False, "system": ""}

    async def fake_upstream(path, body):
        prompt = body["messages"][-1]["content"]
        if "WRITE THE NEW SKILL BLOCK" in prompt:  # reflection call
            return _resp(SKILL, body["model"])
        if "Available tools:" in body["messages"][0]["content"]:
            state["system"] = body["messages"][0]["content"]
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r"woke at ([0-9TZ:\-\.+]+)", text)
        now = datetime.fromisoformat(times[-1])
        if not state["recommended"] and now >= t(2, 6, 30):
            state["recommended"] = True
            return _resp(json.dumps(
                {"tool": "recommend", "args": {"root_id": "big"},
                 "thought": "6:00 post is cascading"}), body["model"])
        return _resp(json.dumps({"tool": "done", "args": {}}),
                     body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_reddit_config(
        built, run_id="reddit-tmc-learn",
        cell=dict(tm="C", **LEARNING_CELL),
        agent=dict(scaffold="task:cron_react", model="mock-luna"),
        sim_start=t(2), sim_end=t(4, 1))
    results = asyncio.run(run_experiment(
        cfg, repo_root=built.parent, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    # the agent's registry (the system prompt's tool list) holds no
    # wait, schedule, or feedback tools — and ends firings with done
    assert "Available tools:" in state["system"]
    for gone in ("sleep", "wait_until", "set_crontab", "run_at",
                 "get_post_popularity", "get_feedback"):
        assert f"- {gone}(" not in state["system"]
    assert "- done(" in state["system"]

    ws = tmp_path / "run" / "workspace"
    # one persistent conversation, one wake marker per cron firing
    transcript = read_jsonl(ws / "logs" / "transcript_agent.jsonl")
    text = " ".join(m["content"] for m in transcript)
    assert text.count("(woke at") > 1  # persisted across firings
    assert "trigger act/cron" in text
    # the program owns the schedule: the act entry + the learn entry
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    assert [e for e in ledger if e["type"] == "crontab_put"]
    # the learn cron drained the own settled recommendation
    records = read_jsonl(ws / "memory" / "records.jsonl")
    own = [r["outcome"] for r in records
           if r["outcome"]["kind"] == "recommendation"]
    assert [o["root_id"] for o in own] == ["big"]
    assert own[0]["status"] == "tail"
    # reflection wrote the skill block through the task's reflect.md
    assert (ws / "memory" / "skills.md").read_text().strip().endswith(SKILL)
    # and the outcome scored: one settled tail recommendation
    assert results["performance"]["tail_hits"] == 1
    assert results["performance"]["primary"]["value"] > 0
