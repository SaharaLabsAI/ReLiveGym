"""Formatted alg=memory learned block: cumulative statistics over all records,
readable id-free lines, deterministic budget-driven truncation with the
own-share / two-tier / stratum-floor policy, raw fallback when unbound."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime import memory  # noqa: E402
from runtime.tokens import count_tokens  # noqa: E402


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    memory.RECORDS_MOD = None
    memory.ENV = None
    memory.STATE = None
    memory.LEGEND = ""


def _records_mod() -> types.ModuleType:
    """A minimal task records module exercising every hook."""
    m = types.ModuleType("records")
    m.BLOCK_LEGEND = "Each line is one settled record."
    m.ENTITY_NOUN = "market"
    m.STAT_FIELDS = [("credit", "all", ("n", "sum", "mean"))]
    m.stratum_of = lambda r: (r.get("outcome") or {}).get("status", "unknown")
    m.is_own = lambda r: (r.get("outcome") or {}).get("kind") == "alert"
    m.entity_of = lambda r: (r.get("outcome") or {}).get("market")
    m.entity_label = lambda k, ctx: f'{k} "question of {k}"'
    m.digest_line = lambda r: (f"{r['t'][:10]} | compact | "
                               f"{(r.get('outcome') or {}).get('status')}")
    m.outcome_line = lambda r: (
        f"{r['t'][:10]} | FULL | {(r.get('outcome') or {}).get('status')} | "
        + "story words " * 12)
    m.action_line = lambda r: (f"{r['t'][:10]} | OWN | "
                               f"{(r.get('outcome') or {}).get('status')}")
    return m


def _fill(n: int, status: str = "miss", kind: str = "breakpoint",
          market: str = "m1", credit: float | None = None,
          day0: int = 1) -> None:
    for k in range(n):
        out = {"kind": kind, "status": status, "market": market}
        if credit is not None:
            out["credit"] = credit
        memory.append_record(t=f"2021-06-{day0 + k // 24:02d}T"
                             f"{k % 24:02d}:00:00Z", src="oracle",
                             outcome=out)


def test_unbound_falls_back_to_raw():
    _fill(2)
    assert memory.RECORDS_MOD is None
    block = memory.render_block()
    assert block.startswith("## What you have learned so far (raw outcomes)")
    assert block == memory.render_block_raw()


def test_formatted_structure_and_stats():
    memory.bind(_records_mod(), env=None, state={})
    _fill(3, status="miss", market="m1", credit=0.5)
    _fill(2, status="covered_news", market="m2", credit=1.0, day0=3)
    memory.append_record(t="2021-06-05T00:00:00Z", src="oracle",
                         outcome={"kind": "alert", "status": "false_alarm",
                                  "market": "m1"})
    block = memory.render_block()
    assert block.splitlines()[0] == \
        "## What you have learned so far (settled outcomes)"
    assert "Each line is one settled record." in block  # legend via bind
    assert "### Cumulative counts (all 6 settled records" in block
    assert "own actions settled — false_alarm: 1" in block
    assert "covered_news: 2" in block and "miss: 3" in block
    assert "credit — n 5, sum 3.5, mean 0.7 (over records carrying the " \
        "field)" in block
    assert "### Cumulative outcomes by market" in block
    assert 'm1 "question of m1": false_alarm 1, miss 3' in block
    assert "### Settled outcomes (history)" in block
    assert "### Your actions and how they settled" in block
    assert "| OWN | false_alarm" in block


def test_hard_budget_bound_and_determinism():
    memory.bind(_records_mod(), env=None, state={})
    _fill(60, status="miss", credit=0.3)
    _fill(30, status="covered_news", day0=5)
    for budget in (120, 300, 800, 2000, 100_000):
        block = memory.render_block(budget_tokens=budget)
        assert count_tokens(block) <= budget, budget
        assert block == memory.render_block(budget_tokens=budget)


def test_two_tiers_and_elision_note():
    memory.bind(_records_mod(), env=None, state={})
    _fill(40, status="miss")
    block = memory.render_block(budget_tokens=500)
    lines = block.splitlines()
    fulls = [l for l in lines if "| FULL |" in l]
    compacts = [l for l in lines if "| compact |" in l]
    assert fulls and compacts  # both resolution tiers present
    # newest records get the full tier; the compact tier is strictly older
    hist = [l for l in lines if "| FULL |" in l or "| compact |" in l]
    assert hist == sorted(hist, key=lambda l: l[:10])
    first_full = min(l[:10] for l in fulls)
    assert all(l[:10] <= first_full for l in compacts)
    m = re.search(r"\(oldest (\d+) of (\d+) omitted; the counts above "
                  r"cover them\)", block)
    assert m and int(m.group(2)) == 40
    assert int(m.group(1)) == 40 - len(fulls) - len(compacts)


def test_own_actions_survive_pressure():
    memory.bind(_records_mod(), env=None, state={})
    _fill(80, status="miss")
    for k in range(5):
        memory.append_record(t=f"2021-06-01T0{k}:30:00Z", src="oracle",
                             outcome={"kind": "alert", "status": "stale",
                                      "market": "m1"})
    block = memory.render_block(budget_tokens=900)
    assert len([l for l in block.splitlines() if "| OWN |" in l]) == 5
    assert "omitted" in block  # while non-own records were elided


def test_stratum_floor_keeps_rare_stratum_visible():
    memory.bind(_records_mod(), env=None, state={})
    _fill(2, status="rare_status", day0=1)  # old and rare
    _fill(60, status="miss", day0=3)  # newer flood
    block = memory.render_block(budget_tokens=800)
    assert len([l for l in block.splitlines()
                if "rare_status" in l and ("FULL" in l or "compact" in l)]) \
        == 2


def test_notes_render_in_own_section_and_stay_out_of_counts():
    memory.bind(_records_mod(), env=None, state={})
    _fill(2, status="miss")
    memory.append_record(t="2021-06-02T00:00:00Z", src="note", outcome={},
                         note="remember  this")
    block = memory.render_block()
    assert "notes you saved — 1" in block
    assert "2021-06-02 00:00 | note | remember this" in block
    assert "unknown" not in block  # the note polluted no stratum
    own_at = block.index("### Your actions")
    assert block.index("| note |") > own_at


def test_entity_rollup_caps_rows():
    memory.bind(_records_mod(), env=None, state={})
    for k in range(15):
        _fill(2 if k < 13 else 1, status="miss", market=f"mk{k:02d}",
              day0=1 + k)
    block = memory.render_block()
    rows = [l for l in block.splitlines() if l.startswith("mk")]
    assert len(rows) == memory.ENTITY_ROWS
    assert "(the other 3 markets hold 4 records)" in block


def test_empty_memory_renders_empty():
    memory.bind(_records_mod(), env=None, state={})
    assert memory.render_block() == ""


def test_missing_line_hooks_fall_back_to_raw():
    m = _records_mod()
    del m.digest_line
    memory.bind(m, env=None, state={})
    _fill(1)
    assert memory.render_block().startswith(
        "## What you have learned so far (raw outcomes)")


# -- the real task records modules -----------------


def _load_task_records(task: str) -> types.ModuleType:
    import importlib.util

    path = REPO_ROOT / "tasks" / task / "agent" / "records.py"
    spec = importlib.util.spec_from_file_location(f"{task}_records_fmt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEX_ID = re.compile(r"\b[0-9a-f]{16,}\b")


def test_bnpm_render_resolves_ids():
    records = _load_task_records("breakout_news_pm")

    class FakeEnv:
        def call(self, tool, **kw):
            assert tool == "get_markets"
            return [{"market_id": "678777",
                     "question": "Will the US strike Iran again in March?",
                     "description": "Resolves YES if strikes occur."}]

    nid = "d1" * 32
    gold = "a4" * 32
    state = {"registered": {nid: {"title": "US strikes expand into Iraq",
                                  "published": "2026-03-01T13:00:00Z"}}}
    memory.bind(records, FakeEnv(), state)
    memory.append_record(
        t="2026-03-02T00:30:00Z", src="oracle",
        outcome={"kind": "alert", "t_settled": "2026-03-01T19:05:25Z",
                 "market_id": "678777", "news_id": nid, "direction": "up",
                 "at": "2026-03-01T00:00:00Z", "status": "covering_timing"},
        action={"did": "alerted", "at": "2026-03-01T00:00:00Z",
                "market_id": "678777", "direction": "up"})
    memory.append_record(
        t="2026-03-02T00:30:00Z", src="oracle",
        outcome={"kind": "breakpoint", "market_id": "678777",
                 "t_move_start": "2026-03-01T19:05:25Z", "direction": "up",
                 "status": "covered_timing", "credit": 0.7, "winnable": True,
                 "gold_groups": [{
                     "story": "Airstrikes expanded onto Iraqi soil.",
                     "confidence": 0.78,
                     "articles": [
                         {"news_id": nid,
                          "published_at": "2026-03-01T13:59:51Z"},
                         {"news_id": gold,
                          "published_at": "2026-03-01T18:00:31Z"}]}]})
    block = memory.render_block()
    assert not HEX_ID.search(block)  # no bare news ids anywhere
    assert '678777 "Will the US strike Iran again in March?"' in block
    assert "US strikes expand into Iraq" in block  # observed article title
    assert "Airstrikes expanded onto Iraqi soil." in block  # judge story
    assert "untitled article published 2026-03-01 18:00" in block
    assert "credit — n 1, sum 0.7, mean 0.7" in block
    assert "### Cumulative outcomes by market" in block


def test_reddit_render_is_id_free():
    records = _load_task_records("reddit_ai_popularity")
    memory.bind(records, env=None, state={})
    memory.append_record(
        t="2026-06-02T00:00:00Z", src="oracle",
        outcome={"kind": "recommendation", "root_id": "t3_abc123xy",
                 "title": "New local model beats the benchmark",
                 "subreddit": "LocalLLaMA",
                 "recommended_at": "2026-06-01T03:00:00Z",
                 "posted_at": "2026-06-01T01:00:00Z", "delay_hours": 2.0,
                 "descendants": 210, "pop": 0.9, "weight": 0.84,
                 "status": "tail", "growth": {"1h": 4, "3h": 30}})
    memory.append_record(
        t="2026-06-02T06:00:00Z", src="oracle",
        outcome={"kind": "post", "root_id": "t3_zzz999aa",
                 "title": "Quiet post that blew up",
                 "subreddit": "MachineLearning",
                 "posted_at": "2026-06-01T06:00:00Z", "descendants": 180,
                 "pop": 0.8, "status": "tail", "growth": {"1h": 1, "3h": 9}})
    block = memory.render_block()
    assert "t3_" not in block  # root ids never print
    assert "r/LocalLLaMA" in block and "r/MachineLearning" in block
    assert "weight 0.84 (recommended 2.0h after posting)" in block
    assert "New local model beats the benchmark" in block
    assert "Quiet post that blew up" in block
    assert "growth 1h:4 3h:30" in block
    assert "weight — n 1, sum 0.84, mean 0.84 (over own records carrying " \
        "the field)" in block
    assert "### Cumulative outcomes by subreddit" in block
    assert "own actions settled — tail: 1" in block
    assert "other outcomes settled — tail: 1" in block
