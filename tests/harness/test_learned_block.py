"""Learned-block stores: one injection path per
representation, one token cap, whole-record dropping, version stamping,
deterministic rendering. Now split across runtime.memory (raw records) and
runtime.skills (curated block)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime import memory, skills  # noqa: E402
from runtime.tokens import count_tokens  # noqa: E402


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """The modules use workspace-relative paths, as programs run
    in-workspace."""
    monkeypatch.chdir(tmp_path)


def test_append_and_render_memory():
    memory.append_record(t="2021-06-01T00:00:00Z", src="oracle",
                         outcome={"story_id": 1, "final_score": 256},
                         action={"did": "recommended"})
    memory.append_record(t="2021-06-02T00:00:00Z", src="self",
                         outcome={"story_id": 2, "final_score": 1}, note="dud")
    block = memory.render_block()
    assert block.startswith("## What you have learned so far")
    lines = block.splitlines()[1:]
    assert len(lines) == 2
    assert "2021-06-01" in lines[0] and "2021-06-02" in lines[1]  # oldest first
    assert "null" not in block  # nulls dropped from the rendering


def test_legend_renders_under_header(monkeypatch):
    """A task-provided LEGEND (records.BLOCK_LEGEND, wired by compose in
    alg=memory cells) renders between the header and the records; the
    default empty legend leaves the block unchanged."""
    memory.append_record(t="2021-06-01T00:00:00Z", src="oracle",
                         outcome={"story_id": 1})
    plain = memory.render_block()
    monkeypatch.setattr(memory, "LEGEND", "Each line is one record.")
    block = memory.render_block()
    lines = block.splitlines()
    assert lines[0].startswith("## What you have learned")
    assert lines[1] == "Each line is one record."
    assert lines[2].startswith("{")
    assert plain.splitlines()[1].startswith("{")  # no legend by default


def test_memory_cap_drops_whole_oldest_records():
    for k in range(200):
        memory.append_record(t=f"2021-06-01T{k // 60:02d}:{k % 60:02d}:00Z",
                             src="oracle", outcome={"k": k, "pad": "x" * 200})
    block = memory.render_block(budget_tokens=500)
    assert count_tokens(block) <= 500 + 40  # header slack only
    lines = block.splitlines()[1:]
    assert all(line.startswith("{") and line.endswith("}") for line in lines)
    assert '"k":199' in lines[-1]  # newest records survive
    assert '"k":0' not in block  # oldest dropped


def test_skills_roundtrip_and_versioning():
    assert skills.render_block() == ""
    assert skills.block_version() == 0
    skills.update_skills("Rule 1: niche is core.")
    assert skills.block_version() == 1
    assert "Rule 1" in skills.render_block()
    skills.update_skills("Rule 1 (rev).")
    assert skills.block_version() == 2


def test_skills_truncated_to_budget():
    skills.update_skills("word " * 5000)
    block = skills.render_block(budget_tokens=100)
    assert count_tokens(block) <= 140  # budget + header slack


def test_rendering_deterministic():
    for k in range(5):
        memory.append_record(t=f"2021-06-0{k + 1}T00:00:00Z", src="oracle",
                             outcome={"k": k})
    assert memory.render_block() == memory.render_block()
