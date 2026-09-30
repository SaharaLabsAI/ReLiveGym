"""Env-tool surface: visibility, leakage, billing, manifest, instruction."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path

import pytest

from harness.config import AgentSpec, RunConfig
from harness.env_tools import ToolError, build_registry
from harness.runtime import Sim

from .conftest import SIM_END, T0, t

FORBIDDEN_KEYS = {"t_res", "t_det", "gap_s", "gap", "winnable", "trap",
                  "news_lead_flag", "scored", "resolution_answer",
                  "answer_status", "resolved_in_window", "category",
                  "credit", "price", "prices", "volume_usd"}


def _cfg() -> RunConfig:
    return RunConfig(run_id="rd-test", task={"name": "resolution_detect"},
                     sim_start=T0, sim_end=SIM_END,
                     agent=AgentSpec(scaffold="silent"))


@pytest.fixture
def sim(tmp_path: Path, task) -> Sim:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return Sim(_cfg(), run_dir, tmp_path / "ws", task)


@pytest.fixture
def call(sim):
    registry = build_registry(sim.task.env_apps(sim))

    def _call(name: str, **args):
        return asyncio.run(registry[name][1](args))

    return _call


def _walk_no_forbidden(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN_KEYS, f"leaked key {k!r} at {path}"
            _walk_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_no_forbidden(v, f"{path}[{i}]")


# -- manifest -------------------------------------------------------------------------


def test_manifest_tools_and_no_price_surface(sim):
    registry = build_registry(sim.task.env_apps(sim))
    names = set(registry)
    assert names == {"list_questions", "get_question", "get_marks",
                     "search_news", "get_article", "mark_outcome"}
    action = [d for d, _ in registry.values() if "action" in d.tags]
    assert [d.name for d in action] == ["mark_outcome"]
    assert not any("price" in d.name for d, _ in registry.values())


# -- question visibility --------------------------------------------------------------


def test_index_hides_unactivated_and_reveals_on_time(sim, call):
    rows = call("list_questions")["questions"]
    # q_late opens day 1, q_mid day 2 — both hidden at sim start
    assert {r["question_id"] for r in rows} == {"q_win", "q_lag", "q_unwin",
                                                "q_open"}
    sim.clock.advance_to(t(2))
    ids = {r["question_id"] for r in call("list_questions")["questions"]}
    assert {"q_mid", "q_late"} <= ids


def test_resolution_is_invisible_in_question_payloads(sim, call):
    # no resolution signal of any kind: resolved questions
    # stay listed, byte-identical in shape to open ones — no status, no
    # outcome, no close time. Detecting settlement IS the task.
    sim.clock.advance_to(t(6.2))  # q_lag past t_res (day 6), closes 6.5
    row = next(r for r in call("list_questions")["questions"]
               if r["question_id"] == "q_lag")
    assert set(row) == {"question_id", "question", "added_at"}
    sim.clock.advance_to(t(6.6))  # q_lag now publicly resolved
    row = next(r for r in call("list_questions")["questions"]
               if r["question_id"] == "q_lag")
    assert set(row) == {"question_id", "question", "added_at"}
    detail = call("get_question", question_id="q_lag")
    assert set(detail) == {"question_id", "question", "added_at",
                           "description", "outcomes"}


def test_unknown_and_unactivated_detail_errors_identical(call):
    with pytest.raises(ToolError) as unknown:
        call("get_question", question_id="ghost")
    with pytest.raises(ToolError) as unactivated:
        call("get_question", question_id="q_mid")
    assert (str(unknown.value).replace("ghost", "q_mid")
            == str(unactivated.value))


def test_no_forbidden_fields_in_free_payloads(sim, call):
    call("mark_outcome", question_id="q_win", outcome="Yes")
    for when in (t(0.5), t(6.6), t(9.5)):
        sim.clock.advance_to(when)
        _walk_no_forbidden(call("list_questions"))
        _walk_no_forbidden(call("get_question", question_id="q_win"))
        _walk_no_forbidden(call("get_marks"))


def test_get_marks_is_own_claim_echo(sim, call):
    assert call("get_marks")["marks"] == []
    call("mark_outcome", question_id="q_win", outcome="Yes")
    marks = call("get_marks")["marks"]
    assert marks == [{"question_id": "q_win", "at": marks[0]["at"],
                      "outcome": "Yes"}]
    # settlement info never appears, even after the question resolves
    sim.clock.advance_to(t(9))
    sim.task.close_due(sim.clock.now)
    _walk_no_forbidden(call("get_marks"))


def test_mark_books_ledger_and_free(sim, call):
    sim.clock.advance_to(t(1))  # past the stub article's publish time
    out = call("mark_outcome", question_id="q_win", outcome="Yes",
               news_id="n1")
    assert out["status"] == "accepted"
    assert sim.wallet_spend == 0.0
    events = [json.loads(line) for line in
              (sim.run_dir / "ledger.jsonl").read_text().splitlines()]
    marks = [e for e in events if e["type"] == "mark_outcome"]
    assert len(marks) == 1 and marks[0]["payload"]["news_id"] == "n1"


def test_rejected_mark_costs_nothing(sim, call):
    from harness.task import NotificationError
    for kwargs in ({"question_id": "ghost", "outcome": "Yes"},
                   {"question_id": "q_win", "outcome": "Maybe"}):
        with pytest.raises((ToolError, NotificationError)):
            call("mark_outcome", **kwargs)
    assert sim.wallet_spend == 0.0
    ledger = (sim.run_dir / "ledger.jsonl")
    assert not ledger.exists() or "mark_outcome" not in ledger.read_text()


def test_news_calls_bill_bnpm_types(sim, call):
    call("search_news", q="test")
    call("get_article", news_id="n1")
    assert sim.wallet_spend == pytest.approx(0.004)
    assert sim.domain_spend.get("news_search") == pytest.approx(0.002)
    assert sim.domain_spend.get("article_call") == pytest.approx(0.002)


def test_instruction_renders_fully(task):
    tpl = string.Template(
        Path("tasks/resolution_detect/INSTRUCTION.md").read_text())
    ctx = dict(task.instruction_context(),
               budget_usd="$10", llm_price_table="...", domain_caps="")
    rendered = tpl.substitute(ctx)
    assert "${" not in rendered
    # contract-only: no price feed, no measurement rule
    for banned in ("Polymarket", "market price", "price feed", "0.99",
                   "t_det"):
        assert banned not in rendered


def test_authored_example_ships(task):
    src = task.authored_example()
    assert "handover" in src and "search_news" in src
