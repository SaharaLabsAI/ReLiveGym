"""Env-tool surface: visibility, leakage, billing, manifest, instruction."""

from __future__ import annotations

import asyncio
import json
import string
from datetime import timedelta
from pathlib import Path

import pytest

from harness.config import AgentSpec, RunConfig
from harness.env_tools import ToolError, build_registry
from harness.runtime import Sim
from harness.task import NotificationError

from .conftest import SIM_END, T0, t

FORBIDDEN_KEYS = {"t_res", "resolution_answer", "answer_status", "easy",
                  "resolved_in_window", "uma_status", "ta_bss",
                  "ta_bss_market", "price", "prices", "volume_usd"}


def _cfg() -> RunConfig:
    return RunConfig(run_id="fp-test", task={"name": "forecast_portfolio"},
                     sim_start=T0, sim_end=SIM_END,
                     agent=AgentSpec(scaffold="uniform"))


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
    assert names == {"list_questions", "get_question", "get_forecasts",
                     "search_news", "get_article", "submit_forecast"}
    action = [d for d, _ in registry.values() if "action" in d.tags]
    assert [d.name for d in action] == ["submit_forecast"]
    assert not any("price" in d.name for d, _ in registry.values())


# -- question visibility --------------------------------------------------------------


def test_index_hides_unactivated_and_reveals_on_time(sim, call):
    rows = call("list_questions")["questions"]
    # q_late opens day 1, q_mid day 2 — both hidden at sim start
    assert {r["question_id"] for r in rows} == {"q_res", "q_lag", "q_open"}
    sim.clock.advance_to(t(2))
    ids = {r["question_id"] for r in call("list_questions")["questions"]}
    assert {"q_mid", "q_late"} <= ids


def test_added_after_and_status_filters(sim, call):
    sim.clock.advance_to(t(6.6))
    fresh = call("list_questions", added_after=t(0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"))["questions"]
    assert {r["question_id"] for r in fresh} == {"q_mid", "q_late"}
    resolved = call("list_questions", status="resolved")["questions"]
    assert {r["question_id"] for r in resolved} == {"q_res", "q_lag"}
    assert all("outcome" in r and "resolved_at" in r for r in resolved)
    open_rows = call("list_questions", status="open")["questions"]
    assert all("outcome" not in r for r in open_rows)


def test_outcome_hidden_until_public_resolution(sim, call):
    sim.clock.advance_to(t(6.2))  # q_lag past t_res (day 6), closes 6.5
    row = next(r for r in call("list_questions")["questions"]
               if r["question_id"] == "q_lag")
    assert row["status"] == "open" and "outcome" not in row
    sim.clock.advance_to(t(6.6))
    row = call("get_question", question_id="q_lag")
    assert row["status"] == "resolved" and row["outcome"] == "No"


def test_unknown_and_unactivated_detail_errors_identical(call):
    with pytest.raises(ToolError) as unknown:
        call("get_question", question_id="ghost")
    with pytest.raises(ToolError) as unactivated:
        call("get_question", question_id="q_mid")
    assert (str(unknown.value).replace("ghost", "q_mid")
            == str(unactivated.value))


def test_no_forbidden_fields_in_free_payloads(sim, call):
    call("submit_forecast", question_id="q_res", forecast={"Yes": 0.6})
    for when in (t(0.5), t(6.6), t(9)):
        sim.clock.advance_to(when)
        _walk_no_forbidden(call("list_questions"))
        _walk_no_forbidden(call("get_question", question_id="q_res"))
        _walk_no_forbidden(call("get_forecasts"))


# -- own-submission history -----------------------------------------------------------


def test_get_forecasts_is_own_action_echo(sim, call):
    assert call("get_forecasts")["forecasts"] == []
    call("submit_forecast", question_id="q_res", forecast={"Yes": 0.6})
    call("submit_forecast", question_id="q_res",
         forecast={"Yes": 0.8, "No": 0.1})
    rows = call("get_forecasts", question_id="q_res")["forecasts"]
    assert len(rows) == 1
    assert rows[0]["current"] == {"Yes": 0.8, "No": 0.1}
    assert [h["forecast"] for h in rows[0]["history"]] == [
        {"Yes": 0.6}, {"Yes": 0.8, "No": 0.1}]


def test_submit_books_ledger_and_free(sim, call):
    call("submit_forecast", question_id="q_res", forecast={"Yes": 0.6})
    assert sim.wallet_spend == 0.0
    events = [json.loads(line) for line in
              (sim.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert any(e["type"] == "submit_forecast" for e in events)


def test_citation_is_free_recorded_and_echoed(sim, call):
    call("submit_forecast", question_id="q_res", forecast={"Yes": 0.6},
         news_id="n1")
    call("submit_forecast", question_id="q_res", forecast={"Yes": 0.7})
    assert sim.wallet_spend == 0.0  # citing is not a paid re-read
    history = call("get_forecasts", question_id="q_res")["forecasts"][0][
        "history"]
    assert history[0]["news_id"] == "n1"
    assert "news_id" not in history[1]  # key absent, not null
    payloads = [json.loads(line)["payload"] for line in
                (sim.run_dir / "ledger.jsonl").read_text().splitlines()
                if json.loads(line)["type"] == "submit_forecast"]
    assert [p.get("news_id") for p in payloads] == ["n1", None]


def test_uncited_and_unpublished_citations_reject_free(sim, call):
    for bad in ("ghost", "n2"):  # n2 is published on day 5, we are at day 0
        with pytest.raises(NotificationError, match="not-yet-published"):
            call("submit_forecast", question_id="q_res",
                 forecast={"Yes": 0.6}, news_id=bad)
    assert sim.wallet_spend == 0.0
    assert call("get_forecasts")["forecasts"] == []


def test_rejected_submission_costs_nothing(sim, call):
    # NotificationError maps to a free 400 in the API layer, like ToolError
    with pytest.raises(NotificationError):
        call("submit_forecast", question_id="q_res",
             forecast={"Yes": 2.0})
    assert sim.wallet_spend == 0.0


# -- news billing ---------------------------------------------------------------------


def test_news_calls_bill_bnpm_types(sim, call):
    call("search_news", q="ceasefire")
    call("get_article", news_id="n1")
    assert sim.wallet_spend == pytest.approx(0.004)
    assert sim.domain_spend.get("news_search") == pytest.approx(0.002)
    assert sim.domain_spend.get("article_call") == pytest.approx(0.002)


# -- instruction ----------------------------------------------------------------------


def test_instruction_renders_fully(task):
    template = string.Template(
        (Path("tasks/forecast_portfolio/INSTRUCTION.md").read_text()))
    harness_ctx = {"budget_usd": "$70", "domain_caps": "",
                   "llm_price_table": "list prices"}
    rendered = template.substitute(
        {**harness_ctx, **task.instruction_context()})
    assert "${" not in rendered
    for word in ("Polymarket", "market price", "price feed"):
        assert word not in rendered


def test_authored_example_ships(task):
    text = task.authored_example()
    assert "handover" in text and "list_questions" in text
