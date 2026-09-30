"""Reflection input contract: the
own/other split counts unlinked alert outcomes as own actions, bare ids
resolve to meaning at render time, the cost report feeds spend to the
learning step, and the fully rendered prompt carries all of it. The
get_costs endpoint is spend-only (penalties stay sig-gated)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime import memory, reflect, skills  # noqa: E402
from runtime.env_client import EnvError  # noqa: E402

UTC = timezone.utc


def _load_bnpm_records():
    spec = importlib.util.spec_from_file_location(
        "bnpm_records",
        REPO_ROOT / "tasks" / "breakout_news_pm" / "agent" / "records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BNPM_RECORDS = _load_bnpm_records()


class StubEnv:
    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        if name not in self.responses:
            raise EnvError(f"unknown tool {name!r}")
        r = self.responses[name]
        if isinstance(r, Exception):
            raise r
        return r

    def now(self):
        return datetime(2026, 3, 3, 0, 30, tzinfo=UTC)


MARKETS = [
    {"market_id": "616902", "question": "Will no Fed rate cuts happen in "
     "2026?", "description": "Resolves YES if the Fed makes no cuts.",
     "start": "2026-03-01T00:00:00Z", "end": "2026-03-28T00:00:00Z"},
]

ALERT = {"t": "2026-03-03T00:30:00Z", "src": "oracle",
         "outcome": {"kind": "alert", "market_id": "616902",
                     "news_id": "abc123", "direction": "up",
                     "status": "false_alarm"},
         "action": None, "note": None}
BREAKPOINT = {"t": "2026-03-03T00:30:00Z", "src": "oracle",
              "outcome": {"kind": "breakpoint", "market_id": "616902",
                          "status": "miss", "credit": 0.0,
                          "winnable": True,
                          "gold_groups": [{"story": "Fed shock.",
                                           "articles": [
                                               {"news_id": "abc123"},
                                               {"news_id": "unseen1"}]}]},
              "action": None, "note": None}
LINKED = {"t": "2026-03-03T00:30:00Z", "src": "oracle",
          "outcome": {"kind": "breakpoint", "market_id": "616902",
                      "status": "covered_news", "credit": 0.5,
                      "news_id": "abc123"},
          "action": {"did": "alerted", "title": "Fed holds rates"},
          "note": None}


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_own_split_counts_alert_outcomes_without_action():
    dist, own, examples = reflect.reflection_view(
        [ALERT, BREAKPOINT, LINKED], BNPM_RECORDS.stratum_of,
        BNPM_RECORDS.is_own)
    assert '"kind":"alert"' in own          # unlinked alert is still own
    assert '"did":"alerted"' in own         # linked record is own
    # both kinds counted, on separately labeled lines (review finding 4)
    assert "own actions settled — covered_news: 1, false_alarm: 1" in dist
    assert "other outcomes settled — miss: 1" in dist
    assert '"status":"miss"' in examples and '"kind":"alert"' not in examples


def test_cumulative_sections():
    records = [ALERT, BREAKPOINT, LINKED]
    ledger = reflect.entity_ledger(records, BNPM_RECORDS.stratum_of,
                                   BNPM_RECORDS.entity_of)
    assert "616902: covered_news 1, false_alarm 1, miss 1" == ledger


def test_digest_line_breakpoint_with_gold():
    bp = json.loads(json.dumps(BREAKPOINT))
    o = bp["outcome"]
    o["t_move_start"] = "2026-03-06T13:41:24Z"
    o["direction"] = "down"
    o["gold_groups"] = [
        {"story": "Minor echo.", "confidence": 0.3,
         "articles": [{"news_id": "x", "published_at": "2026-03-06T13:00:00Z"}]},
        {"story": "Jobs report shocked markets.", "confidence": 0.95,
         "articles": [{"news_id": "abc123",
                       "published_at": "2026-03-06T13:33:16Z",
                       "seen_by_you": True}]},
    ]
    line = BNPM_RECORDS.digest_line(bp)
    assert line.startswith("2026-03-06 | 616902 | down | miss")
    assert "lead 0.1h pub->move" in line      # highest-confidence group wins
    assert "gold: Jobs report shocked markets." in line
    assert "(seen_by_you)" in line
    assert "Minor echo" not in line


def test_digest_line_sig_self_and_unwinnable():
    verdict = {"t": "2026-03-05T00:30:00Z", "src": "self",
               "outcome": {"verdict": "no_breakout", "market_id": "616902",
                           "news_id": "abc123",
                           "news_title": "Fed holds rates"}}
    assert BNPM_RECORDS.digest_line(verdict) == \
        "2026-03-05 | 616902 | verdict no_breakout | Fed holds rates"
    unwin = {"t": "2026-03-04T00:30:00Z", "src": "oracle",
             "outcome": {"kind": "breakpoint", "market_id": "582717",
                         "t_move_start": "2026-03-03T09:00:00Z",
                         "direction": "up", "status": "miss",
                         "no_attributable_news": True}}
    assert "no attributable news (unwinnable)" in BNPM_RECORDS.digest_line(unwin)


def test_own_split_backwards_compatible_without_is_own():
    _, own, examples = reflect.reflection_view(
        [ALERT, BREAKPOINT], BNPM_RECORDS.stratum_of)
    assert own == ""                        # old behavior: linkage only
    assert '"kind":"alert"' in examples


def test_own_history_capped_with_omission_note(monkeypatch):
    Path("prompts").mkdir()
    Path("prompts/reflect.md").write_text("OWN:\n${own_history}\nNEW skill "
                                          "block.")
    for k in range(40):
        memory.append_record(
            t=f"2026-03-01T{k // 60:02d}:{k % 60:02d}:00Z", src="oracle",
            outcome={"kind": "alert", "k": k, "status": "false_alarm",
                     "penalty": 10.0}, action=None, note=None)
    env = StubEnv({"get_markets": MARKETS,
                   "get_costs": {"spend_by_type": {}, "spend_total": 0.0}})
    captured = {}
    monkeypatch.setattr(
        reflect.llm_client, "chat",
        lambda e, m: captured.setdefault("p", m[0]["content"]) and "S")
    reflect.reflect(env, {}, BNPM_RECORDS)
    p = captured["p"]
    assert "(oldest 10 of 40 omitted" in p
    assert '"k":39' in p and '"k":9' not in p  # newest kept, oldest dropped


def test_enrich_resolves_ids_on_copies():
    ctx = {"market_questions": {"616902": "Will no Fed rate cuts happen "
                                "in 2026?"},
           "news_titles": {"abc123": "Fed holds rates"}}
    out = reflect.enrich([ALERT, BREAKPOINT], ctx, BNPM_RECORDS.enrich)
    assert out[0]["outcome"]["news_title"] == "Fed holds rates"
    assert "Fed rate cuts" in out[0]["outcome"]["market_question"]
    gold = out[1]["outcome"]["gold_groups"][0]["articles"]
    assert gold[0]["title"] == "Fed holds rates" and gold[0]["seen_by_you"]
    assert "title" not in gold[1]           # never observed: no resolution
    assert "news_title" not in ALERT["outcome"]  # originals untouched


def test_render_context_reads_markets_and_registry():
    env = StubEnv({"get_markets": MARKETS})
    state = {"registered": {"abc123": {"title": "Fed holds rates",
                                       "published": "2026-03-01T06:00:00Z"},
                            "notitle": {}}}
    ctx = BNPM_RECORDS.render_context(env, state)
    markets = ctx["extra_slots"]["markets"]
    assert "616902" in markets and "resolution:" in markets
    assert ctx["market_questions"]["616902"].startswith("Will no Fed")
    assert ctx["news_titles"] == {"abc123": "Fed holds rates"}


def test_cost_report_delta_posture_and_state():
    env = StubEnv({"get_costs": {
        "spend_by_type": {"news_search": 1.0, "llm": 2.0},
        "spend_total": 3.0}})
    state = {"last_costs": {"news_search": 0.4},
             "last_wait": {"until": "2026-03-05T00:00:00Z"}}
    report = reflect.cost_report(env, state)
    assert "news_search $0.60" in report and "llm $2.00" in report
    assert "cumulative spend: $3.00" in report
    assert "2026-03-05" in report            # standing posture rendered
    assert state["last_costs"] == {"news_search": 1.0, "llm": 2.0}
    # second call, no new spend
    second = reflect.cost_report(env, state)
    assert "spend since last reflection: none" in second


def test_cost_report_survives_missing_endpoint():
    assert "unavailable" in reflect.cost_report(StubEnv({}), {})


def test_prepared_snapshot_shares_one_cost_read_across_consumers(monkeypatch):
    Path("prompts").mkdir()
    Path("prompts/one.md").write_text(
        "${run_header}\n${costs}\n${monitor_performance}\n${skills}")
    Path("INSTRUCTION.md").write_text("scored task economics")
    env = StubEnv({
        "get_markets": MARKETS,
        "get_costs": {"spend_by_type": {"llm": 1.0},
                      "spend_total": 1.0},
    })
    state = {}
    snapshot = reflect.prepare_reflection(env, state, BNPM_RECORDS)
    first = reflect.render_reflection_prompt(
        snapshot, "prompts/one.md", {"monitor_performance": "shared"})
    second = reflect.render_reflection_prompt(
        snapshot, "prompts/one.md", {"monitor_performance": "shared"})
    monkeypatch.setattr(reflect.llm_client, "chat", lambda *_args: "RULE")
    reflect.curate_skills(
        env, state, snapshot, template_path="prompts/one.md",
        extra_slots={"monitor_performance": "shared"})
    reflect.finish_reflection(state, snapshot)
    assert first == second
    assert [name for name, _ in env.calls].count("get_costs") == 1
    assert state["last_reflection"] == snapshot["t"]


def _mini_template() -> str:
    return ("HEAD:\n${run_header}\nSPEC:\n${instruction}\nMARKETS:\n"
            "${markets}\nSKILLS:\n${skills}\nCUM:\n${cum_counts}\n"
            "LEDGER:\n${entity_ledger}\nHIST:\n${own_history}\n"
            "DIST:\n${distribution}\nOWN:\n${own_actions}\n"
            "EX:\n${examples}\nCOSTS:\n${costs}\nWrite the NEW skill block.")


def test_reflect_renders_the_full_contract(monkeypatch):
    Path("prompts").mkdir()
    Path("prompts/reflect.md").write_text(_mini_template())
    Path("INSTRUCTION.md").write_text("maximize tc_f1 within budget")
    for r in (ALERT, BREAKPOINT):
        memory.append_record(**{k: r[k] for k in
                                ("t", "src", "outcome", "action", "note")})
    env = StubEnv({"get_markets": MARKETS,
                   "get_costs": {"spend_by_type": {"news_search": 1.0},
                                 "spend_total": 1.0}})
    state = {"registered": {"abc123": {"title": "Fed holds rates",
                                       "published": "2026-03-01T06:00:00Z"}}}
    captured = {}

    def fake_chat(env_, messages):
        captured["prompt"] = messages[0]["content"]
        return "Rule: silence is golden."

    monkeypatch.setattr(reflect.llm_client, "chat", fake_chat)
    reflect.reflect(env, state, BNPM_RECORDS)

    p = captured["prompt"]
    assert "${" not in p                       # every slot substituted
    assert "maximize tc_f1 within budget" in p
    assert "Will no Fed rate cuts happen in 2026?" in p
    # run-position header: interval vs all-time is explicit
    assert "Reflection #1" in p and "2 settled in total this run" in p
    # the own-alert renders with its content, not just ids
    own_section = p.split("OWN:\n")[1].split("EX:\n")[0]
    assert '"kind":"alert"' in own_section
    assert '"news_title":"Fed holds rates"' in own_section
    # cumulative sections rendered
    assert "616902: false_alarm 1, miss 1" in p
    assert "cumulative spend: $1.00" in p
    assert "penalt" not in p  # dollars are never a score
    assert skills.SKILLS.read_text() == "Rule: silence is golden."
    assert state["last_reflection"] == "2026-03-03T00:30:00Z"
    assert state["reflection_count"] == 1


def test_reflect_with_real_bnpm_template(monkeypatch):
    """The real task template: fenced sections, glossary, persona
    disclaimer, and every slot substituted (prompt review 2026-07-26)."""
    import shutil
    Path("prompts").mkdir()
    shutil.copyfile(REPO_ROOT / "tasks" / "breakout_news_pm" / "agent"
                    / "prompts" / "reflect.md", "prompts/reflect.md")
    Path("INSTRUCTION.md").write_text("# Task\nYou are a long-running "
                                      "automation agent.")
    for r in (ALERT, BREAKPOINT):
        memory.append_record(**{k: r[k] for k in
                                ("t", "src", "outcome", "action", "note")})
    env = StubEnv({"get_markets": MARKETS,
                   "get_costs": {"spend_by_type": {"news_search": 0.5},
                                 "spend_total": 0.5}})
    captured = {}
    monkeypatch.setattr(
        reflect.llm_client, "chat",
        lambda e, m: captured.setdefault("p", m[0]["content"]) and "S")
    reflect.reflect(env, {}, BNPM_RECORDS)
    p = captured["p"]
    assert "${" not in p
    assert "<<<SPEC" in p and "SPEC>>>" in p        # quoted spec fenced
    assert "<<<SKILL" in p and "SKILL>>>" in p      # skill block fenced
    assert "ACTING PROGRAM, not you" in p           # persona disclaimer
    assert "===== 2. GLOSSARY" in p and "winnable =" in p
    assert "===== 5. CUMULATIVE EVIDENCE" in p
    # the breakout digest carries the gold story durably, every firing
    assert "Digest of every settled breakout" in p
    assert '| 616902 "Will no Fed rate cuts happen' in p  # id + question
    assert "| ? | miss | gold: Fed shock." in p
    assert "NOT base rates" in p                    # interval labeled
    assert p.index("===== 8. CURRENT SKILL BLOCK") \
        > p.index("===== 5. CUMULATIVE EVIDENCE")   # evidence before block


def test_reflect_backwards_compatible_minimal_records_mod(monkeypatch):
    Path("prompts").mkdir()
    Path("prompts/reflect.md").write_text(
        "SKILLS:\n${skills}\nDIST:\n${distribution}\n"
        "OWN:\n${own_actions}\nEX:\n${examples}\nNEW skill block.")
    memory.append_record(t="2026-03-02T00:00:00Z", src="oracle",
                         outcome={"k": 1}, action=None, note=None)
    env = StubEnv({"get_costs": {"spend_by_type": {}, "spend_total": 0.0}})
    records_mod = SimpleNamespace(stratum_of=lambda r: "x")
    captured = {}
    monkeypatch.setattr(
        reflect.llm_client, "chat",
        lambda e, m: captured.setdefault("p", m[0]["content"]) and "S")
    reflect.reflect(env, {}, records_mod)
    assert "${" not in captured["p"] and "x: 1" in captured["p"]


def test_get_costs_endpoint_reports_spend_and_budgets(tmp_path):
    """Spend by type (zero-cost types dropped), plus the budgets (total and
    LLM) with what remains."""
    from harness.apps import CostsApp
    from harness.runtime import Sim
    from tests.conftest import make_config, make_weather_task, write_weather_csv

    start = datetime(2021, 6, 1, tzinfo=UTC)
    csv = write_weather_csv(tmp_path / "w.csv", start, [20.0] * 24 * 7)
    cfg = make_config(weather_csv=csv, data_cutoff=start, budget_usd=10.0,
                      domain_budgets={"llm": 4.0})
    (tmp_path / "run").mkdir()
    sim = Sim(cfg, tmp_path / "run", tmp_path / "ws", make_weather_task(cfg))
    sim.bill("news_search", 0.02)
    sim.ledger.append("outcome", sim.clock.now)  # settlements are cost-free
    sim.bill("llm", 1.5)
    sim.ledger.append("notify", sim.clock.now)  # zero-cost types are dropped
    out = asyncio.run(CostsApp(sim).get_costs({}))
    assert out["spend_by_type"] == {"llm": 1.5, "news_search": 0.02}
    assert out["spend_total"] == 1.52
    assert "outcome" not in out["spend_by_type"]
    assert out["budget_usd"] == 10.0 and out["remaining_usd"] == 8.48
    assert out["llm"] == {"budget_usd": 4.0, "spent_usd": 1.5,
                          "remaining_usd": 2.5}
    assert "domains" not in out
