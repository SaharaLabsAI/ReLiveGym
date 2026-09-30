"""Extended TM-B: authored wait programs (harness/authored.py).

Covers provisioning, the workspace jail + file bus, envkit/example
discoverability, the time/pricing contract, handover + its size limit,
trigger/timeout/experiment-end returns of control, redaction through the
program path, validate mode, and the step-cap liveness backstop. The
multi-actor wait-party jail test lands with the first authored-program
party scaffold (no such scaffold exists yet)."""

from __future__ import annotations

import asyncio
import json

import pytest

import harness.authored as authored
from harness.api import build_tool_registry
from harness.env_tools import ToolError
from harness.runtime import Sim
from tasks.reddit_ai_popularity.task import RedditPopularityTask
from tests.tasks.reddit_ai_popularity.conftest import make_reddit_config, t


def _sim(built, tmp_path, tm="B"):
    cfg = make_reddit_config(
        built,
        run_id=f"authored-{tm}",
        cell=dict(tm=tm, tlrn="none", sig="none", alg="none"),
        agent=dict(scaffold="react"),
    )
    run_dir = tmp_path / f"run-{tm}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    task = RedditPopularityTask.from_run_config(cfg, built.parent)
    return Sim(cfg, run_dir, workspace, task)


def call(sim, name, **args):
    entry = build_tool_registry(sim).get(name)
    assert entry is not None, f"tool {name!r} not provisioned"
    return asyncio.run(entry[1](args))


AUTHORED_TOOLS = {"ls", "read_file", "write_file", "edit_file", "run_program"}


# -- provisioning ----------------------------------------------------------------------


def test_provisioned_for_tmB_only(built, tmp_path):
    b = set(build_tool_registry(_sim(built, tmp_path, "B")))
    a = set(build_tool_registry(_sim(built, tmp_path, "A")))
    assert AUTHORED_TOOLS <= b
    assert not (AUTHORED_TOOLS & a)
    # TM-A waits with the timed sleep; TM-B holds no sleep tool at all —
    # run_program is its whole timing surface
    assert "sleep" in a and "sleep" not in b


# -- workspace jail + file bus ---------------------------------------------------------


def test_jail_rejects_escapes(built, tmp_path):
    sim = _sim(built, tmp_path)
    for path in ("../above", "../../tasks/x", "/etc/passwd",
                 "a/../../escape"):
        with pytest.raises(ToolError, match="outside your workspace"):
            call(sim, "write_file", path=path, content="x")
    with pytest.raises(ToolError, match="outside your workspace"):
        call(sim, "read_file", path="../../data/built/roots.jsonl")


def test_file_bus_roundtrip_and_edit(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="params.json", content='{"thresh": 3}')
    got = call(sim, "read_file", path="params.json")
    assert got["content"] == '{"thresh": 3}' and got["total_lines"] == 1
    call(sim, "edit_file", path="params.json", old='"thresh": 3',
         new='"thresh": 5')
    assert '"thresh": 5' in call(sim, "read_file",
                                 path="params.json")["content"]
    with pytest.raises(ToolError, match="occurs 0 times"):
        call(sim, "edit_file", path="params.json", old="absent", new="x")


def test_read_file_page_clipped_at_byte_limit(built, tmp_path):
    """Line-based paging alone let one huge single-line file (an authored
    program's JSON state) blow up the actor's context — the month-run
    failure. A page is clipped at READ_LIMIT_KB with a legible note;
    line-ranged reads below the cap are untouched."""
    sim = _sim(built, tmp_path)
    big = "x" * (authored.READ_LIMIT_KB * 1024 * 3)  # one 48 KB line
    call(sim, "write_file", path="state.json", content=big + "\nsmall\n")
    res = call(sim, "read_file", path="state.json")
    assert len(res["content"].encode()) == authored.READ_LIMIT_KB * 1024
    assert res["truncated"] is True
    assert "KB read limit" in res["clipped"]
    # paging past the huge line still works and stays unclipped
    res = call(sim, "read_file", path="state.json", offset=2)
    assert res["content"] == "small\n" and "clipped" not in res


def test_workspace_ships_envkit_and_example(built, tmp_path):
    sim = _sim(built, tmp_path)
    files = {f["path"] for f in call(sim, "ls")["files"]}
    assert {"envkit.py", "example_gatekeeper.py"} <= files
    kit = call(sim, "read_file", path="envkit.py")["content"]
    for fn in ("def wait(", "def handover(", "def workspace(",
               "def list_posts(", "def get_cascade(", "def recommend("):
        assert fn in kit
    assert "wait_until" not in kit  # sim time moves only via envkit.wait
    example = call(sim, "read_file", path="example_gatekeeper.py")["content"]
    compile(example, "example_gatekeeper.py", "exec")
    # the shipped example dry-runs clean and reaches its first fetch
    res = call(sim, "run_program", path="example_gatekeeper.py",
               validate=True)
    assert res["validate"] == "ok" and res["reached"] == "list_posts"
    # the jail root sits under workspace/agents/, above which the runtime's
    # own state lives, unreachable by these tools
    assert call(sim, "ls")["root"] == "agents/actor"


# -- validate mode (D6) ----------------------------------------------------------------


def test_validate_surfaces_errors_free(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="bad.py", content="def broken(:\n")
    res = call(sim, "run_program", path="bad.py", validate=True)
    assert res["validate"] == "error" and "syntax error" in res["error"]
    assert sim.clock.now == t(2)  # zero sim time
    assert sim.ledger.total_cost() == 0.0  # zero cost


# -- run-and-wait: time, pricing, handover ---------------------------------------------

POLLER = """\
from datetime import timedelta
from envkit import handover, list_posts, now, wait

start = now()
while True:
    page = list_posts(since=start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    if page["posts"]:
        handover({"ids": [p["id"] for p in page["posts"]]})
    wait(now() + timedelta(hours=1))
"""


def test_run_to_handover_time_and_pricing(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="poll.py", content=POLLER)
    res = call(sim, "run_program", path="poll.py", until="2026-03-03T00:00:00Z")
    # sim starts Mar 2 00:00; first post ("big") lands 06:00 — the hourly
    # poller finds it on its 7th fetch, at exactly 06:00
    assert res["woke_for"] == "handover"
    assert res["payload"] == {"ids": ["big"]}
    assert res["now"] == "2026-03-02T06:00:00Z"
    counts = sim.ledger.count_by_type()
    assert counts["reddit_list"] == 7  # each fetch billed...
    # ...waits recorded but free, and never a wait_until
    assert counts["sleep"] == 6 and "wait_until" not in counts
    costs = sim.ledger.cost_by_type()
    assert costs.get("sleep", 0.0) == 0.0
    assert costs["reddit_list"] == pytest.approx(7 * 0.00024)
    assert costs.get("run_program", 0.0) == 0.0
    e = sim.ledger.count_by_type()
    assert e["run_program"] == 1


def test_timeout_returns_control(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="idle.py", content=(
        "from datetime import timedelta\n"
        "from envkit import now, wait\n"
        "while True:\n"
        "    wait(now() + timedelta(hours=1))\n"))
    res = call(sim, "run_program", path="idle.py",
               until="2026-03-02T03:00:00Z")
    assert res["woke_for"] == "timeout"
    assert res["now"] == "2026-03-02T03:00:00Z"


def test_trigger_exits_program(built, tmp_path):
    sim = _sim(built, tmp_path)
    sim.schedule.run_at("learn", t(2, 2))
    call(sim, "write_file", path="idle.py", content=(
        "from datetime import timedelta\n"
        "from envkit import now, wait\n"
        "while True:\n"
        "    wait(now() + timedelta(hours=6))\n"))
    res = call(sim, "run_program", path="idle.py",
               until="2026-03-02T12:00:00Z")
    assert res["woke_for"] == "trigger"
    assert res["trigger"]["id"] == "learn"
    assert res["now"] == "2026-03-02T02:00:00Z"


def test_experiment_end_exits_program(built, tmp_path):
    # a wait past sim_end returns experiment_over to the spine
    cfg = make_reddit_config(
        built, run_id="authored-end",
        cell=dict(tm="B", tlrn="none", sig="none", alg="none"),
        agent=dict(scaffold="react"), sim_end=t(2, 12),
    )
    run_dir = tmp_path / "run-end"
    (run_dir / "workspace").mkdir(parents=True)
    task = RedditPopularityTask.from_run_config(cfg, built.parent)
    sim = Sim(cfg, run_dir, run_dir / "workspace", task)
    call(sim, "write_file", path="idle.py", content=(
        "from datetime import timedelta\n"
        "from envkit import now, wait\n"
        "while True:\n"
        "    wait(now() + timedelta(days=30))\n"))
    res = call(sim, "run_program", path="idle.py",
               until="2026-12-31T00:00:00Z")  # clamped to sim_end
    assert res.get("experiment_over") is True


def test_handover_size_limit_and_file_bus_escape_hatch(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="big.py", content=(
        "from envkit import handover\n"
        "handover({'blob': 'x' * 50_000})\n"))
    res = call(sim, "run_program", path="big.py", until="2026-03-03T00:00:00Z")
    assert res["woke_for"] == "error"
    assert "over the 16 KB limit" in res["error"]
    assert "hand over a pointer" in res["error"]
    # sanctioned pattern: bulk to a file, pointer over the wire
    call(sim, "write_file", path="ptr.py", content=(
        "from envkit import handover, workspace\n"
        "(workspace() / 'bulk.json').write_text('y' * 50_000)\n"
        "handover({'file': 'bulk.json', 'n': 50_000})\n"))
    res = call(sim, "run_program", path="ptr.py", until="2026-03-03T00:00:00Z")
    assert res["woke_for"] == "handover" and res["payload"]["file"] == "bulk.json"
    assert call(sim, "read_file", path="bulk.json")["content"].startswith("y")


# -- isolation / redaction through the program path ------------------------------------


def test_program_sees_only_redacted_fields(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="probe.py", content=(
        "from datetime import timedelta\n"
        "from envkit import get_post, handover, now, wait\n"
        "wait(now() + timedelta(hours=7))  # 'big' posted 06:00, unrevealed\n"
        "p = get_post(root_id='big')\n"
        "handover({'keys': sorted(p), 'descendants': p['descendants'],\n"
        "          'num_comments': p['num_comments']})\n"))
    res = call(sim, "run_program", path="probe.py",
               until="2026-03-03T00:00:00Z")
    assert res["woke_for"] == "handover"
    assert "score" not in res["payload"]["keys"]
    assert "retrieved_on" not in res["payload"]["keys"]
    assert res["payload"]["descendants"] is None  # label hidden pre-reveal
    # live prefix count is served (all 256 synthetic comments land in the
    # first ~21 min, so at +7h the prefix already equals the hidden label)
    assert res["payload"]["num_comments"] == 256


def test_program_actions_are_scored_actions(built, tmp_path):
    """recommend from inside a program is the same priced action tool."""
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="act.py", content=(
        "from datetime import timedelta\n"
        "from envkit import handover, now, recommend, wait\n"
        "wait(now() + timedelta(hours=7))\n"
        "handover(recommend(root_id='big'))\n"))
    res = call(sim, "run_program", path="act.py", until="2026-03-03T00:00:00Z")
    assert res["payload"]["status"] == "accepted"
    assert sim.ledger.count_by_type()["notify"] == 1


# -- legible guard errors + liveness --------------------------------------


def test_runtime_error_is_one_line_with_location(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="boom.py", content="x = 1\ny = x / 0\n")
    res = call(sim, "run_program", path="boom.py",
               until="2026-03-03T00:00:00Z")
    assert res["woke_for"] == "error"
    assert "ZeroDivisionError" in res["error"]
    assert "boom.py line 2" in res["error"]


def test_step_cap_breaks_waitless_loops(built, tmp_path, monkeypatch):
    monkeypatch.setattr(authored, "STEP_CAP", 10_000)
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="spin.py", content=(
        "while True:\n    pass\n"))
    res = call(sim, "run_program", path="spin.py",
               until="2026-03-03T00:00:00Z")
    assert res["woke_for"] == "error"
    assert "step budget exceeded" in res["error"]
    assert sim.clock.now == t(2)  # no sim-time progress without wait


def test_envkit_fails_cleanly_outside_run_program(built, tmp_path):
    sim = _sim(built, tmp_path)
    call(sim, "ls")  # provision the jail
    kit = sim.workspace / "agents" / "actor" / "envkit.py"
    ns: dict = {}
    exec(compile(kit.read_text(), str(kit), "exec"), ns)
    with pytest.raises(RuntimeError, match="only inside run_program"):
        ns["now"]()


def test_run_program_wake_carries_daily_brief(built, tmp_path):
    """Daily cost brief: a run-and-wait return is a wake;
    the first one on a sim date carries `costs`, a validate dry-run and a
    same-date return do not."""
    sim = _sim(built, tmp_path)
    call(sim, "write_file", path="poll.py", content=POLLER)
    dry = call(sim, "run_program", path="poll.py", validate=True)
    assert "costs" not in dry
    res = call(sim, "run_program", path="poll.py", until="2026-03-03T00:00:00Z")
    assert res["now"] == "2026-03-02T06:00:00Z"
    assert res["costs"]["day"] == 1
    assert res["costs"]["llm"]["budget_usd"] == sim.cfg.domain_budgets["llm"]
    assert res["costs"]["spend_by_type"]["reddit_list"] == pytest.approx(7 * 0.00024)
    again = call(sim, "run_program", path="poll.py", until="2026-03-03T00:00:00Z")
    assert again["now"] == "2026-03-02T06:00:00Z" and "costs" not in again
