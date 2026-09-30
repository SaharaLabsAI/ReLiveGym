"""Cron arms of episode mode (tm=C/D): `prepare --tm D` and the cron
driver (harness/episode_drive.py).

No LLM, no OpenCode: the driver test swaps the firing script for a fake
agent that plays over HTTP. The schedule store and the CRUD app are
covered once in tests/harness/test_schedule.py — not again here."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from harness.mcp import EnvError, _Http, main as mcp_main
from tests.control_plane.test_detached import Served, ext_config, ledger_view
from tests.control_plane.test_mcp_episode import weather_base_yaml

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")

CELL_D = {"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"}
SCHEDULE_TOOLS = {"list_schedules", "create_schedule", "update_schedule",
                  "delete_schedule"}


def test_wake_message():
    """The firing prompt is the cron main's wake marker byte for byte, plus the
    A/B actor prompt on the firing that opens the session."""
    from harness.episode_drive import wake_message
    from scaffolds.runtime.agent import wake_marker

    trig = {"id": "act", "kind": "cron", "now": "2026-04-02T00:00:00Z"}
    plain = "(woke at 2026-04-02T00:00:00+00:00 for trigger act/cron)"
    assert wake_marker("2026-04-02T00:00:00+00:00", trig) == plain
    assert wake_message(trig, opens_session=False) == plain
    costs = {"day": 2, "of": 61, "spent_usd": 1.5, "budget_usd": 20.0,
             "remaining_usd": 18.5,
             "llm": {"spent_usd": 1.5, "budget_usd": 20.0, "remaining_usd": 18.5}}
    full = wake_message({**trig, "id": "pre-open", "kind": "at",
                         "note": "check 10-Qs", "costs": costs},
                        opens_session=True)
    assert full == (
        "(woke at 2026-04-02T00:00:00+00:00 for trigger pre-open/at)\n"
        "Your note for this schedule: check 10-Qs\n"
        "Day 2 of 61. Budget: $1.50 spent of $20.00 ($18.50 left). "
        "LLM budget: $1.50 spent of $20.00 ($18.50 left).\n"
        "Read INSTRUCTION.md in this workspace and follow it.")


def test_prepare_tm_d(tmp_path, capsys, monkeypatch):
    """`prepare --tm D`: the default schedule seeded and `__bootstrap__`
    left for the driver, the cron INSTRUCTION, the per-firing script, the
    tm=D bridge surface, and the clock tools refused to the agent token."""
    from tasks.weather_fixture.task import TASK

    base = weather_base_yaml(tmp_path, "D")
    argv = ["prepare", "--task", "weather_fixture", "--tm", "D", "--base", str(base),
            "--out", str(tmp_path / "episodes"), "--mock-llm",
            "--model", "openai:gpt-5.6-luna"]
    with pytest.raises(SystemExit, match="--app opencode"):
        mcp_main(argv)
    with pytest.raises(SystemExit, match="episode_act_cron"):
        mcp_main(argv + ["--app", "opencode"])
    monkeypatch.setattr(TASK, "episode_act_cron", "0 0 * * *")
    assert mcp_main(argv + ["--app", "opencode"]) == 0
    summary = json.loads(capsys.readouterr().out)
    ws, ep_path = Path(summary["workspace"]), Path(summary["episode"])
    ep = json.loads(ep_path.read_text())
    assert (ep["driver"], ep["act_cron"], ep["trigger_id"]) == \
        ("cron", "0 0 * * *", None)

    instruction = (ws / "INSTRUCTION.md").read_text()
    assert "# Your schedule\nA default schedule wakes you on cron `0 0 * * *`" \
        in instruction
    assert "passes only between your wakings" in instruction
    assert "wait tool" not in instruction

    script = Path(ep["firing_script"])
    assert script.name == "run_opencode_firing.sh"
    assert not (ep_path.parent / "run_opencode.sh").exists()
    text = script.read_text()
    assert "printf '%s' \"$ENV_WAKE_MESSAGE\" | " in text
    assert '${ENV_OPENCODE_SESSION:+--session "$ENV_OPENCODE_SESSION"} >> ' in text

    control = _Http(ep["env_url"], ep["token"])
    assert control.get("/status")["active_trigger"] is None  # bootstrap unconsumed
    agent = _Http(ep["env_url"], ep["agent_token"])
    rows = agent.call("list_schedules", {})["schedules"]
    assert [(r["id"], r["owner"], r["cron_expr"]) for r in rows] == \
        [("act", "agent", "0 0 * * *")]
    for name, args in (("sleep", {"until": "2021-06-02T00:00:00Z"}),
                       ("set_crontab", {"entries": []})):
        with pytest.raises(EnvError, match="HTTP 403"):
            agent.call(name, args)

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    bridge = json.loads((ws / "opencode.json").read_text())["mcp"]["env"]

    async def surface():
        params = StdioServerParameters(
            command=sys.executable, args=bridge["command"][1:],
            env={**os.environ, **bridge["environment"]})
        async with stdio_client(params, errlog=open(os.devnull, "w")) as (r, w):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                return {t.name for t in (await sess.list_tools()).tools}

    assert asyncio.run(surface()) == {"get_time", "get_costs", "get_weather",
                                      "notify"} | SCHEDULE_TOOLS
    assert mcp_main(["settle", "--episode", str(ep_path)]) == 0


FAKE_AGENT = '''\
import json, os, sys, urllib.request

URL, TOKEN, DIR = {url!r}, {token!r}, {dir!r}


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={{"Authorization": "Bearer " + TOKEN,
                                          "Content-Type": "application/json"}})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        return {{"status": e.code}}


seen = os.path.join(DIR, "seen.jsonl")
n = sum(1 for _ in open(seen)) if os.path.exists(seen) else 0
with open(seen, "a") as f:
    f.write(json.dumps({{"message": sys.stdin.read(),
                        "session": os.environ["ENV_OPENCODE_SESSION"]}}) + "\\n")
with open(os.path.join(DIR, "opencode_events.jsonl"), "a") as f:
    f.write(json.dumps({{"type": "step_start", "sessionID": "ses_fake"}}) + "\\n")
if n == 0:    # opens the session: schedule a second look, with a note
    post("/call/create_schedule", {{"id": "look", "note": "look again",
                                   "at": "2021-06-02T06:00:00Z"}})
elif n == 1:  # crash
    sys.exit(3)
elif n == 2:  # spend the budget: every later firing is skipped
    assert post("/llm/chat/completions", {{"model": "mock-luna",
                                          "messages": []}})["status"] == 503
'''


def test_drive_fire_sequence(tmp_path):
    """The driver over a 4-day tm=D sim: a restart closes the orphaned
    invocation, then default schedule -> the agent's one-shot with its
    note -> crash_recovery -> a budget-dead firing that is not launched;
    one session throughout; the run reaches `done`."""
    from harness.episode_drive import EpisodeRunner

    cfg = ext_config(tmp_path / "sim", days=4, cell=dict(CELL_D), budget_usd=0.0)
    cfg.agent.scaffold = "ext:mcp"
    srv = Served(cfg, tmp_path / "sim", tmp_path / "sim" / "server")
    h = srv.handle
    ws = tmp_path / "workspace"
    ws.mkdir()
    (tmp_path / "fake_agent.py").write_text(FAKE_AGENT.format(
        url=h["env_url"], token=h["agent_token"], dir=str(tmp_path)))
    script = tmp_path / "firing.sh"
    script.write_text("printf '%s' \"$ENV_WAKE_MESSAGE\" | "  # as the real script
                      f"{sys.executable} {tmp_path / 'fake_agent.py'}\n")
    ep = {"env_url": h["env_url"], "token": h["token"], "model": None,
          "workspace": str(ws), "firing_script": str(script),
          "_path": str(tmp_path / "episode.json")}
    control = _Http(h["env_url"], h["token"])
    control.call("set_crontab", {"entries": [
        {"id": "act", "cron_expr": "0 0 * * *", "agent_owned": True}]})
    # a driver that died inside the bootstrap firing
    assert control.get("/trigger/next")["id"] == "__bootstrap__"

    assert EpisodeRunner(ep, quiet=True).run()["done"]
    srv.join()

    seen = [json.loads(l) for l in (tmp_path / "seen.jsonl").read_text().splitlines()]
    heads = [s["message"].splitlines()[0] for s in seen]
    assert heads == [
        "(woke at 2021-06-02T00:00:00+00:00 for trigger act/crash_recovery)",
        "(woke at 2021-06-02T06:00:00+00:00 for trigger look/at)",
        "(woke at 2021-06-03T00:00:00+00:00 for trigger act/crash_recovery)"]
    assert seen[0]["message"].endswith("follow it.") and seen[0]["session"] == ""
    assert "Your note for this schedule: look again" in seen[1]["message"]
    assert [s["session"] for s in seen[1:]] == ["ses_fake", "ses_fake"]
    assert "follow it." not in seen[2]["message"]

    firings = [json.loads(l) for l in (tmp_path / "firings.jsonl").read_text().splitlines()]
    assert [(f["trigger"]["id"], f["code"], f.get("skipped", False)) for f in firings] == \
        [("act", 0, False), ("look", 3, False), ("act", 0, False), ("act", 0, True)]
    assert list((ws / "logs").glob("crash-*.log"))  # the agent can read its crash
    exits = [e["outcome"] for e in ledger_view(tmp_path / "sim" / "server" / "ledger.jsonl")
             if e["type"] == "agent_exit"]
    assert exits == ["watchdog_killed", "exit=0", "exit=3", "exit=0", "exit=0"]
