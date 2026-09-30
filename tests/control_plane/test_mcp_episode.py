"""Episode mode: the stdio
MCP service exposes exactly the per-tm tool surface of a served ext:mcp
sim (tm=B: no `sleep`, a service-local fenced `run_program`), maps env
errors onto MCP error results verbatim, never leaks the token, and the
prepare/settle controller produces a normal run dir. The fenced executor
is parity-checked against runtime/program.py."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from harness.mcp import _Http, main as mcp_main
from tests.control_plane.test_detached import (
    Served,
    ext_config,
    ledger_view,
)
from tests.control_plane.test_launcher import REPO_ROOT

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")

CELL_A = {"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"}
CELL_B = {"tm": "B", "tlrn": "none", "sig": "none", "alg": "none"}



def is_error(result) -> bool:
    """CallToolResult.is_error (mcp >= 2) / .isError (mcp 1.x)."""
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


def input_schema(tool):
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)

def mcp_episode(tmp_path, cell, days=1, temps=None):
    """A served ext:mcp weather sim + its episode.json, no controller."""
    cfg = ext_config(tmp_path / "sim", days=days, temps=temps,
                     cell=dict(cell))
    cfg.agent.scaffold = "ext:mcp"
    server_dir = tmp_path / "sim" / "server"
    srv = Served(cfg, tmp_path / "sim", server_dir)
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    h = srv.handle
    ep = {"run_id": h["run_id"], "env_url": h["env_url"],
          "token": h["token"], "tm": cell["tm"], "task": "weather_fixture",
          "workspace": str(ws), "server_dir": str(server_dir),
          "clients": []}
    ep_path = tmp_path / "episode.json"
    ep_path.write_text(json.dumps(ep))
    http = _Http(h["env_url"], h["token"])
    trig = http.get("/trigger/next")  # controller consumes __bootstrap__
    return srv, http, ep_path, ws, trig


def settle(http, trig):
    """Controller-style close: exit + drive to done."""
    tid = trig.get("id")
    while True:
        if tid is not None:
            http.post(f"/trigger/{tid}/exit",
                      {"code": 0, "killed": False, "output": "test settle"})
        nxt = http.get("/trigger/next")
        if nxt.get("done"):
            return
        tid = nxt.get("id")


async def _session(ep_path, fn, waiter=None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "harness.mcp", "connect", "--episode", str(ep_path)]
             + (["--waiter", waiter] if waiter else []),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    # explicit errlog: stdio_client defaults to sys.stderr, which under
    # pytest's capsys is a python-level capture without fileno()
    errlog = open(os.devnull, "w")
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            return await fn(sess)


def run_session(ep_path, fn):
    return asyncio.run(_session(ep_path, fn))


def text_of(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


async def call_failing(sess, name: str, args: dict) -> str:
    """Call a tool that must fail and return the error text. mcp 1.x returns a
    result with isError set; mcp >= 2 raises MCPError client-side."""
    try:
        from mcp.shared.exceptions import MCPError
    except ImportError:  # mcp 1.x
        MCPError = ()  # noqa: N806
    try:
        result = await sess.call_tool(name, args)
    except MCPError as e:  # type: ignore[misc]
        return str(e)
    assert is_error(result), text_of(result)
    return text_of(result)


def payload_of(result) -> dict:
    assert not is_error(result), text_of(result)
    return json.loads(text_of(result))


# -- tool surface ---------------------------------------------------------------------


def test_tool_surface_tm_a(tmp_path):
    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_A)

    async def probe(sess):
        listed = await sess.list_tools()
        return {(t.name) : (t.description, input_schema(t)) for t in listed.tools}

    tools = run_session(ep_path, probe)
    assert set(tools) == {"get_time", "get_costs", "sleep",
                          "get_weather", "notify"}
    # docs verbatim + price suffix; schema is the served input_schema
    assert tools["sleep"][0].startswith("sleep(until: iso datetime)")
    assert tools["sleep"][1]["required"] == ["until"]
    assert tools["get_time"][1]["additionalProperties"] is False
    token = json.loads(ep_path.read_text())["token"]
    assert token not in json.dumps({k: v[0] for k, v in tools.items()})
    settle(http, trig)
    srv.join()


def test_tool_surface_tm_b_has_run_program_and_no_sleep(tmp_path):
    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_B)

    async def probe(sess):
        listed = await sess.list_tools()
        return {t.name: t.description for t in listed.tools}

    tools = run_session(ep_path, probe)
    assert set(tools) == {"get_time", "get_costs", "get_weather",
                          "notify", "run_program"}
    assert "sleep" not in tools
    # the run_program doc is the server-side @tool doc verbatim
    from harness.authored import ProgramApp

    assert tools["run_program"].startswith(
        vars(ProgramApp)["run_program"].__tool__["doc"][:40])
    settle(http, trig)
    srv.join()


# -- calls, errors, waits -------------------------------------------------------------


def test_calls_errors_and_sleep_to_end(tmp_path):
    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_A)

    async def act(sess):
        t = payload_of(await sess.call_tool("get_time", {}))
        assert t["now"] == "2021-06-01T00:00:00Z"
        # schema layer: the SDK refuses args that violate input_schema
        thin = await call_failing(sess, "get_weather", {"start": "nope"})
        # mcp 1.x validates against input_schema client-side ("required");
        # mcp >= 2 forwards the call and the env answers 400
        assert "required" in thin or "HTTP 400" in thin
        # env layer: schema-valid but semantically bad -> the env's 400
        bad = await call_failing(sess, "get_weather",
                                 {"start": "nope", "end": "nope"})
        assert "HTTP 400" in bad
        await call_failing(sess, "no_such_tool", {})
        wx = payload_of(await sess.call_tool(
            "get_weather", {"start": "2021-06-01T00:00:00Z",
                            "end": "2021-06-01T03:00:00Z"}))
        assert wx["hourly"]["time"] == []  # nothing observable at t0
        wake = payload_of(await sess.call_tool(
            "sleep", {"until": "2021-06-03T00:00:00Z"}))
        assert wake.get("experiment_over") or wake.get("woke_for")
        # keep sleeping until the run ends
        while not wake.get("experiment_over"):
            wake = payload_of(await sess.call_tool(
                "sleep", {"until": "2021-06-03T00:00:00Z"}))
        return wake

    wake = run_session(ep_path, act)
    token = json.loads(ep_path.read_text())["token"]
    assert token not in json.dumps(wake)
    settle(http, trig)
    res = srv.join()
    assert res["run_id"].startswith("test-run")
    events = [e["type"] for e in
              ledger_view(Path(json.loads(ep_path.read_text())
                               ["server_dir"]) / "ledger.jsonl")]
    assert "trigger" in events and "sleep" in events


# -- tm=B: fenced executor ------------------------------------------------------------

PROGRAM = """\
import envkit
r = envkit.get_weather(start="2021-06-01T00:00:00Z",
                       end="2021-06-01T06:00:00Z")
envkit.wait("2021-06-01T12:00:00Z")
r2 = envkit.get_weather(start="2021-06-01T00:00:00Z",
                        end="2021-06-01T06:00:00Z")
envkit.handover({"first": len(r["hourly"]["time"]),
                 "later": len(r2["hourly"]["time"])})
"""


def write_envkit(http, ws):
    (ws / "envkit.py").write_text(http.get("/contract")["envkit_py"])


def test_run_program_executor_and_parity(tmp_path, monkeypatch):
    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_B)
    write_envkit(http, ws)
    (ws / "watcher.py").write_text(PROGRAM)

    async def act(sess):
        ok = payload_of(await sess.call_tool(
            "run_program", {"path": "watcher.py", "validate": True}))
        assert ok["validate"] == "ok"
        out = payload_of(await sess.call_tool(
            "run_program", {"path": "watcher.py"}))
        outside = await call_failing(sess, "run_program",
                                     {"path": "../episode.json"})
        assert "outside the workspace" in outside
        return out

    out = run_session(ep_path, act)
    assert out["woke_for"] == "handover"
    assert out["payload"]["first"] == 0 and out["payload"]["later"] > 0
    assert out["fetches"] == 2 and out["waits"] == 1
    settle(http, trig)
    srv.join()
    sleeps_a = [e for e in ledger_view(
        Path(json.loads(ep_path.read_text())["server_dir"])
        / "ledger.jsonl") if e["type"] == "sleep"]

    # parity: the same program through runtime/program.py in-process on an
    # identical sim gives the same outcome and the same sleep events
    cfg = ext_config(tmp_path / "sim2", days=1, cell=dict(CELL_B))
    cfg.agent.scaffold = "ext:mcp"
    server2 = tmp_path / "sim2" / "server"
    srv2 = Served(cfg, tmp_path / "sim2", server2)
    h2 = srv2.handle
    http2 = _Http(h2["env_url"], h2["token"])
    trig2 = http2.get("/trigger/next")
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    write_envkit(http2, ws2)
    (ws2 / "watcher.py").write_text(PROGRAM)
    monkeypatch.setenv("ENV_URL", h2["env_url"])
    monkeypatch.setenv("ENV_TOKEN", h2["token"])
    from scaffolds.runtime.program import run_program

    direct = run_program(str(ws2 / "watcher.py"), root=ws2)
    assert {k: direct[k] for k in ("woke_for", "payload", "fetches",
                                   "waits")} \
        == {k: out[k] for k in ("woke_for", "payload", "fetches", "waits")}
    settle(http2, trig2)
    srv2.join()
    sleeps_b = [e for e in ledger_view(server2 / "ledger.jsonl")
                if e["type"] == "sleep"]
    assert sleeps_a == sleeps_b


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="fence built for macOS first")
def test_fence_blocks_non_env_network(tmp_path):
    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_B)
    write_envkit(http, ws)
    (ws / "leak.py").write_text(
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com/', timeout=5)\n"
        "import envkit\nenvkit.handover({'leaked': True})\n")

    async def act(sess):
        return payload_of(await sess.call_tool("run_program",
                                               {"path": "leak.py"}))

    out = run_session(ep_path, act)
    assert out["woke_for"] == "error"
    assert "URLError" in out["error"] or "Errno" in out["error"]
    settle(http, trig)
    srv.join()


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the fences are sandbox-exec (macOS)")
def test_fence_confines_signals():
    """A fenced process may signal its own children, nothing outside (an
    agent's `killall python` must not end the batch)."""
    from harness.mcp import fence_profile

    outside = subprocess.Popen(["sleep", "30"])
    try:
        out = subprocess.run(
            ["sandbox-exec", "-p", fence_profile([]), "/bin/sh", "-c",
             f"sleep 30 & kill $! && echo own-ok; kill {outside.pid} || echo refused"],
            capture_output=True, text=True).stdout.split()
        assert out == ["own-ok", "refused"] and outside.poll() is None
    finally:
        outside.kill()


def test_fence_cmd_skips_nesting_inside_actor_fence(monkeypatch):
    from harness.mcp import _fence_cmd

    monkeypatch.delenv("EPISODE_FENCE", raising=False)
    monkeypatch.setenv("EPISODE_OUTER_FENCE", "1")
    assert _fence_cmd(["python", "x.py"], ["http://127.0.0.1:1"]) == ["python", "x.py"]


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the fences are sandbox-exec (macOS)")
def test_run_program_inside_actor_fence(tmp_path):
    """The bridge under the actor fence: a nested sandbox is refused by
    macOS (every run_program would die `sandbox_apply: Operation not
    permitted`); with EPISODE_OUTER_FENCE
    the program runs unnested and the env is told (POST /program)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from harness.mcp import fence_profile

    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_B)
    write_envkit(http, ws)
    (ws / "hand.py").write_text("import envkit\nenvkit.handover({'ok': True})\n")
    ep = json.loads(ep_path.read_text())
    prof = fence_profile([ep["env_url"]], deny_read=[tmp_path / "sim"],
                         allow_read=[ws])

    async def run(extra_env):
        params = StdioServerParameters(
            command="sandbox-exec",
            args=["-p", prof, sys.executable, "-m", "harness.mcp", "connect",
                  "--episode", str(ep_path)],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT), **extra_env})
        errlog = open(os.devnull, "w")
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                res = await sess.call_tool("run_program", {"path": "hand.py"})
                return json.loads(text_of(res))

    nested = asyncio.run(run({}))
    assert nested["woke_for"] == "error" and "sandbox_apply" in nested["error"]
    ok = asyncio.run(run({"EPISODE_OUTER_FENCE": "1"}))
    assert ok["woke_for"] == "handover", ok
    settle(http, trig)
    srv.join()


# -- controller CLI -------------------------------------------------------------------


def weather_base_yaml(tmp_path, tm: str) -> Path:
    import yaml

    from tests.conftest import write_weather_csv
    from tests.control_plane.test_detached import START

    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 2)
    cfg = {"run_id": "episode-base",
           "task": {"name": "weather_fixture", "location": "LA",
                    "weather_csv": str(csv),
                    "data_cutoff": "2021-06-01T00:00:00Z",
                    "threshold_c": 33.0},
           "sim_start": "2021-06-01T00:00:00Z",
           "sim_end": "2021-06-03T00:00:00Z",
           "cell": {"tm": tm, "tlrn": "none", "sig": "none", "alg": "none"},
           "budget_usd": 5.0,
           "agent": {"scaffold": "react"}}
    p = tmp_path / f"base-tm{tm}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_prepare_act_settle_cli(tmp_path, capsys):
    base = weather_base_yaml(tmp_path, "B")
    out_root = tmp_path / "episodes"
    rc = mcp_main(["prepare", "--task", "weather_fixture", "--tm", "B",
                   "--base", str(base), "--out", str(out_root),
                   "--mock-llm"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    ws = Path(summary["workspace"])
    ep_path = Path(summary["episode"])
    instruction = (ws / "INSTRUCTION.md").read_text()
    assert "# Environment" in instruction  # the fixed appendix
    assert "`run_program`" in instruction  # tm=B wait tool named
    assert "high-temperature alerting" in instruction  # task spec rendered
    # the tm=B teaching kit (jail parity): skill appendix with the
    # jail tools renamed, sleep.py exemplar; weather ships no example
    assert "# Waiting by program" in instruction
    assert "`write_file`" not in instruction and "`edit_file`" not in instruction
    assert (ws / "sleep.py").read_text().startswith('"""Blind wait')
    assert not (ws / "example_gatekeeper.py").exists()
    envkit = (ws / "envkit.py").read_text()
    assert "run with the `run_program` tool" in envkit
    assert "runtime.program.run_program" not in envkit.split("def _bridge")[0]
    assert (ws / "envkit.py").exists()
    mcp_cfg = json.loads((ws / ".mcp.json").read_text())
    assert mcp_cfg["mcpServers"]["env"]["args"][:3] == \
        ["-m", "harness.mcp", "connect"]
    ep = json.loads(ep_path.read_text())
    assert Path(mcp_cfg["mcpServers"]["env"]["args"][-1]).is_absolute()
    # Codex: workspace-scoped project layer, never a global registration
    codex_cfg = (ws / ".codex" / "config.toml").read_text()
    assert "[mcp_servers.env]" in codex_cfg and str(ep_path) in codex_cfg
    assert "tool_timeout_sec = 86400" in codex_cfg
    assert 'default_tools_approval_mode = "approve"' in codex_cfg
    assert not (ep_path.parent / "codex_mcp.toml").exists()
    assert Path(ep["workspace"]).is_absolute()
    assert ep["trigger_id"] == "__bootstrap__"
    assert ep["llm_metered"] is False
    token = ep["token"]
    for f in ws.rglob("*"):
        if f.is_file():
            assert token not in f.read_text(), f  # token never in workspace

    # the actor acts through the registered service
    (ws / "watcher.py").write_text(PROGRAM)

    async def act(sess):
        return payload_of(await sess.call_tool("run_program",
                                               {"path": "watcher.py"}))

    out = run_session(ep_path, act)
    assert out["woke_for"] == "handover"

    rc = mcp_main(["settle", "--episode", str(ep_path)])
    assert rc == 0
    settled = json.loads(capsys.readouterr().out)
    assert settled["primary"]["name"]
    results = Path(settled["results"])
    assert results.exists()
    res = json.loads(results.read_text())
    assert res["config"]["agent"]["scaffold"] == "ext:mcp"
    events = [e["type"] for e in
              ledger_view(results.parent / "ledger.jsonl")]
    assert events.count("agent_exit") >= 1 and "trigger" in events
    # the recorded MCP client (the test's own SDK session)
    assert json.loads(ep_path.read_text())["clients"], "clientInfo recorded"



def test_render_envkit_default_unchanged():
    """The ext:/constructor envkit text is byte-identical to before the episode
    work: only episode mode passes a different `runner`."""
    from harness.contract import DEFAULT_RUNNER_REF, render_envkit

    manifest = [{"name": "get_weather", "doc": "get_weather(start: iso, end: "
                 "iso) -> x", "price": "free", "tags": []}]
    default = render_envkit(manifest)
    assert default == render_envkit(manifest, runner=DEFAULT_RUNNER_REF)
    assert "Import this from a program run via `runtime.program.run_program`; its" in default
    episode = render_envkit(manifest, runner="a program you run with the `run_program` tool")
    assert "Import this from a program you run with the `run_program` tool; its" in episode
    assert default.split("The contract:")[1] == episode.split("The contract:")[1]


def test_example_gatekeeper_rendering():
    """Task examples name the jail file tools; the episode copy names the
    agent's own (weather ships no example, so pin the renderer directly
    on a real task's file)."""
    from harness.mcp import render_example
    from harness.task import task_dir

    src = (task_dir("daily_reddit_digest") / "agent"
           / "example_gatekeeper.py").read_text()
    assert "(write_file / edit_file)" in src
    out = render_example(src)
    assert "(write_file / edit_file)" not in out
    assert "(with your file tools)" in out
    assert out.count("\n") == src.count("\n")  # a rename, nothing more


# -- web tasks + OpenCode ----------


def test_prepare_opencode_registration(tmp_path, capsys, monkeypatch):
    """`--app opencode --model` writes opencode.json: the env bridge from
    the environment with the DATA-PLANE token only, the model routed
    through /llm, bash allowed only under the fence; the controller token
    never enters the workspace; the launch line is fenced on macOS."""
    from harness.mcp import PLAYWRIGHT_PIN, fence_profile

    base = weather_base_yaml(tmp_path, "A")
    out_root = tmp_path / "episodes"
    rc = mcp_main(["prepare", "--task", "weather_fixture", "--tm", "A",
                   "--base", str(base), "--out", str(out_root),
                   "--app", "opencode", "--mock-llm",
                   "--model", "openai:gpt-5.6-luna"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    ws = Path(summary["workspace"])
    ep = json.loads(Path(summary["episode"]).read_text())
    assert ep["agent_token"] != ep["token"] and ep["app"] == "opencode"
    assert ep["llm_metered"] is True and ep["model"] == "gpt-5.6-luna"
    assert not (ws / ".mcp.json").exists() and not (ws / ".codex").exists()
    oc = json.loads((ws / "opencode.json").read_text())
    env = oc["mcp"]["env"]
    assert env["command"][-2:] == ["connect", "--from-env"]
    assert env["environment"]["ENV_TOKEN"] == ep["agent_token"]
    assert env["environment"]["ENV_URL"] == ep["env_url"]
    assert oc["mcp"]["playwright"]["enabled"] is False  # weather: no hosts
    pw = oc["mcp"]["playwright"]["command"]
    assert (f"@playwright/mcp@{PLAYWRIGHT_PIN}" in pw
            or (pw[0] == "node" and pw[1].endswith("cli.js")))
    assert "--isolated" in pw and "--headless" in pw
    prov = oc["provider"]["env"]
    assert prov["options"]["baseURL"] == ep["env_url"] + "/llm"
    assert prov["options"]["apiKey"] == ep["agent_token"]
    assert oc["model"] == "env/gpt-5.6-luna" and oc["enabled_providers"] == ["env"]
    # the window OpenCode compacts against (configs/model_limits.yaml)
    assert prov["models"]["gpt-5.6-luna"]["limit"] == {"context": 1050000,
                                                       "output": 128000}
    assert oc["compaction"] == {"auto": True}
    assert oc["permission"]["playwright_browser_evaluate"] == "deny"
    assert oc["permission"]["playwright_browser_run_code_unsafe"] == "deny"
    for f in ws.rglob("*"):
        if f.is_file():
            assert ep["token"] not in f.read_text(), f
    script = (Path(summary["episode"]).parent / "run_opencode.sh").read_text()
    assert "opencode' 'run'" in script or "opencode run" in script
    assert "OPENCODE_CONFIG=" in script and "-m' 'env/gpt-5.6-luna'" in script
    assert "SHELL=/bin/bash OPENCODE_CONFIG=" in script  # the bash tool IS bash
    assert f"XDG_DATA_HOME='{Path(summary['episode']).parent / 'opencode_data'}'" \
        in script  # a session store per episode: concurrent launches share no DB
    if platform.system() == "Darwin":
        assert "sandbox-exec -p" in script
        assert oc["permission"]["bash"] == "allow"
        ep_dir = Path(summary["episode"]).parent
        prof = fence_profile([ep["env_url"]], deny_read=[ep_dir],
                             allow_read=[ws, ep_dir / "opencode_data"],
                             allow_read_files=[ep_dir / "opencode_events.jsonl",
                                               ep_dir / "opencode_stderr.log"])
        assert prof in script
    else:
        assert oc["permission"]["bash"] == "deny"
    # the from-env bridge serves the tm=A surface with the agent token
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def probe():
        params = StdioServerParameters(
            command=sys.executable, args=env["command"][1:],
            env={**os.environ, **env["environment"]})
        errlog = open(os.devnull, "w")
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                listed = await sess.list_tools()
                t = payload_of(await sess.call_tool("get_time", {}))
                return {x.name for x in listed.tools}, t

    names, t = asyncio.run(probe())
    assert names == {"get_time", "get_costs", "sleep", "get_weather", "notify"}
    assert t["now"] == "2021-06-01T00:00:00Z"
    rc = mcp_main(["settle", "--episode", summary["episode"]])
    assert rc == 0


def test_prepare_web_task_requires_metered_llm(tmp_path, capsys):
    """A task with web hosts refuses to prepare without --model (design
    D7): no unmetered web episodes."""
    pytest.importorskip("tasks.edgar_portfolio.task")
    raw = REPO_ROOT / "tasks" / "edgar_portfolio" / "data" / "raw" / "manifest.json"
    if not raw.exists():
        pytest.skip("EDGAR snapshot not fetched")
    with pytest.raises(SystemExit) as ei:
        mcp_main(["prepare", "--task", "edgar_portfolio", "--tm", "A",
                  "--out", str(tmp_path / "episodes"), "--app", "opencode",
                  "--stretch", "3d"])
    assert "meter LLM usage" in str(ei.value)


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the fences are sandbox-exec (macOS)")
def test_prepare_web_task_splits_fences(tmp_path, capsys):
    """A fenced web task launches under two fences: the
    browser server's reaches every host and unix sockets; OpenCode's
    reaches the env, the read-only hosts and the browser server — never
    the writable host, so bash cannot curl the portal at all."""
    import re
    import urllib.parse

    pytest.importorskip("tasks.edgar_portfolio.task")
    raw = REPO_ROOT / "tasks" / "edgar_portfolio" / "data" / "raw" / "manifest.json"
    if not raw.exists():
        pytest.skip("EDGAR snapshot not fetched")
    rc = mcp_main(["prepare", "--task", "edgar_portfolio", "--tm", "A",
                   "--out", str(tmp_path / "episodes"), "--app", "opencode",
                   "--stretch", "3d", "--mock-llm",
                   "--model", "openai:gpt-5.6-luna"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    ws = Path(summary["workspace"])
    ep = json.loads(Path(summary["episode"]).read_text())
    sec, portal = ep["hosts"]["sec"], ep["hosts"]["portal"]
    pw_url = ep["playwright_url"]
    assert pw_url.startswith("http://localhost:") and pw_url.endswith("/mcp")
    oc = json.loads((ws / "opencode.json").read_text())
    assert oc["mcp"]["playwright"] == {"type": "remote", "url": pw_url,
                                       "enabled": True}
    assert oc["permission"]["bash"] == "allow"
    script = (Path(summary["episode"]).parent / "run_opencode.sh").read_text()
    pw_prof, oc_prof, again_prof = re.findall(r"sandbox-exec -p '([^']*)'", script)
    assert again_prof == oc_prof  # a re-prompted turn runs under the same fence

    def ports(prof):
        return set(re.findall(r'localhost:(\d+)', prof))

    def port(url):
        return str(urllib.parse.urlsplit(url).port)

    assert ports(pw_prof) == {port(sec), port(portal)}
    assert "(remote unix-socket)" in pw_prof and "--no-sandbox" in script
    assert ports(oc_prof) == {port(ep["env_url"]), port(sec), port(pw_url)}
    assert "(remote unix-socket)" not in oc_prof
    assert f"'--port' '{port(pw_url)}'" in script and "nc -z" in script
    # the actor's fence really admits the mirror and refuses the portal
    code = ("import sys, urllib.request\n"
            "for u in sys.argv[1:]:\n"
            "    try:\n        urllib.request.urlopen(u, timeout=5); print('OK')\n"
            "    except Exception as e:\n"
            "        print('DENIED' if 'not permitted' in str(e) else 'ERR ' + str(e))\n")
    out = subprocess.run(["sandbox-exec", "-p", oc_prof, sys.executable, "-c", code,
                          sec + "/files/company_tickers.json", portal + "/portfolio/"],
                         capture_output=True, text=True)
    assert out.stdout.split() == ["OK", "DENIED"], out.stdout + out.stderr[-300:]
    assert mcp_main(["settle", "--episode", summary["episode"]]) == 0


def test_playwright_command_headed_flag():
    """`--playwright-headed` only drops --headless: the fence restricts
    the network, not the display, so a visible Chrome is a launch option."""
    from harness.mcp import playwright_command

    hosts = {"a": "http://127.0.0.1:1", "b": "http://127.0.0.1:2"}
    default = playwright_command("0", "/x/cli.js", None, hosts, port=5)
    headed = playwright_command("0", "/x/cli.js", None, hosts, port=5, headed=True)
    assert "--headless" in default and "--headless" not in headed
    assert [a for a in default if a != "--headless"] == headed
    assert headed[:2] == ["node", "/x/cli.js"] and "--no-sandbox" in headed
    assert headed[headed.index("--allowed-origins") + 1] == "http://127.0.0.1:1;http://127.0.0.1:2"


def _mcp_http_navigate(port: int, url: str) -> str:
    """Minimal streamable-HTTP MCP client: initialize + browser_navigate."""
    import urllib.request

    sid = None

    def rpc(method, params, id_=None):
        nonlocal sid
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if id_ is not None:
            msg["id"] = id_
        hdr = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
        if sid:
            hdr["mcp-session-id"] = sid
        req = urllib.request.Request(f"http://localhost:{port}/mcp",
                                     data=json.dumps(msg).encode(), headers=hdr)
        with urllib.request.urlopen(req, timeout=60) as r:
            sid = r.headers.get("mcp-session-id") or sid
            body = r.read().decode()
        if id_ is None:
            return None
        datas = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
        return json.loads(datas[-1] if datas else body)

    rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}, 1)
    rpc("notifications/initialized", {})
    r = rpc("tools/call", {"name": "browser_navigate", "arguments": {"url": url}}, 2)
    return r["result"]["content"][0]["text"]


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the fences are sandbox-exec (macOS)")
def test_playwright_server_drives_chrome_inside_its_fence(tmp_path):
    """The browser server's fence (unix sockets allowed, Chrome's own
    sandbox off) lets the Playwright MCP navigate; the actor-style fence
    (no unix sockets) reproduces the first paid run's `connect EPERM` —
    the regression that silently turned a browser task into a curl one."""
    import http.server
    import shutil
    import socket
    import threading
    import time

    from harness.mcp import (PLAYWRIGHT_CLI_DEFAULT, _free_port, fence_profile,
                             playwright_command)

    if not PLAYWRIGHT_CLI_DEFAULT.exists() or shutil.which("node") is None:
        pytest.skip("@playwright/mcp not checked out (see README.md, Browser tasks)")
    if not Path("/Applications/Google Chrome.app").exists():
        pytest.skip("no system Chrome for the MCP's default channel")

    class Page(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = b"<h1>fenced page</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    page = f"http://127.0.0.1:{httpd.server_address[1]}"

    def drive(prof: str) -> str:
        port = _free_port()
        cmd = playwright_command("0", str(PLAYWRIGHT_CLI_DEFAULT), None,
                                 {"page": page}, port=port,
                                 output_dir=tmp_path / "out")
        log = open(tmp_path / f"pw-{port}.log", "w")
        proc = subprocess.Popen(
            ["sandbox-exec", "-p", prof] + cmd, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(tmp_path))
        try:
            for _ in range(150):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            return _mcp_http_navigate(port, page + "/")
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    text = drive(fence_profile([page], unix_sockets=True))
    assert f"Page URL: {page}/" in text, text[:300]
    text = drive(fence_profile([page]))
    assert "EPERM" in text, text[:300]
    httpd.shutdown()


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the actor fence is sandbox-exec (macOS)")
def test_actor_fence_hides_episode_dir_but_not_workspace(tmp_path):
    from harness.mcp import fence_profile

    ep_dir = tmp_path.resolve() / "episode"
    ws = ep_dir / "workspace"
    ws.mkdir(parents=True)
    (ep_dir / "episode.json").write_text('{"token": "secret"}')
    (ws / "INSTRUCTION.md").write_text("hello")
    prof = fence_profile(["http://127.0.0.1:1"], deny_read=[ep_dir], allow_read=[ws])
    code = ("import sys\n"
            "print(open(sys.argv[1]).read())\n"
            "try:\n    open(sys.argv[2]).read(); print('LEAK')\n"
            "except PermissionError: print('DENIED')\n")
    out = subprocess.run(["sandbox-exec", "-p", prof, sys.executable, "-c", code,
                          str(ws / "INSTRUCTION.md"), str(ep_dir / "episode.json")],
                         capture_output=True, text=True)
    assert out.stdout.split() == ["hello", "DENIED"], out.stderr[-300:]


def test_reprompt_loop_continues_the_session_until_stalled_or_over(
        tmp_path, monkeypatch):
    """run_opencode.sh's agent loop (tm=A/B): an app that exits without
    waiting is re-prompted in its session — stopped by the stall guard,
    and at once when the run is over."""
    from harness.mcp import REPROMPT_MAX_STALLED, _opencode_script

    srv, http, ep_path, ws, trig = mcp_episode(tmp_path, CELL_A)
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "bin" / "opencode"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n"
                    f"echo \"$* | $(cat)\" >> '{calls}'\n"
                    "echo '{\"sessionID\": \"ses_test\"}'\n"
                    "exit 1\n")  # as `opencode run` does when a turn ends in an API error
    fake.chmod(0o755)
    monkeypatch.setenv("EPISODE_FENCE", "off")
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    script = tmp_path / "run_opencode.sh"
    script.write_text(_opencode_script(
        ws, tmp_path, srv.handle, {}, {}, None, wait_tool="sleep"))

    def run():
        return subprocess.run(["sh", str(script)], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    done = run()
    # a decided stop is a settleable episode: the script exits 0 whatever
    # the app's last exit code was
    assert "stalled" in done.stderr and done.returncode == 0
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 + REPROMPT_MAX_STALLED
    assert "--session" not in lines[0]
    assert all("--session ses_test" in l and "wait tool (`sleep`)" in l
               for l in lines[1:])
    rows = [json.loads(l) for l in
            (tmp_path / "reprompts.jsonl").read_text().splitlines()]
    assert [r["action"] for r in rows] == \
        ["reprompt"] * REPROMPT_MAX_STALLED + ["stop"]

    http.call("sleep", {"until": srv.handle["sim_end"]})
    calls.unlink()
    done = run()
    assert "experiment_over" in done.stderr and done.returncode == 0
    assert len(calls.read_text().splitlines()) == 1  # launch, no re-prompt
    settle(http, trig)
    srv.join()


def test_model_limits_cover_every_priced_model():
    """Every model the cost table prices can also be launched under
    OpenCode: the limits table must have the same keys."""
    from harness.model_costs import (COSTS_PATH, LIMITS_PATH, parse_costs_file,
                                     parse_limits_file)

    missing = set(parse_costs_file(COSTS_PATH)) - set(parse_limits_file(LIMITS_PATH))
    assert not missing, f"configs/model_limits.yaml lacks {sorted(missing)}"


def test_opencode_config_refuses_model_without_limits(tmp_path, monkeypatch):
    from harness import mcp
    from harness import model_costs

    monkeypatch.setattr(model_costs, "resolve_limits", lambda m: None)
    handle = {"env_url": "http://127.0.0.1:1", "token": "t", "agent_token": "a",
              "run_id": "r"}
    with pytest.raises(SystemExit, match="model_limits.yaml"):
        mcp._opencode_json(sys.executable, tmp_path, handle, tm="A",
                           task="weather_fixture", hosts={}, hosts_readonly={},
                           model="nolimits-model", pin=mcp.PLAYWRIGHT_PIN,
                           fenced=False)
