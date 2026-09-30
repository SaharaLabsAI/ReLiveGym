"""Episode mode: external agent harnesses (Codex / Claude Code) acting in
a sim over MCP.

Two roles, five subcommands:

  prepare   the CONTROLLER: boots one `harness.serve` subprocess, builds
            the workspace (INSTRUCTION.md + appendix, envkit.py under
            tm=B) from GET /contract, registers the MCP service
            (.mcp.json for Claude Code, .codex/config.toml for Codex —
            both workspace-scoped, nothing global), consumes
            `__bootstrap__`, and
            writes episode.json (outside the workspace).
  connect   the registered stdio MCP SERVICE the app spawns: data-plane
            only — proxies the per-tm tool surface to POST /call/<name>,
            holds the token, and (tm=B) executes authored programs in a
            network-fenced subprocess (scaffolds/runtime/program_exec.py).
  drive     the CONTROLLER's cron driver (tm=C/D, harness/episode_drive.py):
            the runner loop of scaffolds/runtime/actor.py with the spawn
            replaced by one fenced `opencode run` per firing.
  status    GET /status of the episode.
  settle    close the open invocation and drive the sim to `done`
            (burning any remaining triggers), then report results.json.

The actor is never the experiment controller: `connect` launches nothing
and cannot end the run; lifecycle routes are prepare/settle-only. The env
bills and rate-limits every call server-side regardless of caller — the
service's filtering is the tm discipline (tm=B has no `sleep` in its tool
list), not the metering.

Web tasks + OpenCode:
`prepare --app opencode` also writes `opencode.json` — the env bridge
registered `connect --from-env` (URL + the DATA-PLANE token in the MCP
server's environment; episode.json, which holds the controller token,
never enters the workspace), Playwright MCP with the hosts as allowed
origins, and the model routed through the env's `/llm` proxy so tokens
are metered (D7, required for web tasks). `run_opencode.sh` launches the
actor under two OS fences: the Playwright MCP as a browser
server in a fence that reaches the web hosts, and OpenCode (with its
shell) in a fence that reaches the env, the read-only hosts and the
browser server only — so a writable host can be written from the
browser and from nothing else; no reads of the episode dir outside the
workspace; authored programs get the nested fence (env + read-only hosts).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tools never exposed to an external actor: lifecycle/plumbing surfaces
# (party + jail tools are program shapes this mode doesn't use). The schedule CRUD is the
# one exception: tm=D un-hides it (`_agent_tools`).
NOT_AGENT_TOOLS = frozenset({
    "set_party", "get_crontab", "set_crontab", "run_at",
    "list_schedules", "create_schedule", "update_schedule",
    "delete_schedule", "ls", "read_file", "write_file", "edit_file",
    "run_program"})
SCHEDULE_TOOLS = frozenset({"list_schedules", "create_schedule",
                            "update_schedule", "delete_schedule"})

# Episode-mode rendering of the tm=B skill appendix
# (harness/authored_skill.md): the same text, with the jail file tools
# (server-side WorkspaceApp, not mounted here) named as the agent's own
# file tools. The teaching kit — skill appendix,
# example_gatekeeper.py, sleep.py — is provisioned at workspace init
# together with envkit.py, as in every constructor-built tm=B jail.
SKILL_TOOL_SUBSTITUTIONS = (
    ("`edit_file` the `WAKE_AT` line", "edit the `WAKE_AT` line"),
    ("1. `read_file` `envkit.py` and `example_gatekeeper.py` first.",
     "1. Read `envkit.py` and `example_gatekeeper.py` first."),
    ("2. `write_file` / `edit_file` your own gatekeeper:",
     "2. Write your own gatekeeper (your file tools, in this workspace):"),
)
EPISODE_RUNNER_REF = "a program you run with the `run_program` tool"
# the task examples' docstrings name the jail tools too (4 of 6 tasks)
EXAMPLE_TOOL_SUBSTITUTIONS = (
    ("(write_file / edit_file)", "(with your file tools)"),
)


def render_example(text: str) -> str:
    for old, new in EXAMPLE_TOOL_SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


def render_skill_appendix() -> str:
    text = (REPO_ROOT / "harness" / "authored_skill.md").read_text()
    for old, new in SKILL_TOOL_SUBSTITUTIONS:
        assert old in text, f"authored_skill.md changed under us: {old!r}"
        text = text.replace(old, new)
    return text.rstrip()


# Fixed agent-visible appendix (plan Appendix B): wording changes are
# treatment changes.
INSTRUCTION_APPENDIX = """\
# Environment

Your tools for this task are the ones served by the `env` MCP server,
plus your own file tools in this workspace. Do not use any other tool,
command, or network access for the task — including looking up the
real-world date or time. Simulated time is frozen while you work and
passes only inside your wait tool (`{wait_tool}`). Every priced tool call
draws on your wallet of ${budget:g}. The experiment ends when a wait
returns experiment_over; then stop.
"""

# Web tasks (design replayed_web_server_v1 Appendix A): one added line
# when the run serves web hosts.
HOSTS_LINE = """\
The task's web addresses are the ones listed in the instruction; reach
them with your browser or shell as you normally would. Pages show the
simulated time they were rendered at, and anything loaded before a wait is
stale until fetched again.
"""

# Cron arms (tm=C/D).
# The appendix and the hosts line are the A/B texts with the wait clauses
# replaced; the schedule sections are the cron_react mains' texts with
# "call done" -> "end your turn" (an external app has no done tool).
CRON_INSTRUCTION_APPENDIX = """\
# Environment

Your tools for this task are the ones served by the `env` MCP server,
plus your own file tools in this workspace. Do not use any other tool,
command, or network access for the task — including looking up the
real-world date or time. Simulated time is frozen while you work and
passes only between your wakings. Every priced tool call draws on your
wallet of ${budget:g}. Nothing wakes you after the experiment ends.
"""

CRON_HOSTS_LINE = """\
The task's web addresses are the ones listed in the instruction; reach
them with your browser or shell as you normally would. Pages show the
simulated time they were rendered at, and anything loaded in an earlier
waking is stale until fetched again.
"""

SCHEDULE_SECTION = {
    "C": """\
# Your schedule
A fixed schedule wakes you on cron `{act_cron}` (UTC); you cannot wait or
schedule anything yourself, and time does not pass while you work. Each
waking: observe what you choose to, act if needed, then end your turn.
""",
    "D": """\
# Your schedule
A default schedule wakes you on cron `{act_cron}` (UTC); you may change
or remove it like any schedule of your own. Time does not pass while you
work: each waking runs at one simulated instant and ends when you end
your turn. You can also manage your own wake-up schedules with the
schedule API (list_schedules / create_schedule / update_schedule /
delete_schedule, all free): a one-time schedule fires once at its `at`
and is then removed; a recurring schedule fires on its cron expression;
when one of your schedules fires, you wake with its id and note.
Schedules never fire while you are awake, and none fire after the run
ends. The default schedule is already in place: do not create schedules
that duplicate it or each other; list_schedules shows what already
exists. Each waking: observe what you choose to, act if needed, then end
your turn.
"""}

# Party-episode strings (solo episodes never see either).
MARKET_SECTION = """\
# Your market

This run staffs one agent per market; you are the agent for market
{market_id}. Work only on this market and ignore the others. The other
agents share this environment and its budget.
"""

PARTY_RULE = """\
You are one of several agents sharing this environment. Simulated time
advances only when every agent is waiting, so always end your turn with
a wait — never stop or idle without one in flight.
"""


class EnvError(RuntimeError):
    """The env rejected a request; the message is the agent-facing text."""


class _Http:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"}

    def _req(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:
                detail = ""
            retry = e.headers.get("Retry-After") if e.headers else None
            msg = f"HTTP {e.code}: {detail}"
            if retry:
                msg += f" (Retry-After: {retry}s)"
            raise EnvError(msg) from None

    def get(self, path: str):
        return self._req("GET", path)

    def post(self, path: str, body: dict):
        return self._req("POST", path, body)

    def call(self, name: str, args: dict):
        return self._req("POST", f"/call/{name}", args or {})


def load_episode(path: Path) -> dict:
    ep = json.loads(Path(path).read_text())
    ep["_path"] = str(Path(path).resolve())
    return ep


def _episode_http(ep: dict) -> _Http:
    """Controller-side client (lifecycle routes)."""
    return _Http(ep["env_url"], ep["token"])


def _agent_http(ep: dict) -> _Http:
    """Data-plane client: the agent token when the sim issues one
    , else the run token (older episodes)."""
    return _Http(ep["env_url"], ep.get("agent_token") or ep["token"])


def episode_from_env() -> dict:
    """`connect --from-env`: the bridge's handle from the MCP server's
    environment (written by prepare into opencode.json) — the fenced
    actor cannot read episode.json."""
    try:
        return {"env_url": os.environ["ENV_URL"],
                "agent_token": os.environ["ENV_TOKEN"], "token": None,
                "tm": os.environ["ENV_TM"], "task": os.environ.get("ENV_TASK"),
                "workspace": os.environ["ENV_WORKSPACE"],
                "run_id": os.environ.get("ENV_RUN_ID"),
                "hosts": json.loads(os.environ.get("ENV_HOSTS") or "{}"),
                "hosts_readonly": json.loads(
                    os.environ.get("ENV_HOSTS_READONLY") or "{}"),
                "_path": None}
    except KeyError as e:
        raise SystemExit(f"connect --from-env: missing {e} in the environment")


# -- the fence (the one sandbox we build ourselves) ------------------------


def fence_profile(urls: list[str], deny_read: list[Path] = (),
                  allow_read: list[Path] = (),
                  allow_read_files: list[Path] = (), *,
                  unix_sockets: bool = False) -> str:
    """A sandbox-exec profile: outbound only to the loopback ports of
    `urls`; optionally no reads under `deny_read` except under the
    `allow_read` dirs and the `allow_read_files` (later rules win;
    subpath denial needs the REAL path, /private/tmp
    not /tmp; a process's own redirected stdio inside a denied dir
    breaks Bun's fstat, hence the file allowances). `unix_sockets`
    re-allows local unix-socket connects, which the blanket outbound
    denial also covers: the Playwright MCP reaches its browser over one
    (without it every browser call fails `connect
    EPERM .../pw-*/browser/*.sock` and the actor falls back to curl)."""
    ports = sorted({urllib.parse.urlsplit(u).port for u in urls if u})
    prof = "(version 1)(allow default)(deny network-outbound)"
    # signals stay inside the fence: the actor may kill its own processes,
    # nothing else of this user's: an actor's bash command carrying
    # `killall python` would otherwise SIGTERM the batch runner, every
    # driver and every sim server
    prof += "(deny signal)(allow signal (target same-sandbox))"
    if unix_sockets:
        prof += "(allow network-outbound (remote unix-socket))"
    for p in ports:
        prof += f'(allow network-outbound (remote ip "localhost:{p}"))'
    for d in deny_read:
        real = Path(d).resolve()
        prof += f'(deny file-read* (subpath "{real}"))'
        # stat of the denied dir ITSELF stays allowed (no listing, no
        # contents): realpath(3) walks every component, and node's
        # fs.realpath under the browser server died `EPERM` on the
        # episode dir when writing a snapshot into the workspace below it
        prof += f'(allow file-read-metadata (literal "{real}"))'
    for d in allow_read:
        prof += f'(allow file-read* (subpath "{Path(d).resolve()}"))'
    for f in allow_read_files:
        f = Path(f)
        prof += f'(allow file-read* (literal "{f.parent.resolve() / f.name}"))'
    return prof


def _fence_cmd(argv: list[str], urls: list[str] | str,
               deny_read: list[Path] = (), allow_read: list[Path] = ()) -> list[str]:
    """Wrap `argv` so its network reaches ONLY the given loopback ports
    (the env).
    macOS: sandbox-exec (port-scoped localhost allow; external DNS and
    other local ports refused). Linux: bwrap fence not built — refuse
    rather than run agent
    code unfenced; EPISODE_FENCE=off overrides explicitly."""
    if os.environ.get("EPISODE_FENCE") == "off":
        return argv
    if os.environ.get("EPISODE_OUTER_FENCE") == "1":
        # the bridge already runs inside the actor fence:
        # macOS refuses a nested sandbox_apply ("Operation not permitted"), so
        # the outer fence bounds the program. Its one extra reach — the
        # browser server — is closed server-side: writable hosts refuse
        # writes while a program is marked running (POST /program).
        return argv
    if isinstance(urls, str):
        urls = [urls]
    if platform.system() == "Darwin":
        return ["sandbox-exec", "-p",
                fence_profile(urls, deny_read, allow_read)] + argv
    raise EnvError(
        "no network fence available on this platform for authored "
        "programs (Linux bwrap is later work). Set "
        "EPISODE_FENCE=off to run unfenced at your own risk.")


def _run_program_subprocess(ep: dict, arguments: dict) -> dict:
    """Execute one authored program with runtime/program.py semantics in
    a fenced subprocess; liveness = the run's real-time watchdog."""
    ws = Path(ep["workspace"]).resolve()
    rel = arguments.get("path")
    if not isinstance(rel, str) or not rel:
        raise EnvError("run_program needs a string 'path'")
    file = (ws / rel).resolve()
    if not str(file).startswith(str(ws) + os.sep):
        raise EnvError(f"path {rel!r} is outside the workspace")
    if not file.is_file():
        raise EnvError(f"no such program: {rel!r}")
    exec_py = REPO_ROOT / "scaffolds" / "runtime" / "program_exec.py"
    # the outcome file lives in the workspace: under the actor fence the
    # episode dir is unreadable to the bridge
    out_dir = ws / ".run_program"
    out_dir.mkdir(exist_ok=True)
    out = tempfile.NamedTemporaryFile(
        dir=out_dir, prefix="run_program-", suffix=".json", delete=False)
    out.close()
    argv = [sys.executable, "-I", str(exec_py), "--file", str(file),
            "--root", str(ws), "--out", out.name]
    if arguments.get("until"):
        argv += ["--until", str(arguments["until"])]
    if arguments.get("validate"):
        argv += ["--validate"]
    if arguments.get("waiter_id"):
        argv += ["--waiter-id", str(arguments["waiter_id"])]
    env = {"PATH": os.environ.get("PATH", ""),
           "HOME": os.environ.get("HOME", ""),
           "ENV_URL": ep["env_url"],
           "ENV_TOKEN": ep.get("agent_token") or ep["token"]}
    reach = [ep["env_url"]] + list((ep.get("hosts_readonly") or {}).values())
    http = _agent_http(ep)
    _program_mark(http, True)
    proc = subprocess.Popen(_fence_cmd(argv, reach), env=env,
                            cwd=str(ws), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        while proc.poll() is None:
            time.sleep(1.0)
            try:  # the run's real-time watchdog, applied by the service
                act = http.get("/activity")
                if (act["idle_seconds"] > act["watchdog_seconds"]
                        and not act.get("llm_inflight")):
                    proc.kill()
                    proc.wait(timeout=10)
                    return {"woke_for": "error",
                            "error": "program killed: no environment call "
                                     f"for {act['idle_seconds']:.0f}s (the "
                                     "run's real-time watchdog)"}
            except EnvError:
                pass  # lifecycle route hiccup never kills the program
        stderr = (proc.stderr.read() or b"").decode(errors="replace")
        try:
            return json.loads(Path(out.name).read_text())
        except Exception:
            return {"woke_for": "error",
                    "error": "program subprocess exited "
                             f"{proc.returncode} without an outcome"
                             + (f": {stderr[-400:]}" if stderr.strip() else "")}
    finally:
        Path(out.name).unlink(missing_ok=True)
        _program_mark(http, False)


def _program_mark(http: _Http, active: bool) -> None:
    """Tell the env a program run starts/ends (POST /program): writable
    hosts refuse writes meanwhile (harness/web.py) — the program-fence
    rule kept when the program cannot be fenced tighter
    than the actor. Older envs without the route are tolerated."""
    try:
        http.post("/program", {"active": active})
    except EnvError:
        pass


# -- connect: the stdio MCP service ---------------------------------------------------


def _agent_tools(manifest: list[dict], tm: str) -> list[dict]:
    """The per-tm surface: `sleep` is tm=A's wait tool only (B waits by
    program; a cron-fired agent, C/D, holds no wait tool at all), and the
    schedule CRUD is tm=D's."""
    hidden = set(NOT_AGENT_TOOLS)
    if tm != "A":
        hidden.add("sleep")
    if tm == "D":
        hidden -= SCHEDULE_TOOLS
    return [t for t in manifest if t["name"] not in hidden]


def _local_run_program_tooldef() -> dict:
    """The service-local run_program tool: doc/price/schema verbatim from
    the server-side @tool declaration (plan Appendix B)."""
    from harness.authored import ProgramApp

    meta = vars(ProgramApp)["run_program"].__tool__
    return {"name": "run_program", "doc": meta["doc"],
            "price": meta["price"], "input_schema": meta["schema"]}


def _record_client(ep: dict, info) -> None:
    if not ep.get("_path"):
        return  # --from-env: no episode.json to annotate (fenced actor)
    try:
        path = Path(ep["_path"])
        data = json.loads(path.read_text())
        entry = {"name": getattr(info, "name", None),
                 "version": getattr(info, "version", None),
                 "connected_at": datetime.now(timezone.utc).isoformat()}
        if entry["name"] and entry not in [
                {k: c.get(k) for k in ("name", "version", "connected_at")}
                for c in data.get("clients", [])]:
            data.setdefault("clients", []).append(entry)
            path.write_text(json.dumps(data, indent=1))
    except Exception:
        pass  # bookkeeping only — never fail a tool call over it


async def _serve_connect(ep: dict) -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    http = _agent_http(ep)
    manifest = http.get("/tools")["tools"]
    tools = _agent_tools(manifest, ep["tm"])
    if ep["tm"] == "B":
        tools = tools + [_local_run_program_tooldef()]
    by_name = {t["name"]: t for t in tools}

    async def _list(context, params=None) -> types.ListToolsResult:
        mcp_tools = [types.Tool(
            name=t["name"],
            description=f"{t['doc']} [{t.get('price', 'free')}]",
            inputSchema=t.get("input_schema")
            or {"type": "object", "additionalProperties": True},
        ) for t in tools]
        return types.ListToolsResult(tools=mcp_tools)

    async def _call(context, params) -> types.CallToolResult:
        name = params.name
        arguments = params.arguments
        try:
            client_info = None
            if hasattr(context, "session") and hasattr(context.session, "client_params"):
                client_info = getattr(context.session.client_params, "client_info", getattr(context.session.client_params, "clientInfo", None))
            _record_client(ep, client_info)
        except Exception:
            pass
        if name not in by_name:
            raise EnvError(f"no such tool: {name!r}")
        if ep.get("waiter") and name in ("sleep", "run_program"):
            # party episode: the bridge owns waiter
            # identity — an agent-supplied waiter_id is overridden, so
            # impersonating another waiter is impossible
            arguments = {**(arguments or {}), "waiter_id": ep["waiter"]}
        import asyncio

        if name == "run_program" and ep["tm"] == "B":
            result = await asyncio.to_thread(
                _run_program_subprocess, ep, arguments or {})
        else:
            result = await asyncio.to_thread(
                http.call, name, arguments or {})
        return types.CallToolResult(content=[types.TextContent(type="text",
                                  text=json.dumps(result, default=str))])

    server = Server("env", on_list_tools=_list, on_call_tool=_call)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def cmd_connect(args) -> int:
    import asyncio

    if getattr(args, "from_env", False):
        asyncio.run(_serve_connect(episode_from_env()))
        return 0
    ep = load_episode(Path(args.episode))
    if getattr(args, "waiter", None):
        wss = ep.get("waiters") or {}
        if args.waiter not in wss:
            raise SystemExit(
                f"unknown waiter {args.waiter!r} (episode declares: "
                f"{sorted(wss) or 'none'})")
        ep = {**ep, "workspace": wss[args.waiter], "waiter": args.waiter}
    asyncio.run(_serve_connect(ep))
    return 0


# -- prepare: the controller ----------------------------------------------------------


def _resolve_base(task: str, tm: str, base: str | None) -> Path:
    if base:
        return Path(base)
    root = REPO_ROOT / "tasks" / task / "configs" / "cells"
    hits = sorted(root.rglob(f"*tm{tm}-tlrnnone-signone-algnone.yaml"))
    if not hits:
        raise SystemExit(
            f"no algnone tm{tm} base yaml under {root} — pass --base")
    return hits[0]


def _wait_for(path: Path, proc: subprocess.Popen, seconds: float,
              what: str) -> None:
    t0 = time.monotonic()
    while not path.exists():
        if proc.poll() is not None:
            raise SystemExit(f"harness.serve exited {proc.returncode} "
                             f"before writing {what} (see serve.log)")
        if time.monotonic() - t0 > seconds:
            raise SystemExit(f"timed out waiting for {what}")
        time.sleep(0.2)


def _connect_args(episode_json: Path, waiter: str | None) -> list[str]:
    args = ["-m", "harness.mcp", "connect", "--episode", str(episode_json)]
    if waiter:
        args += ["--waiter", waiter]
    return args


def _mcp_json(python: str, episode_json: Path,
              waiter: str | None = None) -> dict:
    return {"mcpServers": {"env": {
        "command": python,
        "args": _connect_args(episode_json, waiter),
        "env": {"PYTHONPATH": str(REPO_ROOT)}}}}


def _codex_config_toml(python: str, episode_json: Path,
                       waiter: str | None = None) -> str:
    # Codex project layer: with the workspace as the session cwd Codex
    # loads <cwd>/.codex/config.toml, trust inherited from the enclosing
    # trusted root (this repo) — so no global `codex mcp add`, and the
    # registration dies with the episode (codex 0.140.0: `codex mcp get
    # env --json` from such a dir reports the server with both timeouts;
    # from the repo root it is absent).
    # tool_timeout_sec: waits may legitimately block for hours.
    # default_tools_approval_mode: Approve is the ONLY mode that
    # short-circuits the MCP permission prompt — without it headless
    # `codex exec` auto-aborts every call as "user cancelled MCP tool
    # call" (codex-mcp/src/mcp/mod.rs); approval_policy="never" alone is
    # not enough under the managed workspace-write profile.
    args = _connect_args(episode_json, waiter)
    return ("# Written by `harness.mcp prepare` for this workspace.\n"
            "[mcp_servers.env]\n"
            f'command = "{python}"\n'
            f"args = {json.dumps(args)}\n"
            f'env = {{ PYTHONPATH = "{REPO_ROOT}" }}\n'
            "startup_timeout_sec = 60\n"
            "tool_timeout_sec = 86400\n"
            'default_tools_approval_mode = "approve"\n')


PLAYWRIGHT_PIN = "0.0.81"  # the @playwright/mcp version the runs used
# a checked-out CLI beats `npx`: OpenCode's bundled npm crashes on
# `process.stderr.isTTY` when the actor's stdio is redirected to files
# (Bun 1.3.14), and npx cannot fetch inside the fence
PLAYWRIGHT_CLI_DEFAULT = (REPO_ROOT / ".tools" / "playwright-mcp" / "node_modules"
                          / "@playwright" / "mcp" / "cli.js")
ACTOR_PROMPT = "Read INSTRUCTION.md in this workspace and follow it."

# The agent loop of a tm=A/B OpenCode episode (parity with the constructor's
# actor: `while
# actor.turn()` never lets a reply without a tool call end the run — it
# answers with a format reminder at the same sim instant). `opencode run`
# exits when the model stops calling tools, so run_opencode.sh continues
# the session with this message until `reprompt` says stop (a run would
# otherwise end with a prose "currently waiting until ..." and no wait
# call in flight).
REPROMPT_MESSAGE = (
    "(your turn ended without a tool call; the experiment is not over. "
    "Simulated time passes only inside your wait tool (`{wait_tool}`) — "
    "continue until a wait returns experiment_over.)")
# consecutive re-prompts at one sim instant before the loop gives up (the
# runaway guard; the constructor's actor has max_calls_per_wake)
REPROMPT_MAX_STALLED = 5


def playwright_command(pin: str, cli: str | None, browser: str | None,
                       hosts: dict, *, port: int | None = None,
                       output_dir: Path | None = None,
                       headed: bool = False) -> list[str]:
    """The Playwright MCP launch: `node <cli.js>` when a checkout exists
    (PLAYWRIGHT_MCP_CLI, --playwright-cli, or .tools/playwright-mcp —
    `npm install --prefix .tools/playwright-mcp @playwright/mcp@<pin>`),
    else `npx -y @playwright/mcp@<pin>`. Isolated, headless unless
    `headed` (a visible Chrome window to watch the agent act — the fence
    restricts the network, not the display), the
    hosts as the allowed origins, optionally a browser channel (chrome,
    msedge…).
    With `port` it is the fenced browser SERVER (streamable
    HTTP at http://localhost:<port>/mcp — the server admits only that
    Host; OpenCode connects to it as a remote MCP) with Chrome's own
    process sandbox off: nested inside sandbox-exec the renderer crashes
    (`Target crashed`), and the OS fence is the
    boundary anyway."""
    cli = cli or os.environ.get("PLAYWRIGHT_MCP_CLI") or (
        str(PLAYWRIGHT_CLI_DEFAULT) if PLAYWRIGHT_CLI_DEFAULT.exists() else None)
    cmd = ["node", cli] if cli else ["npx", "-y", f"@playwright/mcp@{pin}"]
    cmd += ["--isolated", "--allowed-origins", ";".join(hosts.values())]
    if not headed:
        cmd.append("--headless")
    if browser:
        cmd += ["--browser", browser]
    if port is not None:
        cmd += ["--port", str(port), "--host", "127.0.0.1", "--no-sandbox"]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    return cmd


def playwright_url(port: int) -> str:
    return f"http://localhost:{port}/mcp"


def _free_port() -> int:
    """An ephemeral loopback port for the browser server (chosen at
    prepare, bound at launch — the same small race the hosts accept)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _opencode_json(python: str, ws: Path, handle: dict, *, tm: str, task: str,
                   hosts: dict, hosts_readonly: dict, model: str | None,
                   pin: str, fenced: bool, playwright_cli: str | None = None,
                   browser: str | None = None,
                   playwright_port: int | None = None,
                   headed: bool = False) -> dict:
    """The per-workspace OpenCode config: env bridge from
    the environment (data-plane token only), Playwright with the hosts as
    allowed origins — a remote MCP at the fenced browser server when
    `playwright_port` is set , else spawned over stdio — the
    model routed through /llm, and the tool policy — bash allowed under
    the OS fence, denied where no fence exists."""
    env = {"PYTHONPATH": str(REPO_ROOT), "ENV_URL": handle["env_url"],
           "ENV_TOKEN": handle.get("agent_token") or handle["token"],
           "ENV_TM": tm, "ENV_TASK": task, "ENV_WORKSPACE": str(ws),
           "ENV_RUN_ID": handle["run_id"], "ENV_HOSTS": json.dumps(hosts),
           "ENV_HOSTS_READONLY": json.dumps(hosts_readonly)}
    if fenced:
        env["EPISODE_OUTER_FENCE"] = "1"  # no nested sandbox in run_program
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "env": {"type": "local",
                    "command": [python, "-m", "harness.mcp", "connect", "--from-env"],
                    "environment": env, "enabled": True,
                    "timeout": 86_400_000},  # ms: party waits / programs block
            "playwright": ({"type": "remote",
                            "url": playwright_url(playwright_port),
                            "enabled": bool(hosts)}
                           if playwright_port is not None else
                           {"type": "local",
                            "command": playwright_command(pin, playwright_cli,
                                                          browser, hosts,
                                                          headed=headed),
                            "enabled": bool(hosts)}),
        },
        "permission": {"*": "deny", "env_*": "allow", "playwright_*": "allow",
                       # the JS escape hatches: arbitrary page
                       # or Playwright code turns the browser into curl
                       # (scripted form submits and mirror fetches through
                       # run_code_unsafe)
                       "playwright_browser_evaluate": "deny",
                       "playwright_browser_run_code_unsafe": "deny",
                       "read": "allow", "edit": "allow", "glob": "allow",
                       "grep": "allow", "list": "allow",
                       "bash": "allow" if fenced else "deny",
                       "webfetch": "deny", "websearch": "deny"},
        "share": "disabled",
        "autoupdate": False,
    }
    if model:
        cfg["provider"] = {"env": {
            "npm": "@ai-sdk/openai-compatible", "name": "env proxy",
            "options": {"baseURL": handle["env_url"] + "/llm",
                        "apiKey": handle.get("agent_token") or handle["token"]},
            "models": {model: {"name": model, "tool_call": True,
                               # OpenCode compacts only when the model
                               # declares a window (a custom provider
                               # starts at 0 = never): configs/
                               # model_limits.yaml, refused if absent
                               "limit": _model_limit(model)}}}}
        cfg["enabled_providers"] = ["env"]
        cfg["model"] = f"env/{model}"
        cfg["small_model"] = f"env/{model}"
        # auto-compaction on (OpenCode's default, stated so a later
        # default flip cannot change the actor); prune left at its
        # default (false): old tool outputs stay verbatim until compacted
        cfg["compaction"] = {"auto": True}
    return cfg


def _model_limit(model: str) -> dict[str, int]:
    from harness.model_costs import LIMITS_PATH, resolve_limits

    lim = resolve_limits(model)
    if lim is None:
        raise SystemExit(f"model {model!r} has no entry in {LIMITS_PATH} — "
                         "OpenCode cannot compact without its context "
                         "window; add context/output before launching")
    return {"context": lim["context"], "output": lim["output"]}


def _q(arg) -> str:
    """Single-quote one shell word (absolute paths, profiles)."""
    return "'" + str(arg).replace("'", "'\\''") + "'"


def _opencode_script(ws: Path, episode_dir: Path, handle: dict, hosts: dict,
                     hosts_readonly: dict, model: str | None, *,
                     playwright: list[str] | None = None,
                     playwright_port: int | None = None,
                     cron: bool = False, wait_tool: str = "sleep") -> str:
    """run_opencode.sh — the actor launch. Fenced (macOS):
    two sandbox-exec fences. The browser server (Playwright MCP over
    HTTP) reaches every web host and its browser's unix socket; OpenCode
    — and so its shell — reaches the env, the READ-ONLY hosts and the
    browser server only. A writable host is therefore unreachable from
    bash by construction: the browser is the sole write path (an actor
    whose browser breaks would otherwise curl every form). No
    hosts (a tool-only task): one fence, no browser server. Unfenced:
    the plain launch, Playwright over stdio, bash denied in opencode.json.
    Absolute paths; the episode dir unreadable except the workspace and
    the actor's own log files.

    `cron` writes the per-firing launch of the cron arms
    (run_opencode_firing.sh, run by harness/episode_drive.py): the same
    launch, prompted with $ENV_WAKE_MESSAGE, continuing the session
    $ENV_OPENCODE_SESSION when set, logs appended — the fence admits the
    actor's stdio files by literal path, so they cannot vary per firing.
    Both variables are expanded inside double quotes (an agent-authored
    schedule note never passes through shell parsing), and the message
    goes in on stdin: `opencode run` wraps a multi-word argument in
    literal quotes and backslash-escapes the quotes inside it.

    tm=A/B (not `cron`): the launch is followed by the agent loop — while
    `harness.mcp reprompt` (controller side, outside the fences) prints
    the session id, the same launch continues that session with
    REPROMPT_MESSAGE on stdin, logs appended. The browser server stays up
    across re-prompts, so the portal login survives them."""
    argv = ["opencode", "run", "--dir", str(ws)]
    if model:
        argv += ["-m", f"env/{model}"]
    argv += ["--format", "json"]
    prompt = ('${ENV_OPENCODE_SESSION:+--session "$ENV_OPENCODE_SESSION"}'
              if cron else _q(ACTOR_PROMPT))
    to = ">>" if cron else ">"
    events = episode_dir / "opencode_events.jsonl"
    stderr = episode_dir / "opencode_stderr.log"
    # SHELL pins the `bash` tool to bash: OpenCode resolves the tool's
    # shell from $SHELL, and under zsh (the macOS default) bash idioms
    # such as `set -- $pair` do not word-split, which silently malforms
    # curl URLs built that way
    # XDG_DATA_HOME: OpenCode keeps sessions in ONE SQLite file under its
    # data dir; episodes launched together die at startup with `database
    # is locked`, and a tm=C/D episode starts a
    # process per firing. Each episode gets its own — which also keeps the
    # session next to the run (`XDG_DATA_HOME=<episode>/opencode_data
    # opencode export <session>`) and other episodes' sessions out of reach.
    data_home = episode_dir / "opencode_data"
    head = (f"SHELL=/bin/bash OPENCODE_CONFIG={_q(ws / 'opencode.json')} "
            f"XDG_DATA_HOME={_q(data_home)} ")
    if cron:
        head = "printf '%s' \"$ENV_WAKE_MESSAGE\" | " + head
    run = ' '.join(_q(a) for a in argv)
    tail = f"{run} {prompt} {to} {_q(events)} 2{to} {_q(stderr)}"
    again = f'{run} --session "$SID" >> {_q(events)} 2>> {_q(stderr)}'

    def loop(fence: str) -> list[str]:
        if cron:
            return []
        msg = REPROMPT_MESSAGE.format(wait_tool=wait_tool)
        return ["# the agent loop: a turn that ends without a tool call is "
                "re-prompted until the run is over",
                "while :; do",
                f"  SID=$(cd {_q(REPO_ROOT)} && {_q(sys.executable)} -m "
                f"harness.mcp reprompt --episode {_q(episode_dir / 'episode.json')})",
                "  RC=$?",
                '  [ "$RC" -eq 0 ] || break',
                f"  printf '%s' {_q(msg)} | {head}{fence}{again}",
                "done",
                # the script's status is the loop's verdict, not the last
                # `opencode run`'s: that one exits 1 when its turn ends in
                # an API error (a spent budget, a provider 400), and the
                # batch runner reads non-zero as a broken launch and never
                # settles
                '[ "$RC" -eq 1 ]']
    fenced = platform.system() == "Darwin" and os.environ.get("EPISODE_FENCE") != "off"
    lines = ["#!/bin/sh",
             "# written by harness.mcp prepare — the actor launch "]
    if not fenced:
        lines += [f"cd {_q(ws)} || exit 1", head + tail] + loop("")
        return "\n".join(lines) + "\n"
    reach = [handle["env_url"]] + list(hosts_readonly.values())
    lines.append(f"cd {_q(ws)} || exit 1")
    if playwright and playwright_port is not None:
        pw_log = episode_dir / "playwright_mcp.log"
        pw_prof = fence_profile(list(hosts.values()), deny_read=[episode_dir],
                                allow_read=[ws], allow_read_files=[pw_log],
                                unix_sockets=True)
        lines += [
            "# 1. the browser server: its fence reaches the web hosts (all of "
            "them) and the browser's unix socket, nothing else",
            f"sandbox-exec -p {_q(pw_prof)} {' '.join(_q(a) for a in playwright)}"
            f" {to} {_q(pw_log)} 2>&1 &",
            "PW=$!",
            # a signalled script must EXIT, not fall through into the agent loop.
            # sandbox-exec forks node: kill its children too, or the browser
            # server outlives the episode
            "trap 'pkill -P $PW 2>/dev/null; kill $PW 2>/dev/null' EXIT",
            "trap 'exit 143' INT TERM",
            "i=0",
            f"until nc -z 127.0.0.1 {playwright_port} 2>/dev/null; do",
            "  i=$((i+1))",
            "  if [ $i -ge 150 ]; then echo 'playwright MCP did not start "
            "(see playwright_mcp.log)' >&2; exit 1; fi",
            "  sleep 0.2",
            "done",
            "# 2. the actor: its fence reaches the env, the read-only hosts and "
            "the browser server — never a writable host"]
        reach.append(playwright_url(playwright_port))
    prof = fence_profile(reach, deny_read=[episode_dir],
                         allow_read=[ws, data_home],
                         allow_read_files=[events, stderr])
    lines.append(head + f"sandbox-exec -p {_q(prof)} " + tail)
    lines += loop(f"sandbox-exec -p {_q(prof)} ")
    return "\n".join(lines) + "\n"


def _party_spec(args, raw: dict) -> list[tuple[str, str | None]] | None:
    """Party episodes: [(waiter_id, market_id-or-None)], or
    None for a solo episode. --per-market derives one waiter per entry
    of the base config's task.markets roster; --party takes explicit
    waiter ids (any task; no market scoping in the INSTRUCTION)."""
    per_market = getattr(args, "per_market", False)
    party = getattr(args, "party", None)
    if per_market and party:
        raise SystemExit("--per-market and --party are mutually exclusive")
    if per_market:
        markets = (raw.get("task") or {}).get("markets") or []
        ids = [str((m or {}).get("market_id") or "") for m in markets]
        if not ids or "" in ids:
            raise SystemExit("--per-market: the base config has no "
                             "task.markets roster")
        return [(f"m{i}", i) for i in ids]
    if party:
        ids = [w.strip() for w in party.split(",") if w.strip()]
        if len(ids) < 2:
            raise SystemExit("--party wants >=2 comma-separated waiter ids")
        if len(set(ids)) != len(ids):
            raise SystemExit("--party: duplicate waiter ids")
        return [(w, None) for w in ids]
    return None


def cmd_watchdog(args) -> int:
    """Party stall watchdog. A session that ends its
    turn without waiting parks the barrier for everyone, silently; while
    at least one waiter IS parked, a roster member that stays un-parked
    for --stall-seconds of real time is evicted (set_party minus the
    stalled ids — the server refuses to drop a PARKED waiter, so an
    agent that is actually waiting can never be evicted) and the episode
    is flagged in episode.json. Any sim-time progress resets all timers.
    Exits when the run is done or the server is gone."""
    ep_path = Path(args.episode)
    ep = load_episode(ep_path)
    http = _episode_http(ep)
    timers: dict[str, float] = {}
    last_sim_now = None
    errors = 0
    while True:
        time.sleep(args.poll_seconds)
        try:
            st = http.get("/status")
            errors = 0
        except (EnvError, OSError):
            errors += 1
            if errors >= 6:  # settled or crashed — nothing left to watch
                return 0
            continue
        if st.get("done"):
            return 0
        party = st.get("party")
        if not party:
            timers.clear()
            continue
        if st.get("sim_now") != last_sim_now:
            last_sim_now = st.get("sim_now")
            timers.clear()  # the party made progress
            continue
        parked = set(party.get("parked") or [])
        if not parked:
            timers.clear()  # nobody is being held hostage
            continue
        now = time.monotonic()
        for w in party["roster"]:
            if w in parked:
                timers.pop(w, None)
            else:
                timers.setdefault(w, now)
        stalled = sorted(w for w, t0 in timers.items()
                         if now - t0 > args.stall_seconds)
        if not stalled:
            continue
        roster = [w for w in party["roster"] if w not in stalled]
        tw = party.get("trigger_waiter")
        tw = tw if tw in roster else roster[0]
        try:
            http.call("set_party", {"waiter_ids": roster,
                                    "trigger_waiter": tw})
        except EnvError as e:  # e.g. it parked between poll and evict
            print(f"eviction of {stalled} refused: {e}", flush=True)
            continue
        for w in stalled:
            timers.pop(w, None)
        stamp = datetime.now(timezone.utc).isoformat()
        print(f"{stamp} evicted stalled waiter(s) {stalled}: no wait for "
              f">{args.stall_seconds:g}s while {sorted(parked)} were "
              "parked", flush=True)
        try:
            data = json.loads(ep_path.read_text())
            data.setdefault("evictions", []).extend(
                {"waiter": w, "at": stamp} for w in stalled)
            data["flags"] = sorted(set(data.get("flags") or [])
                                   | {"party_eviction"})
            ep_path.write_text(json.dumps(data, indent=1))
        except Exception:
            pass  # bookkeeping only


def cmd_prepare(args) -> int:
    import yaml

    from harness.config import load_config

    tm = args.tm.upper()
    cron = tm in ("C", "D")  # driven by `drive`, not by a wait tool
    act_cron = None
    if cron:
        from harness.task import load_task_class

        if args.per_market or args.party:
            raise SystemExit("--per-market / --party are wait constructs: "
                             "not available with tm C/D")
        if args.app != "opencode":
            raise SystemExit("tm C/D episodes are fired by `harness.mcp "
                             "drive`, which launches OpenCode: pass --app "
                             "opencode")
        act_cron = load_task_class(args.task).episode_act_cron
        if not act_cron:
            raise SystemExit(f"task {args.task!r} declares no "
                             "episode_act_cron (harness/task.py): no cron "
                             "arm for it")
    base = _resolve_base(args.task, tm, args.base)
    raw = yaml.safe_load(base.read_text())
    raw["agent"] = dict(raw.get("agent") or {}, scaffold="ext:mcp")
    run_id = args.run_id or f"episode-{args.task}-tm{tm}"
    raw["run_id"] = run_id
    if args.budget is not None:
        raw["budget_usd"] = args.budget
    dir_ = (Path(args.out) / f"{run_id}-s{args.seed}").resolve()  # the
    #   registration and episode.json must carry ABSOLUTE paths: the app
    #   spawns `connect` from ITS cwd
    if dir_.exists() and any(dir_.iterdir()):
        raise SystemExit(f"episode directory already exists: {dir_}")
    ws = dir_ / "workspace"
    server_dir = dir_ / "server"
    ws.mkdir(parents=True, exist_ok=True)
    dir_.mkdir(parents=True, exist_ok=True)
    gen = dir_ / "config_gen.yaml"
    gen.write_text(yaml.safe_dump(raw, sort_keys=False))
    load_config(gen)  # fail here, not inside the subprocess
    if args.model and args.app in ("all", "opencode"):
        from harness.model_costs import parse_model_spec

        _model_limit(parse_model_spec(args.model)[1])  # before any boot

    serve_cmd = [sys.executable, "-m", "harness.serve", "--config",
                 str(gen), "--run-dir", str(server_dir),
                 "--repo-root", str(REPO_ROOT),
                 "--seed", str(args.seed)]
    if args.model:
        serve_cmd += ["--model", args.model]
    if args.stretch:
        serve_cmd += ["--stretch", args.stretch]
    if args.mock_llm:
        serve_cmd += ["--mock-llm"]
    log = (dir_ / "serve.log").open("w")
    proc = subprocess.Popen(serve_cmd, stdout=log, stderr=log,
                            cwd=str(REPO_ROOT), start_new_session=True)
    _wait_for(server_dir / "run.json", proc, 90, "run.json")
    handle = json.loads((server_dir / "run.json").read_text())

    http = _Http(handle["env_url"], handle["token"])
    contract = http.get("/contract")
    hosts = contract.get("hosts") or {}
    hosts_readonly = contract.get("hosts_readonly") or {}
    if hosts and not (args.model or args.mock_llm):
        proc.terminate()
        raise SystemExit("web tasks meter LLM usage through the env's /llm "
                         "proxy (design D7): pass --model api_provider:model "
                         "(or --mock-llm for a plumbing check)")
    fenced = platform.system() == "Darwin" and os.environ.get("EPISODE_FENCE") != "off"
    model = handle.get("model") or None
    budget = contract["llm"]["budget_usd"]
    wait_tool = "run_program" if tm == "B" else "sleep"
    instruction = (contract.get("instruction_md") or "").rstrip()
    if cron:
        appendix = CRON_INSTRUCTION_APPENDIX.format(budget=budget)
    else:
        appendix = INSTRUCTION_APPENDIX.format(wait_tool=wait_tool,
                                               budget=budget)
    if hosts:
        appendix += "\n" + (CRON_HOSTS_LINE if cron else HOSTS_LINE)
    party = _party_spec(args, raw)
    if party:
        appendix += "\n" + PARTY_RULE
    episode_json = dir_ / "episode.json"
    # fenced web task: the browser runs as its own fenced server
    pw_port = ((args.playwright_port or _free_port())
               if fenced and hosts and args.app in ("all", "opencode") else None)

    def build_workspace(wsdir: Path, market_id: str | None,
                        waiter: str | None) -> None:
        wsdir.mkdir(parents=True, exist_ok=True)
        sections = [instruction] if instruction else []
        if market_id is not None:
            sections.append(
                MARKET_SECTION.format(market_id=market_id).rstrip())
        if tm == "B":
            # the tm=B teaching kit, provisioned with envkit.py
            from harness.authored import SLEEP_PROGRAM
            from harness.contract import render_envkit
            from harness.task import task_dir

            sections.append(render_skill_appendix())
            (wsdir / "envkit.py").write_text(
                render_envkit(contract["tools"], runner=EPISODE_RUNNER_REF))
            (wsdir / "sleep.py").write_text(SLEEP_PROGRAM)
            example = task_dir(args.task) / "agent" / "example_gatekeeper.py"
            if example.exists():
                (wsdir / "example_gatekeeper.py").write_text(
                    render_example(example.read_text()))
        if cron:
            sections.append(
                SCHEDULE_SECTION[tm].format(act_cron=act_cron).rstrip())
        sections.append(appendix)
        (wsdir / "INSTRUCTION.md").write_text("\n\n".join(sections) + "\n")
        if args.app in ("all", "claude"):
            (wsdir / ".mcp.json").write_text(
                json.dumps(_mcp_json(sys.executable, episode_json, waiter),
                           indent=1))
        if args.app in ("all", "codex"):
            (wsdir / ".codex").mkdir(exist_ok=True)
            (wsdir / ".codex" / "config.toml").write_text(
                _codex_config_toml(sys.executable, episode_json, waiter))
        if args.app in ("all", "opencode") and waiter is None:
            (wsdir / "opencode.json").write_text(json.dumps(
                _opencode_json(sys.executable, wsdir, handle, tm=tm,
                               task=args.task, hosts=hosts,
                               hosts_readonly=hosts_readonly, model=model,
                               pin=args.playwright_pin, fenced=fenced,
                               playwright_cli=args.playwright_cli,
                               browser=args.playwright_browser,
                               playwright_port=pw_port,
                               headed=args.playwright_headed),
                indent=1))

    waiters: dict[str, str] = {}
    if party:
        for w, market_id in party:
            sub = ws / w
            build_workspace(sub, market_id, w)
            waiters[w] = str(sub)
    else:
        build_workspace(ws, None, None)

    if cron:
        # the cadence, installed at sim_start exactly as a checked-in cron main
        # does in its bootstrap firing: under D the default schedule (an
        # agent-owned row, seeded once), under C a fixed crontab row.
        # __bootstrap__ stays unconsumed — it is the driver's first firing
        http.call("set_crontab", {"entries": [
            {"id": "act", "cron_expr": act_cron,
             **({"agent_owned": True} if tm == "D" else {})}]})
        trig = {}
    else:
        trig = http.get("/trigger/next")  # consume __bootstrap__
    if party:
        ids = [w for w, _ in party]
        http.call("set_party", {"waiter_ids": ids,
                                "trigger_waiter": ids[0]})
    episode = {"run_id": handle["run_id"], "env_url": handle["env_url"],
               "token": handle["token"],
               "agent_token": handle.get("agent_token") or handle["token"],
               "tm": tm, "task": args.task, "model": model,
               "workspace": str(ws), "server_dir": str(server_dir),
               "serve_pid": proc.pid, "trigger_id": trig.get("id"),
               "llm_metered": bool(args.model), "hosts": hosts,
               "hosts_readonly": hosts_readonly, "app": args.app,
               "playwright_url": (playwright_url(pw_port)
                                  if pw_port is not None else None),
               "fenced": fenced,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "clients": []}
    script = dir_ / ("run_opencode_firing.sh" if cron else "run_opencode.sh")
    if cron:
        episode.update(driver="cron", act_cron=act_cron,
                       firing_script=str(script))
    if party:
        episode.update(waiters=waiters,
                       party={"trigger_waiter": party[0][0]},
                       evictions=[])
    episode_json.write_text(json.dumps(episode, indent=1))
    if party and args.stall_seconds > 0:
        wd_log = (dir_ / "watchdog.log").open("w")
        wd = subprocess.Popen(
            [sys.executable, "-m", "harness.mcp", "watchdog",
             "--episode", str(episode_json),
             "--stall-seconds", str(args.stall_seconds)],
            stdout=wd_log, stderr=wd_log, cwd=str(REPO_ROOT),
            start_new_session=True)
        episode["watchdog_pid"] = wd.pid
        episode_json.write_text(json.dumps(episode, indent=1))
    print(json.dumps({"episode": str(episode_json),
                      "workspace": str(ws),
                      "run_id": handle["run_id"],
                      "sim": f"{handle['sim_start']} -> {handle['sim_end']}",
                      "tm": tm, "hosts": hosts,
                      "opencode": (str(script)
                                   if args.app in ("all", "opencode") else None)},
                     indent=1))
    if party:
        lines = "\n".join(
            f"  codex exec -C {waiters[w]} -s workspace-write -m gpt-5.5 "
            '"Read INSTRUCTION.md in this workspace and follow it." &'
            for w, _ in party)
        print(f"\nParty episode: one sub-workspace per agent under {ws}/. "
              "Open EACH in its own agent session (as that session's "
              f"folder/cwd), e.g.:\n{lines}\nThe clock advances only when "
              "every agent is waiting"
              + (f"; the stall watchdog evicts a non-waiting agent after "
                 f"{args.stall_seconds:g}s" if args.stall_seconds > 0
                 else " (stall watchdog DISABLED)")
              + ". When done: "
              f"python -m harness.mcp settle --episode {episode_json}",
              file=sys.stderr)
        return 0
    if args.app in ("all", "opencode"):
        pw_cmd = (playwright_command(args.playwright_pin, args.playwright_cli,
                                     args.playwright_browser, hosts,
                                     port=pw_port, output_dir=ws / ".playwright-mcp",
                                     headed=args.playwright_headed)
                  if pw_port is not None else None)
        script.write_text(_opencode_script(ws, dir_, handle, hosts, hosts_readonly,
                                           model, playwright=pw_cmd,
                                           playwright_port=pw_port, cron=cron,
                                           wait_tool=wait_tool))
        script.chmod(0o755)
        (dir_ / "opencode_data").mkdir(exist_ok=True)  # the episode's XDG_DATA_HOME
        if not fenced:
            fence_note = ("UNFENCED (no sandbox-exec on this platform): bash is "
                          "denied in opencode.json instead")
        elif pw_port is not None:
            fence_note = ("two fences: the browser server reaches the web hosts; "
                          "OpenCode + its shell reach the env, the read-only "
                          "hosts and the browser server only — the browser is "
                          "the sole write path; the episode dir unreadable "
                          "except the workspace")
        else:
            fence_note = ("fenced: outbound only to the env port, the episode "
                          "dir unreadable except the workspace")
        launch = (f"python -m harness.mcp drive --episode {episode_json}\n"
                  f"(one `sh {script.name}` per firing)" if cron
                  else f"sh {script}")
        print(f"\nOpenCode ({fence_note}):\n  {launch}", file=sys.stderr)
    if args.app in ("all", "codex", "claude"):
        # the interactive apps read their registration from the workspace
        # cwd; name only the ones this prepare actually wrote
        reads = {"claude": "Claude Code reads .mcp.json there",
                 "codex": "Codex .codex/config.toml"}
        which = ", ".join(reads[a] for a in ("claude", "codex")
                          if args.app in ("all", a))
        print(f"\nOpen {ws} itself as the folder/cwd in your agent app "
              f"({which}), then prompt:\n  Read INSTRUCTION.md in this "
              "workspace and follow it.", file=sys.stderr)
    print("\nWhen done: "
          f"python -m harness.mcp settle --episode {episode_json}",
          file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    ep = load_episode(Path(args.episode))
    print(json.dumps(_episode_http(ep).get("/status"), indent=1))
    return 0


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def cmd_settle(args) -> int:
    ep = load_episode(Path(args.episode))
    results = Path(ep["server_dir"]) / "results.json"
    burned = 0
    if _pid_alive(ep.get("driver_pid")):
        raise SystemExit(f"the cron driver (pid {ep['driver_pid']}) is still "
                         "firing this episode: stop it first")
    try:
        http = _episode_http(ep)
        trig_id = (None if results.exists()  # a driven episode that reached `done`
                   else http.get("/status")["active_trigger"])
        while not results.exists():
            if trig_id is not None:
                http.post(f"/trigger/{trig_id}/exit",
                          {"code": 0, "killed": False,
                           "output": "episode settled by controller"})
            nxt = http.get("/trigger/next")
            if nxt.get("done"):
                break
            trig_id = nxt.get("id")  # operator stopped early: burn the rest
            burned += 1
    except OSError:
        # connection refused: the server already answered `done` to the
        # driver and closed its listener — results.json is being written
        pass
    t0 = time.monotonic()
    while not results.exists() and time.monotonic() - t0 < 30:
        time.sleep(0.2)
    if not results.exists():
        raise SystemExit("sim reported done but results.json never appeared")
    res = json.loads(results.read_text())
    out = {"run_id": res.get("run_id"),
           "primary": res.get("performance", {}).get("primary"),
           "spent_usd": res.get("resources", {}).get("spent_usd"),
           "flags": res.get("flags"),
           "triggers_burned": burned,
           "results": str(results)}
    print(json.dumps(out, indent=1))
    return 0


def cmd_reprompt(args) -> int:
    """The continue-or-stop decision of run_opencode.sh's agent loop, made
    each time `opencode run` exits. Exit 0 + the session id on stdout:
    re-prompt. Exit 1: stop, the episode is ready to settle — the run is
    over, the LLM budget is spent, or REPROMPT_MAX_STALLED re-prompts in
    a row left the clock where it was. Exit 3: stop, broken launch — the
    app never opened a session. One row per decision in reprompts.jsonl."""
    from harness.episode_drive import BUDGET_DEAD_FLAGS, session_id_in

    ep = load_episode(Path(args.episode))
    dir_ = Path(ep["_path"]).parent
    log = dir_ / "reprompts.jsonl"
    st = _episode_http(ep).get("/status")
    rows = ([json.loads(l) for l in log.read_text().splitlines()]
            if log.exists() else [])
    stalled = 0
    for row in reversed(rows):
        if row["sim_now"] != st["sim_now"]:
            break
        stalled += 1
    events = dir_ / "opencode_events.jsonl"
    sid = session_id_in(events.read_bytes()) if events.exists() else None
    if st["done"] or st["sim_now"] >= st["sim_end"]:
        stop = "experiment_over"
    elif BUDGET_DEAD_FLAGS & set(st["flags"]):
        stop = "budget_exhausted"
    elif sid is None:
        stop = "no_session"
    elif stalled >= REPROMPT_MAX_STALLED:
        stop = "stalled"
    else:
        stop = None
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sim_now": st["sim_now"],
                            "real_time": datetime.now(timezone.utc).isoformat(),
                            "action": "stop" if stop else "reprompt",
                            **({"reason": stop} if stop else {}),
                            "session_id": sid}) + "\n")
    if stop:
        print(f"agent loop stopped: {stop} at {st['sim_now']}", file=sys.stderr)
        return 3 if stop == "no_session" else 1
    print(sid)
    return 0


def cmd_drive(args) -> int:
    from harness.episode_drive import drive

    return drive(Path(args.episode))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness.mcp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("connect", help="stdio MCP service (spawned by the app)")
    c.add_argument("--episode", default=None)
    c.add_argument("--from-env", action="store_true",
                   help="take the handle from ENV_URL/ENV_TOKEN/ENV_TM/"
                        "ENV_WORKSPACE/... (OpenCode registration; the "
                        "fenced actor cannot read episode.json)")
    c.add_argument("--waiter", default=None,
                   help="party episodes: bind this bridge to one waiter "
                        "(its sub-workspace; waiter_id forced on waits)")
    c.set_defaults(fn=cmd_connect)

    r = sub.add_parser("prepare", help="controller: boot a sim + workspace")
    r.add_argument("--task", required=True)
    r.add_argument("--tm", required=True,
                   choices=["A", "B", "C", "D", "a", "b", "c", "d"],
                   help="A/B: the agent waits (sleep / run_program); C/D: "
                        "cron-fired by `drive` (D adds the schedule CRUD)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--base", default=None, help="base run yaml (default: "
                   "the task's algnone smoke cell for this tm)")
    r.add_argument("--out", default="episodes")
    r.add_argument("--run-id", default=None)
    r.add_argument("--budget", type=float, default=None)
    r.add_argument("--stretch", default=None, help="shorten the sim "
                   "(harness.serve --stretch, e.g. 2d)")
    r.add_argument("--model", default=None,
                   help="api_provider:model for /llm metering (omit: the "
                        "external harness bills its own account and the "
                        "episode is flagged llm_unmetered)")
    r.add_argument("--mock-llm", action="store_true")
    r.add_argument("--per-market", action="store_true",
                   help="party episode: one agent per entry "
                        "of the base config's task.markets roster")
    r.add_argument("--party", default=None,
                   help="party episode: explicit comma-separated waiter "
                        "ids (any task; no market scoping)")
    r.add_argument("--stall-seconds", type=float, default=600,
                   help="party stall watchdog: evict an agent that ends "
                        "its turn without waiting after this much real "
                        "time (0 disables the watchdog)")
    r.add_argument("--app", default="all",
                   choices=["all", "opencode", "codex", "claude"],
                   help="which registrations to write into the workspace "
                        "(default: all three)")
    r.add_argument("--playwright-pin", default=PLAYWRIGHT_PIN,
                   help="@playwright/mcp version for the OpenCode config")
    r.add_argument("--playwright-cli", default=None,
                   help="path to @playwright/mcp's cli.js (default: "
                        "$PLAYWRIGHT_MCP_CLI, else .tools/playwright-mcp, "
                        "else npx)")
    r.add_argument("--playwright-port", type=int, default=None,
                   help="fixed port for the fenced browser server (default: "
                        "an ephemeral one picked now). Batch launchers pass "
                        "one per episode from outside the ephemeral range: "
                        "under tm=C/D the port is free between firings, and "
                        "a concurrent episode could otherwise be handed it")
    r.add_argument("--playwright-headed", action="store_true",
                   help="show the browser window (watch the agent act); "
                        "default headless")
    r.add_argument("--playwright-browser", default=None,
                   help="Playwright --browser channel (chrome, msedge, "
                        "chromium…); default: Playwright's bundled chromium")
    r.set_defaults(fn=cmd_prepare)

    w = sub.add_parser("watchdog", help="party stall watchdog "
                       "(spawned by prepare; see --stall-seconds)")
    w.add_argument("--episode", required=True)
    w.add_argument("--stall-seconds", type=float, default=600)
    w.add_argument("--poll-seconds", type=float, default=10)
    w.set_defaults(fn=cmd_watchdog)

    d = sub.add_parser("drive", help="controller: the cron driver of a "
                       "tm=C/D episode — fires the app per trigger until "
                       "the run is done")
    d.add_argument("--episode", required=True)
    d.set_defaults(fn=cmd_drive)

    rp = sub.add_parser("reprompt", help="controller: run_opencode.sh's "
                        "agent loop — exit 0 + session id to re-prompt a "
                        "tm=A/B actor whose turn ended, exit 1 to stop")
    rp.add_argument("--episode", required=True)
    rp.set_defaults(fn=cmd_reprompt)

    s = sub.add_parser("status")
    s.add_argument("--episode", required=True)
    s.set_defaults(fn=cmd_status)

    t = sub.add_parser("settle", help="controller: finish the run")
    t.add_argument("--episode", required=True)
    t.set_defaults(fn=cmd_settle)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
