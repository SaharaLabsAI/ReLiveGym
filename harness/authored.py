"""Authored wait programs — extended TM-B.

The TM-B ReACT actor may write its own wake condition as code and run it
forward through sim time: WorkspaceApp is the author affordance (file
tools jailed to the actor's private scratch, workspace/agents/<name>/),
ProgramApp is run-and-wait (execute an authored gatekeeper until it calls
envkit.handover, a scheduled trigger fires, or the deadline). Both are
provisioned for tm == "B" only (harness/apps.py) — the A/B contrast stays
"which wait tools the ReACT spine holds", and control always returns to
the spine (the handover payload / timeout is an ordinary tool result).

Isolation is topological, not behavioral: the program's only
world door is the generated `envkit` module in its jail, whose functions
dispatch to the same priced, redacting task handlers the actor's tools
use — a client, never the server. In-process execution is a resolved
decision (D1); Python-level introspection attacks and same-filesystem
reads are explicitly out of the threat model until leg-3 stubbing proves
leaky. Time/pricing contract: fetch is the only observation and the
only meter — each fetch bills its real-rate fee and consumes the task's
rate limiter exactly like the same-named agent tool; envkit.wait is the
only clock, and it is free; local compute and jail file IO are free at a
frozen instant. This is the ONLY event-driven wait mechanism — mechanism
neutrality by construction — and under tm=B the ONLY wait, period: tm=B
manifests carry no sleep tool; every jail is seeded with sleep.py (the
edit-WAKE_AT blind-wait exemplar quoted in harness/authored_skill.md,
which run.py appends to the rendered INSTRUCTION.md); scaffold plumbing
that must park (the per-market coordinator) writes its own one-liner
into its own jail and runs that.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timezone
from pathlib import Path

from harness.env_tools import EnvApp, ToolError, build_registry, tool
from harness.timeutil import iso, parse_iso

HANDOVER_LIMIT_KB = 16  # loose context-window guard on handover payloads (D3)
READ_LIMIT_KB = 16  # same guard on read_file pages: line-based paging
#   alone lets one huge single-line file (e.g. an authored program's
#   accumulated JSON state) land verbatim in the actor's context — the
#   month-run failure mode (a 3.9 MB state file -> 1.3M-token request)
STEP_CAP = 2_000_000  # traced lines per run_program call: liveness only (D2)

# Authored programs are synchronous and hold their worker thread for the
# program's whole lifetime (a parked envkit.wait blocks the thread), so
# they run on their own pool. On the event loop's default executor they
# starve the loop's other run_in_executor work — including asyncio's
# getaddrinfo, so every LLM call hangs pre-socket — once enough programs
# park at once (two in-process permarket sims = 18 parked programs, more
# than the min(32, ncpu+4) default pool).
_PROGRAM_EXECUTOR = ThreadPoolExecutor(
    max_workers=64, thread_name_prefix="authored-program")

# Seeded into every jail: the trivial gatekeeper, wake time in code
# (semantic wake times ALWAYS live in the program; run_program's `until` is
# a dumb backstop). The skill appendix quotes it verbatim
# (harness/authored_skill.md must stay in sync); the seeded WAKE_AT is
# deliberately in the past so running it unedited fails legibly.
SLEEP_PROGRAM = '''\
"""Blind wait: edit WAKE_AT, then run_program(path="sleep.py")."""

import envkit

WAKE_AT = "2026-01-01T00:00:00Z"  # EDIT ME: when to wake
envkit.wait(WAKE_AT)
'''

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _parse_time(s, field: str) -> datetime:
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {field!r}: {s!r}")


# -- the per-actor jail ----------------------------------------------------------------


def _jail(sim, args: dict) -> Path:
    """The calling actor's private scratch root, provisioned on first use
    with envkit.py + the task's example gatekeeper. Identity is cooperative
    (waiter_id, exactly as for the wait tools; mono react actors omit it and
    share the default): the jail protects the file bus against races and
    accidents across concurrent actors , not against an adversarial
    actor — the same trust line as D1's in-process execution."""
    wid = args.get("waiter_id") or "actor"
    if not isinstance(wid, str) or not _NAME_RE.match(wid):
        raise ToolError(f"invalid waiter_id {wid!r}")
    root = (sim.workspace / "agents" / wid).resolve()
    if not (root / "envkit.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "envkit.py").write_text(_render_envkit(sim), encoding="utf-8")
        (root / "sleep.py").write_text(SLEEP_PROGRAM, encoding="utf-8")
        example = sim.task.authored_example()
        if example and not (root / "example_gatekeeper.py").exists():
            (root / "example_gatekeeper.py").write_text(example,
                                                        encoding="utf-8")
    return root


def _resolve(root: Path, path, field: str = "path") -> Path:
    if not isinstance(path, str) or not path:
        raise ToolError(f"{field!r} must be a non-empty string")
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ToolError(f"path outside your workspace: {path!r}")
    return p


# -- author affordance: workspace file tools -------------------------


class WorkspaceApp(EnvApp):
    """Read/Write/Edit/ls jailed to the actor's scratch — the same file
    surface a coding agent uses; authored modules, one-shot scripts, and
    the code<->agent file bus are all just files here. The runtime's own
    state (state.json, learned blocks, main.py) lives ABOVE the jail root
    and is unreachable by construction."""

    @tool("ls(path?: str) -> free: list your private scratch workspace "
          "(yours alone; ships envkit.py — the API an authored program "
          "imports — sleep.py, the blind-wait program, and "
          "example_gatekeeper.py, a runnable watcher template)",
          schema={"additionalProperties": False, "properties": {"path": {"type": "string"}}, "required": [], "type": "object"})
    async def ls(self, args: dict) -> dict:
        root = _jail(self.sim, args)
        base = _resolve(root, args.get("path") or ".")
        if not base.exists():
            raise ToolError(f"no such path: {args.get('path')!r}")
        files = sorted(p for p in base.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        return {"root": str(root.relative_to(self.sim.workspace)),
                "files": [{"path": str(p.relative_to(root)),
                           "bytes": p.stat().st_size} for p in files]}

    @tool("read_file(path: str, offset?: int, limit?: int) -> free: read a "
          "workspace file (1-based line offset, default limit 2000 lines; "
          f"a page is clipped at {READ_LIMIT_KB} KB)",
          schema={"additionalProperties": False, "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}, "path": {"type": "string"}}, "required": ["path"], "type": "object"})
    async def read_file(self, args: dict) -> dict:
        root = _jail(self.sim, args)
        p = _resolve(root, args.get("path"))
        if not p.is_file():
            raise ToolError(f"no such file: {args.get('path')!r}")
        offset = args.get("offset", 1)
        limit = args.get("limit", 2000)
        if not isinstance(offset, int) or offset < 1:
            raise ToolError("offset must be an int >= 1")
        if not isinstance(limit, int) or limit < 1:
            raise ToolError("limit must be an int >= 1")
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        page = lines[offset - 1:offset - 1 + limit]
        content = "".join(page)
        out = {"path": args["path"], "offset": offset,
               "total_lines": len(lines), "lines": len(page),
               "truncated": offset - 1 + limit < len(lines)}
        blob = content.encode("utf-8")
        if len(blob) > READ_LIMIT_KB * 1024:
            content = blob[:READ_LIMIT_KB * 1024].decode(
                "utf-8", errors="ignore")
            out["truncated"] = True
            out["clipped"] = (f"page over the {READ_LIMIT_KB} KB read "
                              f"limit ({len(blob)} bytes) — showing the "
                              f"head; have a program summarize the file "
                              f"instead of reading it whole")
        out["content"] = content
        return out

    @tool("write_file(path: str, content: str) -> free: create or "
          "overwrite a workspace file (parent dirs created)")
    async def write_file(self, args: dict) -> dict:
        root = _jail(self.sim, args)
        p = _resolve(root, args.get("path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("'content' must be a string")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": args["path"], "bytes": len(content.encode())}

    @tool("edit_file(path: str, old: str, new: str) -> free: in-place "
          "edit — `old` must occur exactly once in the file")
    async def edit_file(self, args: dict) -> dict:
        root = _jail(self.sim, args)
        p = _resolve(root, args.get("path"))
        if not p.is_file():
            raise ToolError(f"no such file: {args.get('path')!r}")
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ToolError("edit_file needs a non-empty string 'old' and "
                            "a string 'new'")
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            raise ToolError(f"`old` occurs {n} times (must be exactly 1)")
        text = text.replace(old, new)
        p.write_text(text, encoding="utf-8")
        return {"path": args["path"], "bytes": len(text.encode())}


# -- run-and-wait --------------------------------------------------------


class _Exit(BaseException):
    """Program-ending control flow (handover / trigger / timeout /
    validate). BaseException so an authored `except Exception` cannot
    swallow the spine's return-of-control."""

    def __init__(self, result: dict):
        self.result = result


class _Guard(Exception):
    """A tripped guard, with its documented one-line reason."""


# thread id -> the executing program's bridge; envkit resolves through this,
# which is what keeps one shared envkit module safe under wait-party
# concurrency (each actor's program runs in its own worker thread)
_BRIDGES: dict[int, "_Bridge"] = {}


class _Bridge:
    """What the authored program gets: bound methods that
    run the priced task handlers on the env loop and return plain,
    already-redacted dicts. Pass a client, never the server."""

    def __init__(self, sim, loop, registry, root: Path, deadline: datetime,
                 waiter_id: str | None, validate: bool):
        from harness.apps import SleepApp  # lazy: apps.py imports this module

        self.sim = sim
        self.loop = loop
        self.registry = registry
        self.root = root
        self.deadline = deadline
        self.waiter_id = waiter_id
        self.validate = validate
        self.n_fetches = 0
        self.n_waits = 0
        self._sleep = SleepApp(sim).sleep  # blind, free, trigger-delivering

    def _run(self, coro):
        res = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        self.sim.touch()  # keep the real-time watchdog fed during long runs
        return res

    def now(self) -> datetime:
        return self.sim.clock.now

    def call(self, name: str, args: dict):
        if self.validate:
            raise _Exit({"validate": "ok", "reached": name})
        entry = self.registry.get(name)
        if entry is None:
            raise _Guard(f"unknown env function {name!r}")
        res = self._run(entry[1](dict(args or {})))
        self.n_fetches += 1
        return res

    def wait(self, until):
        if self.validate:
            raise _Exit({"validate": "ok", "reached": "wait"})
        if isinstance(until, str):
            try:
                until = parse_iso(until)
            except ValueError:
                raise _Guard(f"wait(until): bad iso datetime {until!r}")
        if not isinstance(until, datetime):
            raise _Guard("wait(until) takes a datetime or an iso string")
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        until = min(until, self.deadline)
        if until <= self.sim.clock.now:
            raise _Guard("wait(until) must be in the future — sim time "
                         "only moves forward")
        wargs = {"until": iso(until)}
        if self.waiter_id:
            wargs["waiter_id"] = self.waiter_id
        res = self._run(self._sleep(wargs, brief=False))  # not a wake
        self.n_waits += 1
        if res.get("experiment_over"):
            raise _Exit({"experiment_over": True, "woke_for": "timeout"})
        if res.get("woke_for") == "trigger":
            raise _Exit({"woke_for": "trigger", "trigger": res["trigger"]})
        if self.sim.clock.now >= self.deadline:
            raise _Exit({"woke_for": "timeout"})
        return {"now": res["now"]}

    def handover(self, payload) -> None:
        if self.validate:
            raise _Exit({"validate": "ok", "reached": "handover"})
        try:
            blob = json.dumps(payload, default=str)
        except (TypeError, ValueError) as e:
            raise _Guard(f"handover payload is not JSON-serializable: {e}")
        if len(blob) > HANDOVER_LIMIT_KB * 1024:
            raise _Guard(
                f"handover payload {len(blob) / 1024:.1f} KB over the "
                f"{HANDOVER_LIMIT_KB} KB limit — write bulk to a workspace "
                f"file and hand over a pointer")
        raise _Exit({"woke_for": "handover", "payload": json.loads(blob)})


def _program_registry(sim) -> dict:
    """The program-callable surface: exactly the task's own env tools
    (priced + redacted by the same handlers the actor uses). Wait-tagged
    tools are excluded — sim time moves only through envkit.wait."""
    return {name: entry
            for name, entry in build_registry(sim.task.env_apps(sim)).items()
            if "wait" not in entry[0].tags}


class ProgramApp(EnvApp):
    @tool("run_program(path: str, until?: iso datetime, validate?: bool) "
          "-> run a Python program from your workspace forward through "
          "sim time, until it calls envkit.handover(payload) — returned "
          "here verbatim — a scheduled trigger fires, or the `until` "
          "deadline (a hard timeout backstop, default: experiment end; "
          "the program picks its actual wake times via envkit.wait). "
          "Inside it the world is observable ONLY through envkit's fetch "
          "functions (each billed like the same-named tool); sim time "
          "advances ONLY inside envkit.wait (free); local compute and "
          "workspace file IO are free. validate=true dry-runs it at zero "
          "cost and zero sim time, surfacing errors. Read envkit.py in "
          "your workspace for the exact API.",
          price="free (its fetches are billed)", tags=("wait",),
          schema={"additionalProperties": False, "properties": {"path": {"type": "string"}, "until": {"format": "date-time", "type": "string"}, "validate": {"type": "boolean"}, "waiter_id": {"type": "string"}}, "required": ["path"], "type": "object"})
    async def run_program(self, args: dict) -> dict:
        sim = self.sim
        validate = bool(args.get("validate"))
        root = _jail(sim, args)
        path = args.get("path")
        file = _resolve(root, path)
        if not file.is_file():
            raise ToolError(f"no such program: {path!r} (ls your workspace)")
        if sim.clock.finished:
            return {"experiment_over": True, "now": iso(sim.clock.now)}
        if validate:
            deadline = sim.clock.now
        elif args.get("until") is None:
            deadline = sim.cfg.sim_end
        else:
            deadline = min(_parse_time(args.get("until"), "until"),
                           sim.cfg.sim_end)
            if deadline <= sim.clock.now:
                raise ToolError("'until' must be a future iso datetime — "
                                "the program's hard timeout backstop")
        src = file.read_text(encoding="utf-8")
        n_fetches = n_waits = 0
        try:
            code = compile(src, str(file), "exec")
        except SyntaxError as e:
            outcome = {"woke_for": "error",
                       "error": f"syntax error: {e.msg} "
                                f"({file.name} line {e.lineno})"}
        else:
            bridge = _Bridge(sim, asyncio.get_running_loop(),
                             _program_registry(sim), root, deadline,
                             args.get("waiter_id"), validate)
            outcome = await asyncio.get_running_loop().run_in_executor(
                _PROGRAM_EXECUTOR, _execute, code, bridge, root)
            n_fetches, n_waits = bridge.n_fetches, bridge.n_waits
        if validate and "validate" not in outcome:
            # tripped (or ran out) before the first envkit call
            if outcome.get("woke_for") == "error":
                outcome = {"validate": "error", "error": outcome["error"]}
            else:
                outcome = {"validate": "ok",
                           "note": "program ran without touching envkit"}
        async with sim.lock:
            sim.ledger.append("run_program", sim.clock.now, cost=0.0,
                              path=path, validate=validate,
                              woke_for=outcome.get("woke_for"),
                              fetches=n_fetches, waits=n_waits)
            costs = {}
            if not validate and n_waits and not sim.clock.finished:
                brief = sim.daily_brief(sim.clock.now)  # first wake of a
                if brief is not None:                   # sim date
                    costs = {"costs": brief}
        return {"now": iso(sim.clock.now), **outcome, **costs}


def _load_envkit(root: Path) -> None:
    """Make `import envkit` inside authored code resolve to this jail's
    generated stub. One shared sys.modules entry is safe: its functions
    look up the per-thread bridge, so concurrent actors' programs each see
    their own jail/deadline through the same module object."""
    path = root / "envkit.py"
    mod = sys.modules.get("envkit")
    if mod is not None and getattr(mod, "__file__", None) == str(path):
        return
    spec = importlib.util.spec_from_file_location("envkit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["envkit"] = mod
    spec.loader.exec_module(mod)


def _format_error(e: BaseException, prefix: str) -> str:
    tb, loc = e.__traceback__, ""
    while tb is not None:
        f = tb.tb_frame.f_code.co_filename
        if f.startswith(prefix):
            loc = f" ({Path(f).name} line {tb.tb_lineno})"
        tb = tb.tb_next
    return f"{type(e).__name__}: {e}{loc}"


def _execute(code, bridge: _Bridge, root: Path) -> dict:
    """Run compiled authored code in a worker thread. The trace hook is the
    D2 liveness backstop (counts lines executed in jail files only), not a
    determinism sandbox — `random`/wall-clock are allowed."""
    tid = threading.get_ident()
    _BRIDGES[tid] = bridge
    jail = str(root)
    prefix = jail + os.sep
    steps = 0

    def tracer(frame, event, arg):
        nonlocal steps
        if not frame.f_code.co_filename.startswith(prefix):
            return None
        if event == "line":
            steps += 1
            if steps > STEP_CAP:
                raise _Guard(
                    "step budget exceeded — a gatekeeper must envkit.wait();"
                    " a loop that never waits makes no sim-time progress")
        return tracer

    sys.path.insert(0, jail)
    _load_envkit(root)
    g = {"__name__": "__main__", "__file__": code.co_filename}
    sys.settrace(tracer)
    try:
        exec(code, g)
    except _Exit as e:
        return e.result
    except _Guard as e:
        return {"woke_for": "error", "error": str(e)}
    except SystemExit:
        return {"woke_for": "return",
                "note": "program exited without handover"}
    except BaseException as e:
        return {"woke_for": "error", "error": _format_error(e, prefix)}
    finally:
        sys.settrace(None)
        del _BRIDGES[tid]
        try:
            sys.path.remove(jail)
        except ValueError:
            pass
        for name, mod in list(sys.modules.items()):
            # purge jail modules so an actor's edits never go stale in the
            # import cache (concurrent holders keep their references alive)
            if (getattr(mod, "__file__", None) or "").startswith(prefix):
                del sys.modules[name]
    return {"woke_for": "return", "note": "program returned without handover"}


# -- the generated envkit stub ------------------------------

_ENVKIT_HEADER = '''\
"""envkit — an authored program's ONLY window on the world.

Import this from a program run via the run_program tool; its functions
fail cleanly anywhere else. The contract:

- Observation = the fetch functions below; each call is billed exactly
  like the same-named agent tool. Fetching never moves the clock.
- wait(until) is the ONLY way sim time advances, and it is free. A loop
  that never waits makes no sim-time progress and trips the step cap.
- handover(payload) ends the run: payload (JSON-able, <= <LIMIT> KB)
  becomes the run_program result your ReACT loop sees next turn. Write
  bulk to files under workspace() and hand over a pointer + summary.
- A due scheduled trigger or the run_program deadline also ends the run
  (woke_for "trigger" / "timeout").
- Local compute and file IO under workspace() — your private scratch —
  are free: they happen at a frozen instant.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path


def _bridge():
    from harness.authored import _BRIDGES  # bound only while run_program runs
    b = _BRIDGES.get(threading.get_ident())
    if b is None:
        raise RuntimeError("envkit functions work only inside run_program")
    return b


def workspace() -> Path:
    """Your private scratch root — the only writable place."""
    return _bridge().root


def now() -> datetime:
    """Current sim time (free; constant between wait() calls)."""
    return _bridge().now()


def deadline() -> datetime:
    """This run's hard deadline (run_program's `until`, or experiment
    end): reaching it exits the program with woke_for "timeout"."""
    return _bridge().deadline


def wait(until: "datetime | str") -> dict:
    """Advance sim time to `until` (free); returns {"now": ...}. A due
    trigger, the run deadline, or experiment end exits the program
    instead, returning control to your ReACT loop."""
    return _bridge().wait(until)


def handover(payload) -> None:
    """End the run: `payload` becomes the run_program tool result."""
    _bridge().handover(payload)
'''


def _render_envkit(sim) -> str:
    """envkit.py content for a jail: the static contract + one wrapper per
    task tool, doc verbatim from the live manifest — the security boundary
    and the API reference are one artifact, so they cannot drift."""
    parts = [_ENVKIT_HEADER.replace("<LIMIT>", str(HANDOVER_LIMIT_KB))]
    reg = _program_registry(sim)
    for name in sorted(reg):
        tdef = reg[name][0]
        doc = tdef.doc.replace("\\", "\\\\").replace('"""', "'''")
        parts.append(f"def {name}(**args) -> dict:\n"
                     f'    """{doc} [{tdef.price}]"""\n'
                     f'    return _bridge().call("{name}", args)\n')
    return "\n\n".join(parts)
