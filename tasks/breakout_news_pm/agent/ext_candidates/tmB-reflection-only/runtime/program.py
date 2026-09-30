"""Client-side authored programs.
Program library — copied into every workspace as runtime/program.py.

`run_program(path, until=None, waiter_id=None)` executes an authored
Python file forward through sim time IN THE ACTOR'S OWN PROCESS (the
calling thread), with an `envkit` module whose only world door is the
environment's HTTP contract:

    envkit.fetch_*(...)   -> POST /call/<tool>   (billed like the same-named tool)
    envkit.wait(until)    -> POST /call/sleep    (parks in the server; party-aware)
    envkit.now()          -> POST /call/get_time (free)
    envkit.handover(x)    -> ends the run; x is the returned payload

The invariant of the server-side ProgramApp (harness/authored.py) is kept
by construction: the server is the only site that advances sim time —
every wait is an HTTP call that parks or delivers a trigger — and every
observation is a billed, rate-limited tool call. What moved is only WHERE
the authored code runs: here, so no agent code executes inside the env
process. Liveness is the run's real-time watchdog (a program that stops
calling the environment gets its process killed), not a step counter.

Same outcomes as the tool: {"now", "woke_for": handover|trigger|timeout|
error|return, ...} or {"experiment_over": True}. Concurrent programs (one
per waiter thread) share one envkit module: its functions resolve the
per-thread bridge, so each sees its own root/deadline/waiter_id.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .env_client import Env, iso, parse_iso

HANDOVER_LIMIT_KB = 16

# thread id -> the executing program's bridge
_BRIDGES: dict[int, "_Bridge"] = {}

# Optional harness-owned observer of program-side tool calls, called as
# (waiter_id, tool, args, result, sim_time) after every billed envkit
# fetch. Best-effort: it can never change the program's control flow.
# (tmB-reflection-only: under TM-B the authored programs do most of the
# searching and part of the notifying, bypassing the agent's registry.)
CALL_OBSERVER = None


def current_bridge() -> "_Bridge":
    b = _BRIDGES.get(threading.get_ident())
    if b is None:
        raise RuntimeError("envkit functions work only inside run_program")
    return b


class _Exit(BaseException):
    """Program-ending control flow (handover / trigger / timeout /
    validate). BaseException so an authored `except Exception` cannot
    swallow the return of control."""

    def __init__(self, result: dict):
        self.result = result


class _Guard(Exception):
    """A tripped guard, with its documented one-line reason."""


class _Bridge:
    """What the authored program gets: bound methods over the env client,
    returning plain dicts."""

    def __init__(self, env: Env, root: Path, deadline: datetime,
                 waiter_id: str | None, validate: bool):
        self.env = env
        self.root = root
        self.deadline = deadline
        self.waiter_id = waiter_id
        self.validate = validate
        self.n_fetches = 0
        self.n_waits = 0
        self._now: datetime | None = None

    def now(self) -> datetime:
        if self._now is None:
            self._now = parse_iso(self.env.call("get_time")["now"])
        return self._now

    def call(self, name: str, args: dict):
        if self.validate:
            raise _Exit({"validate": "ok", "reached": name})
        res = self.env.call(name, **dict(args or {}))
        self.n_fetches += 1
        if CALL_OBSERVER is not None:
            try:
                CALL_OBSERVER(self.waiter_id, name, dict(args or {}), res,
                              iso(self.now()))
            except Exception:
                pass
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
        if until <= self.now():
            raise _Guard("wait(until) must be in the future — sim time "
                         "only moves forward")
        wargs = {"until": iso(until)}
        if self.waiter_id:
            wargs["waiter_id"] = self.waiter_id
        res = self.env.call("sleep", **wargs)
        self.n_waits += 1
        self._now = parse_iso(res["now"])
        if res.get("experiment_over"):
            raise _Exit({"experiment_over": True, "woke_for": "timeout"})
        if res.get("woke_for") == "trigger":
            raise _Exit({"woke_for": "trigger", "trigger": res["trigger"]})
        if self._now >= self.deadline:
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
                f"file and hand over a pointer + summary")
        raise _Exit({"woke_for": "handover", "payload": json.loads(blob)})


# -- envkit ------------------------------------------------------------------------------


def ensure_envkit(env: Env, root: Path) -> Path:
    """`root/envkit.py`: the typed stub of the program-callable tools. If
    the program's directory has none, the sim's published one (GET
    /contract) is written there — one generator, server-side."""
    path = root / "envkit.py"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(env.contract()["envkit_py"], encoding="utf-8")
    return path


def _load_envkit(root: Path) -> None:
    path = root / "envkit.py"
    mod = sys.modules.get("envkit")
    if mod is not None and getattr(mod, "__file__", None) == str(path):
        return
    spec = importlib.util.spec_from_file_location("envkit", path)
    mod = importlib.util.module_from_spec(spec)
    mod._current_bridge = current_bridge  # the stub resolves through this
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


# -- run ------------------------------------------------------------------------------------


def run_program(path, until=None, waiter_id: str | None = None,
                env: Env | None = None, root=None,
                validate: bool = False) -> dict:
    """Run the authored file at `path` forward through sim time until it
    hands over, a scheduled trigger fires (delivered to `waiter_id` /
    the solo caller), or `until` (default: experiment end). `root` (default:
    the file's directory) is the program's private scratch — envkit.
    workspace(); it is put on sys.path for the run. `validate=True`
    dry-runs at zero cost and zero sim time up to the first envkit call."""
    env = env or Env()
    file = Path(path).resolve()
    root = Path(root).resolve() if root else file.parent
    if not file.is_file():
        return {"woke_for": "error", "error": f"no such program: {path!r}"}
    ensure_envkit(env, root)
    time_info = env.call("get_time")
    now = parse_iso(time_info["now"])
    sim_end = parse_iso(time_info["sim_end"])
    if now >= sim_end:
        return {"experiment_over": True, "now": time_info["now"]}
    if validate:
        deadline = now
    elif until is None:
        deadline = sim_end
    else:
        deadline = parse_iso(until) if isinstance(until, str) else until
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        deadline = min(deadline, sim_end)
        if deadline <= now:
            raise ValueError("'until' must be a future datetime — the "
                             "program's hard timeout backstop")
    src = file.read_text(encoding="utf-8")
    try:
        code = compile(src, str(file), "exec")
    except SyntaxError as e:
        return {"now": time_info["now"], "woke_for": "error",
                "error": f"syntax error: {e.msg} ({file.name} line {e.lineno})"}
    bridge = _Bridge(env, root, deadline, waiter_id, validate)
    bridge._now = now
    outcome = _execute(code, bridge, root)
    if validate and "validate" not in outcome:
        if outcome.get("woke_for") == "error":
            outcome = {"validate": "error", "error": outcome["error"]}
        else:
            outcome = {"validate": "ok",
                       "note": "program ran without touching envkit"}
    outcome.setdefault("fetches", bridge.n_fetches)
    outcome.setdefault("waits", bridge.n_waits)
    return {"now": iso(bridge.now()), **outcome}


def _execute(code, bridge: _Bridge, root: Path) -> dict:
    tid = threading.get_ident()
    _BRIDGES[tid] = bridge
    jail = str(root)
    prefix = jail + os.sep
    sys.path.insert(0, jail)
    _load_envkit(root)
    g = {"__name__": "__main__", "__file__": code.co_filename}
    try:
        exec(code, g)
    except _Exit as e:
        return dict(e.result)
    except _Guard as e:
        return {"woke_for": "error", "error": str(e)}
    except SystemExit:
        return {"woke_for": "return",
                "note": "program exited without handover"}
    except BaseException as e:
        return {"woke_for": "error", "error": _format_error(e, prefix)}
    finally:
        del _BRIDGES[tid]
        try:
            sys.path.remove(jail)
        except ValueError:
            pass
        for name, mod in list(sys.modules.items()):
            # purge program modules so edits never go stale in the import
            # cache (concurrent holders keep their references alive)
            if name != "envkit" and \
                    (getattr(mod, "__file__", None) or "").startswith(prefix):
                del sys.modules[name]
    return {"woke_for": "return", "note": "program returned without handover"}
