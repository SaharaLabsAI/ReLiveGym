"""ReplayEnv: the agent's own env client frozen at a past instant — or,
in rollout mode, driven by a virtual clock. Program library (agent-side;
the server is untouched; task-free).

A replay re-runs one past wake of a ReACT agent under a candidate
learned block. The agent must see exactly
what it could have seen at that instant, so every call still goes
through the real, billed client, with task-declared clamps applied on
top of whatever the server clamps:

  get_time                 -> the replay instant t
  wait tools               -> END the episode: the live agent's wake ended
                              where it chose to wait, so the replayed one
                              does too (result carries done=True and the
                              wait it asked for, kept in `waits`);
                              run_program(validate=true) is a dry run, not
                              a wait: passed through (into the replay jail)
  schedule tools           -> refused
  any name in `clamps`     -> clamps[name](args, self): the task's own
                              rule for that tool (date filters, visibility,
                              capturing the scored action instead of
                              sending it — see the task's replay.py)
  file tools (TM-B)        -> with `jail` set, redirected to a replay-owned
                              jail seeded from the live one (FILE_TOOLS
                              below); the live program is never edited
  everything else          -> passed through unchanged

Rollout mode (`end` set): the instant `t`
MOVES. A wait tool advances `t` to min(until, end) and returns the
server's own wake shape ({"now", "woke_for": "sleep"}), so the Agent
appends its wake marker and the episode goes on; reaching `end` returns
{"experiment_over": true} and the loop ends. `run_program` executes the
authored program CLIENT-SIDE (runtime/program.py) over this env: its
envkit.now()/wait() are the virtual clock, its fetches go through the
clamps (billed, pinned to t), its handover comes back as the tool
result — exactly the live shape. The clamps read `t_iso` at call time,
so a task's `as_of` follows the clock.

`captured` collects what the clamps decided to record (the episode's
actions); `calls` logs every call, ok or refused, so a replay is
auditable call by call.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .env_client import EnvError, iso, parse_iso

WAITS = ("sleep", "wait_until", "run_program")
REFUSED = ("set_party", "run_at", "set_crontab", "create_schedule",
           "update_schedule", "delete_schedule")
# The authored-workspace file tools (TM-B). A replayed wake may read and
# edit its program exactly as the live wake could, but never the LIVE
# jail: with `jail` set, these calls are redirected to a replay-owned
# jail (the server provisions one per waiter_id) seeded once with a copy
# of the live jail's files through the same free tools. The copy is the
# jail as it is NOW, not as it was at t — the approximation the bnpm
# tmB replay always made; without this redirect its replayed edits
# would land in the live program.
FILE_TOOLS = ("ls", "read_file", "write_file", "edit_file")
_JAIL_CHARS = re.compile(r"[^A-Za-z0-9_-]")  # the server's waiter_id rule
_COPY_PAGE_LINES = 2000  # read_file's default page; halved on a byte clip


def refusal(name: str, detail: str) -> EnvError:
    """An EnvError worded like the server's own 400 for this tool."""
    return EnvError(f'POST /call/{name} -> 400: {{"detail":"{detail}"}}')


def jail_name(*parts) -> str:
    """A waiter_id-safe jail name for a replayed episode."""
    return _JAIL_CHARS.sub("-", "replay-" + "-".join(str(p) for p in parts))


class ReplayEnv:
    def __init__(self, env, t: datetime, clamps: dict | None = None,
                 refused: tuple = REFUSED, waits: tuple = WAITS, log=None,
                 jail: str | None = None, end: datetime | None = None,
                 fetch_tools: tuple = (), max_fetches: int | None = None):
        self.env = env
        self.t = t
        self.clamps = dict(clamps or {})
        self.refused = tuple(refused)
        self.wait_tools = tuple(waits)
        self.log = log  # callable(event: dict) or None
        self.captured: list[dict] = []  # recorded by the clamps
        self.waits: list[dict] = []  # the wait(s) the agent asked for
        self.store: dict = {}  # scratch for the clamps (e.g. ids seen)
        self.calls: list[dict] = []
        self.llm_calls = 0
        self.jail = jail  # replay-owned jail for the file tools, if any
        self.jail_copy: dict | None = None  # what the seed copy did
        # rollout mode (v3): the clock moves until `end`
        self.end = end
        self.end_reason: str | None = None  # "reached_end" once t hits end
        self.fetch_tools = tuple(fetch_tools)  # billed data views (task)
        self.max_fetches = max_fetches  # per rollout; None = no cap
        self.n_fetches = 0
        self.program_runs: list[dict] = []  # client-side run_program runs
        self._sim_end: str | None = None
        self._jail_dir: Path | None = None
        self._stub_written = False

    @property
    def t_iso(self) -> str:
        return iso(self.t)

    # -- the Env surface --------------------------------------------------------------

    def tools(self) -> list[dict]:
        return self.env.tools()

    def now(self) -> datetime:
        return self.t

    def llm(self, path: str, body: dict) -> dict:
        self.llm_calls += 1
        return self.env.llm(path, body)

    def contract(self) -> dict:
        return self.env.contract()

    def call(self, name: str, **args):
        try:
            out = self._dispatch(name, args)
        except EnvError as e:
            self._log(name, args, ok=False, error=str(e))
            raise
        self._log(name, args, ok=True)
        return out

    def passthrough(self, name: str, **args):
        """The real call, for clamps that forward after adjusting args."""
        return self.env.call(name, **args)

    # -- dispatch -----------------------------------------------------------------------

    def _dispatch(self, name: str, args: dict):
        if name == "get_time":
            if self.end is None:
                return {"now": self.t_iso}
            return {"now": self.t_iso, "sim_end": self._real_sim_end()}
        if name == "run_program" and args.get("validate"):
            # a dry run: zero sim time, zero cost, no wait — the live
            # actor validates and keeps working, so the replayed one does
            # too (in the replay jail, whose program it may have edited)
            if self.jail:
                self._seed_jail(args.get("waiter_id") or "actor")
                args = {**args, "waiter_id": self.jail}
            return self.env.call(name, **args)
        if name in self.wait_tools:
            if self.end is None:
                self.waits.append(args)
                return {"done": True, "replay_wait": args,
                        "note": "replayed wake ends at its wait"}
            return self._rollout_wait(name, args)
        if name in self.refused:
            raise refusal(name, f"{name} is not available in a replayed wake")
        if name in self.fetch_tools and self.max_fetches is not None:
            self.n_fetches += 1
            if self.n_fetches > self.max_fetches:
                self.end_reason = self.end_reason or "fetch_cap"
                raise refusal(name, f"replay fetch cap reached: "
                                    f"{self.max_fetches} data calls per rollout")
        clamp = self.clamps.get(name)
        if clamp is not None:
            return clamp(args, self)
        if name in FILE_TOOLS and self.jail:
            self._seed_jail(args.get("waiter_id") or "actor")
            return self.env.call(name, **{**args, "waiter_id": self.jail})
        return self.env.call(name, **args)

    # -- the virtual clock (rollout mode) --------------------------------------------------

    def advance(self, until: datetime) -> dict:
        """Move the clock to min(until, end), in the server's wake shape.
        A wait into the past does not move it (as the server); reaching
        `end` ends the episode: {"experiment_over": true}."""
        if until <= self.t:
            return {"now": self.t_iso, "woke_for": "sleep"}
        self.t = min(until, self.end)
        if self.t >= self.end:
            self.end_reason = self.end_reason or "reached_end"
            return {"experiment_over": True, "now": self.t_iso}
        self.waits.append({"until": iso(until)})
        return {"now": self.t_iso, "woke_for": "sleep"}

    def _rollout_wait(self, name: str, args: dict) -> dict:
        if name == "run_program":
            return self._run_program(args)
        raw = args.get("until")
        try:
            until = parse_iso(str(raw))
        except ValueError:
            raise refusal(name, f"until must be an iso datetime, got {raw!r}")
        return self.advance(until)

    def _real_sim_end(self) -> str:
        if self._sim_end is None:
            self._sim_end = self.env.call("get_time")["sim_end"]
        return self._sim_end

    def _run_program(self, args: dict) -> dict:
        """The authored program, run client-side over THIS env (its
        clock, clamps and jail) — runtime/program.py's bridge is the
        same contract the server's ProgramApp implements."""
        from . import program

        path = args.get("path")
        if not isinstance(path, str) or not path:
            raise refusal("run_program", f"no such program: {path!r} "
                                         "(ls your workspace)")
        root = self.jail_dir(args.get("waiter_id") or "actor")
        try:
            file = (root / path).resolve()
            file.relative_to(root.resolve())
        except ValueError:
            raise refusal("run_program", f"no such program: {path!r} "
                                         "(ls your workspace)")
        if not file.is_file():
            raise refusal("run_program", f"no such program: {path!r} "
                                         "(ls your workspace)")
        until = args.get("until")
        if until is not None:
            try:
                u = parse_iso(str(until))
            except ValueError:
                raise refusal("run_program", f"until must be an iso datetime, "
                                             f"got {until!r}")
            if u <= self.t:
                raise refusal("run_program", "'until' must be a future iso "
                              "datetime — the program's hard timeout backstop")
        t0 = self.t_iso
        try:
            out = program.run_program(str(file), until=until, waiter_id=self.jail,
                                      env=self, root=root)
        except ValueError as e:
            raise refusal("run_program", str(e))
        self.program_runs.append({"path": path, "until": until, "from": t0,
                                  "to": self.t_iso,
                                  "woke_for": out.get("woke_for"),
                                  "experiment_over": bool(out.get("experiment_over")),
                                  "fetches": out.get("fetches"),
                                  "waits": out.get("waits")})
        return out

    def jail_dir(self, live: str) -> Path:
        """The replay jail's directory on disk (the run's workspace is the
        cwd; the server's `ls` reports the jail root relative to it),
        seeded from the live jail and carrying the CLIENT-side envkit
        stub (the server renders one that resolves through the env
        process; a program run here resolves through runtime.program)."""
        if self._jail_dir is None:
            self._seed_jail(live)
            listing = self.env.call("ls", waiter_id=self.jail)
            self._jail_dir = Path(listing["root"])
        if not self._stub_written:
            self._jail_dir.mkdir(parents=True, exist_ok=True)
            (self._jail_dir / "envkit.py").write_text(
                self.env.contract()["envkit_py"], encoding="utf-8")
            self._stub_written = True
        return self._jail_dir

    # -- the replay jail ----------------------------------------------------------------

    def _seed_jail(self, live: str) -> None:
        """Copy the live jail's files into the replay jail, once per
        episode, through the free file tools (a page at a time; a single
        line over the read clip cannot be copied whole and is reported)."""
        if self.jail_copy is not None:
            return
        self.jail_copy = {"from": live, "files": 0, "lossy": []}
        listing = self.env.call("ls", waiter_id=live)
        for f in listing.get("files", []):
            path = f["path"]
            if path == "envkit.py":  # rendered per jail by the server
                continue
            content, lossy = self._read_whole(live, path)
            self.env.call("write_file", waiter_id=self.jail, path=path,
                          content=content)
            self.jail_copy["files"] += 1
            if lossy:
                self.jail_copy["lossy"].append(path)

    def _read_whole(self, wid: str, path: str) -> tuple[str, bool]:
        parts: list[str] = []
        offset, limit, lossy = 1, _COPY_PAGE_LINES, False
        while True:
            page = self.env.call("read_file", waiter_id=wid, path=path,
                                 offset=offset, limit=limit)
            if page.get("clipped") and limit > 1:
                limit = max(1, limit // 2)  # retry the page smaller
                continue
            if page.get("clipped"):
                lossy = True  # one line over the clip: head only
            parts.append(page.get("content", ""))
            if not page.get("truncated") or not page.get("lines"):
                return "".join(parts), lossy
            offset += page["lines"]
            limit = _COPY_PAGE_LINES

    def _log(self, name: str, args: dict, **fields) -> None:
        row = {"tool": name, "args": args, **fields}
        self.calls.append(row)
        if self.log:
            self.log(row)

