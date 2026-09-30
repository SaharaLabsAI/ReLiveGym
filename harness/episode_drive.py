"""The cron driver of episode mode: tm=C/D for an external app.

    python -m harness.mcp drive --episode <episode.json>

The constructor's cron arms are two halves. The server half — ScheduleStore,
AgentScheduleApp, `Scheduler.next` advancing the clock to the next due
trigger — runs inside every `harness.serve` and is reused untouched. The
process half is the runner loop of scaffolds/runtime/actor.py: `GET
/trigger/next` -> spawn -> real-time watchdog -> `POST /trigger/<id>/exit`
until the server says done. This driver IS that loop (a `Runner`
subclass), with the spawn replaced: one fenced `opencode run` per firing,
continuing one session for the whole run — the cron main's persistent
transcript — and prompted with the wake marker the cron main writes into it.

It runs on the controller side: outside both fences, holding the
controller token from episode.json, which the actor cannot read. The
agent's half of the contract is the `env` MCP bridge and nothing else.

Artefacts, next to episode.json:
  firings.jsonl      one row per firing: trigger, sim/real times, exit,
                     session id, the byte range of the firing's events
                     in opencode_events.jsonl
  drive_state.json   {session_id} — what a restarted driver needs
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.mcp import ACTOR_PROMPT, load_episode
from scaffolds.runtime.actor import Runner, watchdog_verdict
from scaffolds.runtime.agent import wake_marker

# the proxy flags the run on the first refused LLM call (llm_proxy.py)
BUDGET_DEAD_FLAGS = frozenset({"budget_exhausted", "llm_budget_exhausted"})
KILL_GRACE_SECONDS = 5.0
STDERR_TAIL_BYTES = 4096


def wake_message(trig: dict, opens_session: bool) -> str:
    """The prompt of one firing: the cron main's wake marker (note and daily cost
    brief included, same rules), plus the fixed A/B actor prompt on the
    firing that opens the session."""
    now = datetime.fromisoformat(trig["now"].replace("Z", "+00:00"))
    msg = wake_marker(now.isoformat(), trig)
    if opens_session:
        msg += "\n" + ACTOR_PROMPT
    return msg


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _slice(path: Path, start: int) -> bytes:
    if not path.exists():
        return b""
    with open(path, "rb") as f:
        f.seek(start)
        return f.read()


def session_id_in(events: bytes) -> str | None:
    """The session of a firing, from its `--format json` events (every
    event carries `sessionID`)."""
    for line in events.splitlines():
        try:
            sid = json.loads(line).get("sessionID")
        except (ValueError, AttributeError):
            continue
        if sid:
            return sid
    return None


class EpisodeRunner(Runner):
    def __init__(self, ep: dict, quiet: bool = False):
        self.dir = Path(ep["_path"]).parent
        super().__init__(ep["env_url"], ep["token"], ep.get("model") or "",
                         Path(ep["workspace"]), self.dir, quiet=quiet)
        self.script = Path(ep["firing_script"])
        self.events = self.dir / "opencode_events.jsonl"
        self.stderr = self.dir / "opencode_stderr.log"
        self.firings = self.dir / "firings.jsonl"
        self.state = self.dir / "drive_state.json"
        self.session_id: str | None = None
        if self.state.exists():
            self.session_id = json.loads(self.state.read_text()).get("session_id")

    # -- Runner hooks -------------------------------------------------------------------

    def prepare(self) -> dict:
        """The workspace was written by `harness.mcp prepare`. A driver
        restarted mid-firing closes the orphaned invocation as a kill, so
        the next trigger arrives as crash_recovery."""
        (self.workspace / "logs").mkdir(exist_ok=True)
        active = self.api.get("/status")["active_trigger"]
        if active is not None:
            self.log(f"closing orphaned invocation {active}")
            self.api.post(f"/trigger/{active}/exit", {
                "code": 1, "killed": True, "output": "driver restarted"})
        return {}

    # the workspace holds no program: code_change and rollback stay inert
    # even if the agent writes a main.py or runs git in it
    def _code(self):
        return None, None

    def _head(self):
        return None

    def _spawn(self, trig: dict) -> tuple[int, bool, str]:
        if self._budget_dead():
            # an agent without LLM calls can do nothing; with no floor on
            # agent crons, booting Chrome + the app for each remaining
            # firing would cost real days. The ledger still gets the
            # trigger and its agent_exit. (The firing the budget dies IN
            # needs no rule: OpenCode retries the refused call for about
            # a minute, logs an `error` event and exits 0, as the
            # constructor's agents exit 0 on a budget refusal.)
            self._record(trig, skipped=True, code=0, killed=False)
            return 0, False, "not launched: the LLM budget is spent"
        opens = self.session_id is None
        start, err_start = _size(self.events), _size(self.stderr)
        real_start = time.time()
        code, killed, out = self._launch(wake_message(trig, opens))
        if opens:
            self.session_id = session_id_in(_slice(self.events, start))
            self.state.write_text(json.dumps({"session_id": self.session_id}))
        self._record(trig, code=code, killed=killed, real_start=real_start,
                     events_offset=[start, _size(self.events)])
        tail = _slice(self.stderr, err_start)[-STDERR_TAIL_BYTES:]
        return code, killed, out + tail.decode(errors="replace")

    # -- one firing ---------------------------------------------------------------------

    def _budget_dead(self) -> bool:
        return bool(BUDGET_DEAD_FLAGS & set(self.api.get("/status")["flags"]))

    def _launch(self, message: str) -> tuple[int, bool, str]:
        """Run the firing script in its own process group (the script, the
        two fences, the app, the bridge, the browser server) under the
        run's real-time watchdog."""
        env = {**os.environ, "ENV_WAKE_MESSAGE": message,
               "ENV_OPENCODE_SESSION": self.session_id or ""}
        proc = subprocess.Popen(["sh", str(self.script)], env=env,
                                start_new_session=True,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        killed = False
        poll = 0.1
        try:
            while proc.poll() is None:
                time.sleep(poll)
                act = self.api.get("/activity")
                wd = float(act["watchdog_seconds"])
                self.watchdog_seconds = wd
                poll = max(0.05, min(0.5, wd / 4))
                verdict = watchdog_verdict(act)
                if verdict is not None and proc.poll() is None:
                    self._kill_reason = verdict
                    self._kill(proc)
                    killed = True
        finally:
            # a stopped driver (Ctrl-C, SIGTERM, a dead server) takes its
            # firing with it: the firing is a session of its own, so no
            # terminal signal reaches it
            if proc.poll() is None:
                self._kill(proc)
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        return proc.wait(), killed, out

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        # TERM first: the script's trap stops the browser server, which
        # closes Chrome (a process group of its own)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=KILL_GRACE_SECONDS)
                return
            except subprocess.TimeoutExpired:
                continue

    def _record(self, trig: dict, **row) -> None:
        row = {"trigger": {k: trig[k] for k in ("id", "kind", "due_time", "owner")
                           if k in trig},
               "sim_now": trig["now"], "session_id": self.session_id,
               "real_end": time.time(), **row}
        if row.get("killed"):
            row["kill_reason"] = self._kill_reason
        with open(self.firings, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def drive(episode_json: Path) -> int:
    ep = load_episode(episode_json)
    if ep.get("driver") != "cron":
        raise SystemExit("not a cron episode: `drive` fires tm=C/D episodes "
                         "(prepare --tm C|D)")
    path = Path(ep["_path"])

    def mark(pid: int | None) -> None:
        data = json.loads(path.read_text())
        data["driver_pid"] = pid
        if pid is not None:
            data["driver_started_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=1))

    mark(os.getpid())
    # SIGTERM as an exception, so the `finally`s (kill the firing, clear
    # driver_pid) run for a batch launcher's stop as for Ctrl-C
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        result = EpisodeRunner(ep).run()
    finally:
        mark(None)
    print(f"\nDone at {result.get('now')}. Report: python -m harness.mcp "
          f"settle --episode {path}")
    return 0
