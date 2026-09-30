"""Scheduler: the server half of the program contract's lifecycle
(formerly the whole Supervisor).

The lifecycle is split along the line "who owns sim time" vs "who owns
the process":

  server side (this module)              actor side (scaffolds/runtime/actor.py)
  ---------------------------------      ----------------------------------------
  advance the clock to the next due      spawn `python main.py` in the workspace
    trigger; consume; book outcomes        with ENV_URL/ENV_TOKEN/ENV_TRIGGER
  ledger: trigger, agent_exit,           real-time watchdog (polls GET /activity,
    code_change, rollback,                 kills the child)
    failed_degenerate                    crash logs (workspace/logs/crash-*.log),
  crash streak, crash_recovery kind,       code_history/ snapshots, the git
    last-good sha, rollback DECISION       reset of a rollback

The runner drives it over two routes: `GET /trigger/next` (advance and
hand out the next trigger, or `{done: true}` at sim_end / failed-
degenerate) and `POST /trigger/<id>/exit` (the invocation's outcome). The
clock advances only between invocations — `next` refuses while one is
active — so the single-process invariant of harness/runtime.py holds
whether the runner is a thread of the same process (harness.run, the
combined mode) or a program on another machine (harness.serve).

Crash handling (unchanged semantics): non-zero exit or watchdog kill
delivers the *next* trigger with kind=crash_recovery; K consecutive
crashed invocations without a single successful helper interaction mark
the run failed-degenerate; the clock still runs to sim_end so misses
settle honestly. Alg-C/D rollback: after K_ROLLBACK
consecutive crashes with the workspace HEAD ahead of the last good sha,
the exit response carries `rollback_to` and the runner hard-resets — a
crashed self-edit costs time, not the run.
"""

from __future__ import annotations

import time

from harness.runtime import Sim
from harness.schedule import Trigger
from harness.timeutil import iso

K_ROLLBACK = 2  # consecutive crashes after a self-edit before rolling back
OUTPUT_TAIL_BYTES = 1 << 20  # agent_output.log keeps at most this per exit


def _owner_detail(trig: Trigger) -> dict:
    """Extra `trigger` ledger fields for agent-owned schedules (TM-D):
    owner + target, so analysis separates agent-chosen wakes from the
    base cadence. Empty for system triggers — pre-TM-D events unchanged."""
    if trig.owner == "system":
        return {}
    d = {"owner": trig.owner}
    if trig.target is not None:
        d["target"] = trig.target
    return d


class SchedulerError(Exception):
    """A lifecycle request that contradicts the protocol (HTTP 409)."""


class Scheduler:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.active: Trigger | None = None
        self.done = False
        self._consecutive_crashes = 0
        self._crashed_last = False
        self._last_code_hash: str | None = None
        self._last_good_sha: str | None = None
        self._first_call = True
        self.paused = False  # finished at a pause clamp (checkpoint, not results)

    def seed_code_hash(self, sha: str | None) -> None:
        """A resumed run's last known main.py digest (harness/checkpoint.py),
        so the first trigger of the new stage ledgers `code_change` when
        the program changed at the cut."""
        self._last_code_hash = sha

    # -- GET /trigger/next ------------------------------------------------------------

    async def next(self, code_sha: str | None = None,
                   head_sha: str | None = None) -> dict:
        """Advance the clock to the next due trigger and hand it out, or
        `{done: true}`. `code_sha` is the runner's hash of main.py (ledgers
        `code_change`); `head_sha` its git HEAD (seeds last-good on the
        first call). Caller must have no invocation active."""
        sim = self.sim
        async with sim.lock:
            if self.active is not None:
                raise SchedulerError(
                    f"invocation for trigger {self.active.id!r} still "
                    "active: report its exit first")
            if self._first_call:
                self._first_call = False
                self._last_good_sha = head_sha
            if self.done:
                return {"done": True, "now": iso(sim.clock.now)}
            if sim.clock.finished or "failed-degenerate" in sim.flags:
                return self._finish()
            trig = sim.schedule.peek_next(sim.clock.now)
            if trig.due_time >= sim.clock.horizon:
                # a trigger due at or past the horizon is never consumed
                # here: past sim_end it is moot, at a pause instant it
                # belongs to the next stage (the restored schedule fires
                # it on resume)
                sim.clock.advance_to(sim.clock.horizon)
                return self._finish()
            sim.clock.advance_to(trig.due_time)
            sim.schedule.consume(trig)
            sim.book_due_outcomes()
            if self._crashed_last:
                trig = Trigger(trig.id, "crash_recovery", trig.due_time,
                               owner=trig.owner, note=trig.note,
                               target=trig.target)
            if code_sha is not None and code_sha != self._last_code_hash:
                if self._last_code_hash is not None:
                    sim.ledger.append("code_change", sim.clock.now,
                                      sha256_12=code_sha)
                self._last_code_hash = code_sha
            sim.ledger.append("trigger", sim.clock.now, id=trig.id,
                              kind=trig.kind, **_owner_detail(trig))
            sim.api_calls_this_run = 0
            sim.last_activity = time.monotonic()
            self.active = trig
            out = {**trig.payload(), "now": iso(sim.clock.now)}
            # a `learn` firing wakes no actor agent: the date's brief waits
            # for the next trigger that does
            brief = (sim.daily_brief(sim.clock.now)
                     if trig.id != "learn" else None)
            if brief is not None:
                out["costs"] = brief
            return out

    def _finish(self) -> dict:
        """End of run (caller holds sim.lock): close whatever the grace
        window left open. Idempotent. Under a pause clamp
        (clock.pause_at; harness/checkpoint.py) nothing is settled: the
        run stops at the cut with every pending claim still pending, and
        the server writes a checkpoint instead of results."""
        sim = self.sim
        if not self.done:
            sim.clock.advance_to(sim.clock.horizon)
            if sim.clock.paused:
                self.paused = True
            else:
                sim.book_all_outcomes()
            self.done = True
        out = {"done": True, "now": iso(sim.clock.now)}
        if self.paused:
            out["paused"] = True
        return out

    # -- POST /trigger/<id>/exit -----------------------------------------------------

    async def report_exit(self, trigger_id: str, code: int, killed: bool,
                          output: str, head_sha: str | None = None) -> dict:
        """The invocation ended. Ledgers agent_exit, aborts a dangling wait
        party, updates the crash streak / flags, decides a rollback.
        Returns {now, crashed, rollback_to?, failed_degenerate}."""
        sim = self.sim
        async with sim.lock:
            if self.active is None or self.active.id != trigger_id:
                raise SchedulerError(
                    f"no active invocation for trigger {trigger_id!r}")
            trig, self.active = self.active, None
            if sim.party is not None:
                # release dangling parked requests — a crash must not leave
                # the clock hostage (waitparty.py); the next spawn starts
                # party-less
                sim.party.abort(sim.clock.now)
                sim.party = None
            self._log_invocation(trig, code, killed, output)

            crashed = killed or code != 0
            resp: dict = {"now": iso(sim.clock.now), "crashed": crashed,
                          "failed_degenerate": False}
            if crashed:
                self._crashed_last = True
                if sim.api_calls_this_run > 0:
                    self._consecutive_crashes = 1
                else:
                    self._consecutive_crashes += 1
                target = self._rollback_target(head_sha)
                if target is not None:
                    resp["rollback_to"] = target
                if self._consecutive_crashes >= sim.cfg.max_consecutive_crashes:
                    sim.flags.append("failed-degenerate")
                    sim.ledger.append("failed_degenerate", sim.clock.now,
                                      consecutive_crashes=self._consecutive_crashes)
                    resp["failed_degenerate"] = True
            else:
                self._crashed_last = False
                self._consecutive_crashes = 0
                self._last_good_sha = head_sha or self._last_good_sha
            return resp

    # -- Alg-C/D rollback ----------------------------------------------

    def _rollback_target(self, head: str | None) -> str | None:
        """Decide (never execute) a rollback: after K_ROLLBACK consecutive
        crashes with HEAD ahead of last-good, ledger it, reset the streak
        (the rolled-back code gets its chance) and return the target sha.
        Caller holds sim.lock."""
        sim = self.sim
        if self._consecutive_crashes < K_ROLLBACK or self._last_good_sha is None:
            return None
        if head is None or head == self._last_good_sha:
            return None  # nothing to roll back to / already there
        sim.ledger.append("rollback", sim.clock.now,
                          from_sha=head[:12], to_sha=self._last_good_sha[:12],
                          consecutive_crashes=self._consecutive_crashes)
        self._consecutive_crashes = 0
        return self._last_good_sha

    # -- logging ------------------------------------------------------------------------

    def _log_invocation(self, trig: Trigger, code: int,
                        killed: bool, output: str) -> None:
        sim = self.sim
        outcome = "watchdog_killed" if killed else f"exit={code}"
        sim.ledger.append("agent_exit", sim.clock.now, trigger_id=trig.id,
                          outcome=outcome)
        log = sim.run_dir / "agent_output.log"
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"===== {iso(sim.clock.now)} trigger={trig.id} "
                    f"kind={trig.kind} {outcome} =====\n")
            f.write(output)
            if output and not output.endswith("\n"):
                f.write("\n")

    # -- GET /activity ------------------------------------------------------------------

    def activity(self) -> dict:
        sim = self.sim
        return {"idle_seconds": time.monotonic() - sim.last_activity,
                "llm_inflight": sim.llm_inflight,
                "oldest_llm_inflight_seconds": sim.oldest_llm_inflight_seconds(),
                "llm_timeout_seconds": sim.llm_allowance_seconds(),  # all attempts
                "watchdog_seconds": sim.cfg.watchdog_seconds,
                "active_trigger": self.active.id if self.active else None}

    # -- GET /status --------------------------------------------------------------------

    def status(self) -> dict:
        sim = self.sim
        return {"run_id": sim.cfg.run_id,
                "sim_now": iso(sim.clock.now),
                "sim_start": iso(sim.cfg.sim_start),
                "sim_end": iso(sim.cfg.sim_end),
                "done": self.done,
                "paused": self.paused,
                "pause_at": (iso(sim.clock.pause_at)
                             if sim.clock.pause_at is not None else None),
                "active_trigger": self.active.id if self.active else None,
                "flags": list(sim.flags),
                "spent_usd": sim.ledger.total_cost(),
                "budget_usd": sim.cfg.budget_usd,
                "hosts": dict(sim.hosts),  # web hosts (harness/web.py)
                # party view for the episode stall watchdog ; absent in solo runs
                **({"party": {"roster": list(sim.party.roster),
                              "parked": sorted(sim.party.parked),
                              "trigger_waiter": sim.party.trigger_waiter}}
                   if sim.party is not None and sim.party.roster else {})}
