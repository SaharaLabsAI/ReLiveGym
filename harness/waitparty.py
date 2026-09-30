"""Wait party: k concurrent waiters in one agent process.

A discrete-event barrier over the sim clock. With a party declared
(set_party), sleep calls park — their HTTP requests await an asyncio
future — until every roster member is parked; then the earliest deadline
across all waiters (or a due scheduled trigger) advances the clock ONCE
and wakes exactly the waiters whose deadline lands there.

The clock invariant (harness/clock.py) is preserved: time stays frozen
while any agent computes, and while the process lives the barrier is
the one advancing site. Party mode is opt-in per process: sim.party is
None (solo) unless the program calls set_party, the solo handler paths
in harness/apps.py are untouched, and the supervisor aborts + clears
the party when the process exits.

Sleeps book nothing (matching the solo handler). Scheduled triggers are
delivered only to the declared trigger_waiter. There is no declarative
condition grammar — event-driven waiting is authored gatekeeper code
(harness/authored.py), whose fetches
bill like any other call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from harness.env_tools import ToolError
from harness.timeutil import iso

if TYPE_CHECKING:
    from harness.runtime import Sim


@dataclass
class _Parked:
    waiter_id: str
    until: datetime
    armed_at: datetime
    future: asyncio.Future
    brief: bool = True  # carries the daily cost brief (an agent wake)


class WaitParty:
    """Roster + parked state + the barrier. Every method expects
    sim.lock held by the caller, except `wait`, which takes the lock
    itself and awaits its future outside it."""

    def __init__(self, sim: "Sim"):
        self.sim = sim
        self.roster: list[str] = []
        self.trigger_waiter: str | None = None
        self.parked: dict[str, _Parked] = {}

    # -- roster (set_party handler) -----------------------------------------------

    def set_roster(self, waiter_ids: list[str], trigger_waiter: str) -> None:
        if len(set(waiter_ids)) != len(waiter_ids):
            raise ToolError("set_party: duplicate waiter ids")
        if trigger_waiter not in waiter_ids:
            raise ToolError("set_party: trigger_waiter must be in waiter_ids")
        gone = sorted(w for w in self.parked if w not in waiter_ids)
        if gone:
            raise ToolError(f"set_party: cannot remove parked waiter(s) {gone}"
                            " (a waiter leaves the party from its own turn, "
                            "then stops waiting)")
        self.roster = list(waiter_ids)
        self.trigger_waiter = trigger_waiter
        # a shrink can complete the barrier for the waiters still parked
        self._maybe_advance()

    # -- one wait call --------------------------------------------------------------

    async def wait(self, waiter_id, until: datetime,
                   brief: bool = True) -> dict:
        sim = self.sim
        async with sim.lock:
            if not isinstance(waiter_id, str) or waiter_id not in self.roster:
                raise ToolError(
                    f"unknown waiter_id {waiter_id!r} (declare the roster "
                    f"with set_party before waiting)")
            if waiter_id in self.parked:
                raise ToolError(f"waiter {waiter_id!r} is already parked")
            if sim.clock.finished:
                return {"experiment_over": True, "now": iso(sim.clock.now)}
            now = sim.clock.now
            if until <= now:
                # a past-deadline waiter never parks (it would stall the
                # barrier); solo semantics: the sleep just returns
                return {"now": iso(now), "woke_for": "sleep"}
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self.parked[waiter_id] = _Parked(waiter_id, until, now, fut,
                                             brief)
            self._maybe_advance()
        return await fut

    # -- the barrier ----------------------------------------------------------------

    def _maybe_advance(self) -> None:
        sim = self.sim
        if not self.roster or len(self.parked) < len(self.roster):
            return
        now = sim.clock.now
        sim_end = sim.cfg.sim_end
        deadlines = {w.waiter_id: min(w.until, sim_end)
                     for w in self.parked.values()}
        # real triggers only: the midnight fallback never ends a wait
        trig = sim.schedule.peek_due(now)
        wake = min(min(deadlines.values()), sim_end)
        if trig is not None:
            wake = min(wake, trig.due_time)
        if wake < now:
            # unreachable by the min rule (an earlier event would have won
            # an earlier barrier); a hole here would wedge the clock
            raise RuntimeError(
                f"wait party computed a wake in the past: {wake} < {now}")
        if wake > now:
            sim.clock.advance_to(wake)
            sim.book_due_outcomes()
            now = sim.clock.now
        if sim.clock.finished:
            for w in self._parked_sorted():
                self._resolve(w, {"experiment_over": True, "now": iso(now)})
            return
        if trig is not None and trig.due_time == wake:
            tw = self.parked[self.trigger_waiter]
            sim.schedule.consume(trig)
            sim.ledger.append("trigger", now, id=trig.id, kind=trig.kind,
                              via="sleep")
            self._resolve(tw, {"now": iso(now), "woke_for": "trigger",
                               "trigger": trig.payload()}, brief=True)
        for w in self._parked_sorted():
            if min(w.until, sim_end) <= wake:
                self._resolve(w, {"now": iso(now), "woke_for": "sleep"},
                              brief=True)

    def _parked_sorted(self) -> list[_Parked]:
        # deterministic resolution (and ledger) order at equal instants
        return sorted(self.parked.values(), key=lambda p: p.waiter_id)

    def _resolve(self, w: _Parked, payload: dict, brief: bool = False) -> None:
        from harness.apps import log_sleep

        self.parked.pop(w.waiter_id, None)
        if not w.future.done():
            log_sleep(self.sim, w.waiter_id, w.until, w.armed_at,
                      "experiment_over" if payload.get("experiment_over")
                      else "aborted" if payload.get("aborted")
                      else payload.get("woke_for", "sleep"))
            if brief and w.brief:  # after log_sleep: same ledger order as the solo
                payload = {**payload,  # sleep handler (brief follows sleep)
                           **_costs(self.sim, self.sim.clock.now)}
            w.future.set_result(payload)

    # -- process exit (supervisor) --------------------------------------------------

    def abort(self, now: datetime) -> None:
        """Process exited/killed with waiters still parked: release the
        dangling requests."""
        for w in self._parked_sorted():
            self._resolve(w, {"aborted": True, "now": iso(now)})
        self.roster = []
        self.trigger_waiter = None


def _costs(sim, now) -> dict:
    """`{"costs": brief}` on the first wake of a sim date, else `{}`."""
    brief = sim.daily_brief(now)
    return {"costs": brief} if brief is not None else {}
