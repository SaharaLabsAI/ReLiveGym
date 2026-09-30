"""Simulated clock.

Time is frozen while agent code executes; only the supervisor (between agent
processes) and the /sleep handler (while exactly one runs) advance it. Both go
through `advance_to`, which enforces monotonicity and clamps at the horizon.

The horizon is `sim_end`, or an earlier `pause_at`: a paused clock reads as
finished to every handler — the actor sees the same `experiment_over` it
sees at the real end — while the server writes a checkpoint instead of
results (harness/checkpoint.py). The clamp is the whole mechanism: nothing
in the sim advances past it, so the world state at the cut is exactly the
state a continuing run would have had there.
"""

from __future__ import annotations

from datetime import datetime


class SimClock:
    def __init__(self, sim_start: datetime, sim_end: datetime,
                 pause_at: datetime | None = None):
        if pause_at is not None and not (sim_start < pause_at < sim_end):
            raise ValueError(
                f"pause_at must lie strictly inside the run window "
                f"({sim_start} .. {sim_end}), got {pause_at}")
        self._now = sim_start
        self.sim_end = sim_end
        self.pause_at = pause_at

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def horizon(self) -> datetime:
        """Where the clock stops: sim_end, or the pause instant."""
        return self.sim_end if self.pause_at is None else min(self.sim_end, self.pause_at)

    @property
    def finished(self) -> bool:
        return self._now >= self.horizon

    @property
    def paused(self) -> bool:
        """At the pause instant (and not at the real end)."""
        return self.pause_at is not None and self.pause_at <= self._now < self.sim_end

    def advance_to(self, t: datetime) -> datetime:
        """Advance to t (clamped to the horizon). Returns the new now."""
        if t < self._now:
            raise ValueError(f"time cannot go backward: {t} < {self._now}")
        self._now = min(t, self.horizon)
        return self._now
