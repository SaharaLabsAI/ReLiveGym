"""Rate-limit machines, sim-time driven.

The window algorithms were lifted from the crypto task's replay proxy
(tasks/crypto_price_consistency/incidents/proxy.py), which pioneered
documented-real limits; that proxy now imports them from here and layers
its venue-specific realism (ban escalation, capacity modulation) on top.

Two layers:

- `LimiterState` + `consume(spec, state, now, weight, factor)` — the pure
  machines (fixed_window, sliding_window, token_bucket). `spec` is a dict
  in the limiters.yaml shape; `now` is a unix timestamp of SIM time.
- `RateLimiter` — the harness-facing wrapper a task app registers under
  `sim.limiters[name]`: consume() raises `RateLimited` (HTTP 429) on
  excess and keeps allowed/rejected counters for results.resources.

The limit is advertised to the agent in INSTRUCTION.md (doc()); there is
deliberately NO usage/remaining endpoint — agents track their own
consumption. Retry-After is sent only where the spec opts in
(`retry_after_header`), mirroring what the real API documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class RateLimited(Exception):
    """Tool call rejected by a rate limit. Maps to HTTP 429 in the API
    layer; `retry_after_s` becomes a Retry-After header only when
    `send_header` is set."""

    def __init__(self, message: str, retry_after_s: float | None = None,
                 send_header: bool = False):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.send_header = send_header


@dataclass
class LimiterState:
    window_start: float | None = None
    used: float = 0.0
    stamps: list[tuple[float, float]] = field(default_factory=list)  # sliding
    tokens: float | None = None
    last_refill: float | None = None
    retry_until: float | None = None   # active 429 Retry-After deadline
    ban_until: float | None = None
    violations: int = 0
    last_violation: float | None = None


def consume(spec: dict[str, Any], state: LimiterState, now: float,
            weight: float, factor: float = 1.0) -> tuple[bool, float | None]:
    """Try to consume `weight` at unix-seconds `now`. Returns
    (allowed, retry_after_s). `factor` scales the budget (capacity
    modulation; 1.0 = documented budget)."""
    kind = spec["window"]
    if kind == "none":  # no limit: every request allowed, nothing tracked
        return True, None
    if kind == "token_bucket":
        burst = spec["burst"] * factor
        if state.tokens is None:
            state.tokens, state.last_refill = burst, now
        state.tokens = min(
            burst, state.tokens
            + (now - state.last_refill) * spec["refill_per_second"] * factor)
        state.last_refill = now
        if state.tokens >= weight:
            state.tokens -= weight
            return True, None
        deficit = weight - state.tokens
        refill = max(spec["refill_per_second"] * factor, 1e-9)
        return False, deficit / refill
    window = spec["window_seconds"]
    budget = spec["budget"] * factor
    if kind == "fixed_window":
        bucket = now // window
        if state.window_start != bucket:
            state.window_start, state.used = bucket, 0.0
        if state.used + weight <= budget:
            state.used += weight
            return True, None
        return False, (bucket + 1) * window - now
    if kind == "sliding_window":
        state.stamps = [(ts, w) for ts, w in state.stamps
                        if ts > now - window]
        if sum(w for _, w in state.stamps) + weight <= budget:
            state.stamps.append((now, weight))
            return True, None
        if not state.stamps:  # budget modulated below one request
            return False, float(window)
        oldest = min(ts for ts, _ in state.stamps)
        return False, oldest + window - now
    raise ValueError(f"unknown limiter window kind {kind!r}")


class RateLimiter:
    """One advertised, env-enforced rate limit. Register under
    sim.limiters[name] so results.resources can report consumption."""

    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.spec = spec
        self.state = LimiterState()
        self.n_allowed = 0
        self.n_rejected = 0

    def consume(self, now: datetime, weight: float | None = None) -> None:
        w = weight if weight is not None else self.spec.get("request_weight", 1)
        allowed, retry_after = consume(self.spec, self.state,
                                       now.timestamp(), w)
        if allowed:
            self.n_allowed += 1
            return
        self.n_rejected += 1
        raise RateLimited(
            f"rate limit exceeded for {self.name}: {self.doc()}",
            retry_after_s=retry_after,
            send_header=bool(self.spec.get("retry_after_header")))

    def doc(self) -> str:
        """One-line human-readable limit, for INSTRUCTION.md tables."""
        kind = self.spec["window"]
        if kind == "none":
            return "none (unlimited)"
        if kind == "token_bucket":
            return (f"{self.spec['refill_per_second']:g} requests/s "
                    f"(burst {self.spec['burst']:g})")
        w = self.spec["window_seconds"]
        span = (f"{w:g} s" if w < 60 else f"{w / 60:g} min" if w < 3600
                else f"{w / 3600:g} h")
        return f"{self.spec['budget']:g} requests per {span}"

    def stats(self) -> dict:
        return {"limit": self.doc(), "allowed": self.n_allowed,
                "rejected_429": self.n_rejected}
