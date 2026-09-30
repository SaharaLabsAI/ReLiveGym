"""Sim: the shared per-run context (clock, schedule, task, ledger, wallet).

Owned by the run entry point; mutated by the API handlers and the supervisor.
Everything runs on one asyncio event loop, guarded by a single lock. Only two
places ever advance the clock: the supervisor (between agent processes) and the
/sleep handler (while exactly one agent process is alive) — mutually exclusive by
the single-process invariant.

Money: ONE wallet per run. Every dollar in
the ledger's cost column is real spend — LLM tokens at real rates plus API
fees at real rates. Outcomes are cost-free ledger events; performance is
the task's metrics, reported separately from resources.
"""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from harness.clock import SimClock
from harness.config import RunConfig
from harness.env_tools import PaymentRequired
from harness.ledger import Ledger
from harness.limits import RateLimiter
from harness.schedule import ScheduleStore
from harness.task import Task


class Sim:
    def __init__(self, cfg: RunConfig, run_dir: Path, workspace: Path, task: Task,
                 pause_at: datetime | None = None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.workspace = workspace
        # pause_at: an optional clamp inside the window (harness/checkpoint.py)
        self.clock = SimClock(cfg.sim_start, cfg.sim_end, pause_at)
        self.schedule = ScheduleStore(run_dir / "schedule.json")
        self.task = task
        self.ledger = Ledger(run_dir / "ledger.jsonl")
        # full LLM request/response bodies, one JSONL event per proxied call
        self.llm_log = (Ledger(run_dir / "llm_log.jsonl", keep_events=False)
                        if cfg.log_llm_traffic else None)
        self.lock = asyncio.Lock()
        self.party = None  # WaitParty | None; opt-in via set_party, reset
        #                    by the scheduler at process exit (waitparty.py)
        self.scheduler = None  # harness.supervisor.Scheduler, set by the run
        #                        entry point (lifecycle routes in api.py)
        self.token = secrets.token_hex(16)  # the CONTROLLER token: every route
        # the data-plane token: /tools, /call,
        # /llm only — what an external actor's bridge and LLM client hold
        self.agent_token = secrets.token_hex(16)
        self.url: str | None = None  # set once the HTTP server is bound
        # web hosts (harness/web.py): name -> url, name -> WebHostSpec
        self.hosts: dict[str, str] = {}
        self.program_active = False  # an authored program is running (POST /program)
        self.host_specs: dict = {}

        # the wallet: real USD spent so far (LLM tokens + API fees), plus
        # per-domain tallies for the cfg.domain_budgets caps
        self.wallet_spend = 0.0
        self.llm_spend = 0.0  # the LLM share of wallet_spend
        self.llm_provider_spend = 0.0  # provider-issued LLM bills, NOT
        #   booked: kept beside llm_spend so config-vs-provider drift is
        #   one subtraction in results.json
        self.domain_spend: dict[str, float] = {}
        # task-registered rate limiters, name -> RateLimiter (harness/limits.py)
        self.limiters: dict[str, RateLimiter] = {}

        # watchdog bookkeeping (real time)
        self.last_activity = time.monotonic()
        self.llm_inflight = 0  # watchdog suspended while > 0 (see llm_begin)
        self._llm_inflight_started: dict[int, float] = {}  # call token -> monotonic
        self._llm_token = 0
        self.api_calls_this_run = 0  # reset by the supervisor at each spawn

        self.flags: list[str] = []  # failed-degenerate, budget_exhausted, ...

        # daily cost brief: the date
        # last briefed and the sim instant of that date's first wake
        self._briefed_date: date | None = None
        self._brief_instant: datetime | None = None
        # last: the task may keep the sim (ledger-backed grading rules)
        task.bind(self)

    def touch(self) -> None:
        """Record agent activity: resets the watchdog, counts as a successful
        helper interaction for the K-consecutive-crashes rule."""
        self.last_activity = time.monotonic()
        self.api_calls_this_run += 1

    # -- in-flight LLM calls: the watchdog's suspension, bounded ------------------------

    def llm_begin(self) -> int:
        """Register one upstream LLM call; returns its token for llm_end."""
        self._llm_token += 1
        self._llm_inflight_started[self._llm_token] = time.monotonic()
        self.llm_inflight = len(self._llm_inflight_started)
        return self._llm_token

    def llm_end(self, token: int) -> None:
        self._llm_inflight_started.pop(token, None)
        self.llm_inflight = len(self._llm_inflight_started)

    def llm_allowance_seconds(self) -> float:
        """How long one /llm request may legitimately stay in flight: every
        attempt's timeout plus the backoffs between them. GET /activity
        reports it as llm_timeout_seconds; the runner kills past it +
        watchdog_seconds."""
        cfg = self.cfg
        return (cfg.llm_timeout_seconds * (cfg.llm_retries + 1)
                + sum(cfg.llm_retry_backoff_seconds * 2 ** k
                      for k in range(cfg.llm_retries)))

    def oldest_llm_inflight_seconds(self) -> float:
        """Real-time age of the longest-pending upstream call (0 when none):
        what the runner compares against llm_timeout + watchdog."""
        if not self._llm_inflight_started:
            return 0.0
        return time.monotonic() - min(self._llm_inflight_started.values())

    # -- daily cost brief ---------------------------------------------------------------

    def daily_brief(self, now: datetime) -> dict | None:
        """The cost & budget brief carried by the first wake payload on each
        simulated UTC date (and by every payload at that same instant, so
        all agents waking together see it). No catch-up for skipped dates.
        Caller holds sim.lock."""
        if now == self._brief_instant:
            return self._brief(now)
        if now.date() == self._briefed_date:
            return None
        self._briefed_date, self._brief_instant = now.date(), now
        brief = self._brief(now)
        self.ledger.append("brief", now, day=brief["day"],
                           spent_usd=brief["spent_usd"],
                           llm_spent_usd=brief["llm"]["spent_usd"])
        return brief

    def budget_status(self) -> dict:
        """Budget vs spend, shared by the brief and get_costs."""
        def block(cap: float, spent: float) -> dict:
            return {"budget_usd": cap, "spent_usd": round(spent, 8),
                    "remaining_usd": round(max(cap - spent, 0.0), 8)}
        caps = dict(self.cfg.domain_budgets)
        out = block(self.cfg.budget_usd, self.wallet_spend)
        out["llm"] = block(caps.pop("llm", 0.0),
                           self.domain_spend.get("llm", 0.0))
        domains = {t: block(c, self.domain_spend.get(t, 0.0))
                   for t, c in sorted(caps.items())}
        if domains:
            out["domains"] = domains
        out["spend_by_type"] = {k: v for k, v in sorted(
            self.ledger.cost_by_type().items()) if v}
        return out

    def _brief(self, now: datetime) -> dict:
        cfg = self.cfg
        total = math.ceil((cfg.sim_end - cfg.sim_start) / timedelta(days=1))
        day = (now.date() - cfg.sim_start.date()).days + 1
        return {"day": day, "of": total, **self.budget_status()}

    # -- wallet ---------------------------------------------------------------------

    def bill(self, type: str, cost: float, **detail) -> None:
        """Book one paid interaction against the wallet. A cost-bearing
        call is refused (HTTP 402 via the API layer) when the global
        budget is spent (flag budget_exhausted) or this type's domain cap
        is (flag <type>_budget_exhausted) — free calls always go through.
        Caller holds sim.lock."""
        if cost > 0 and self.wallet_spend >= self.cfg.budget_usd:
            if "budget_exhausted" not in self.flags:
                self.flags.append("budget_exhausted")
            self.ledger.append("payment_rejected", self.clock.now,
                               type_refused=type, reason="budget_usd")
            raise PaymentRequired(
                f"budget exhausted: ${self.cfg.budget_usd:g} for this run "
                f"is spent; paid calls are refused")
        cap = self.cfg.domain_budgets.get(type)
        if cost > 0 and cap is not None \
                and self.domain_spend.get(type, 0.0) >= cap:
            flag = f"{type}_budget_exhausted"
            if flag not in self.flags:
                self.flags.append(flag)
            self.ledger.append("payment_rejected", self.clock.now,
                               type_refused=type, reason=f"domain:{type}")
            raise PaymentRequired(
                f"{type} budget exhausted: ${cap:g} for this run is "
                f"spent; {type} calls are refused")
        self.ledger.append(type, self.clock.now, cost=cost, **detail)
        self.wallet_spend += cost
        if cost:
            self.domain_spend[type] = self.domain_spend.get(type, 0.0) + cost

    # -- settlement -----------------------------------------------------------------

    def book_due_outcomes(self) -> None:
        """Settle periods due by the current sim time; outcomes are
        cost-free ledger events (event-sourced results)."""
        for e in self.task.close_due(self.clock.now):
            self.ledger.append("outcome", self.clock.now, ref=e.ref,
                               status=e.status, **e.detail)

    def book_all_outcomes(self) -> None:
        for e in self.task.close_all():
            self.ledger.append("outcome", self.clock.now, ref=e.ref,
                               status=e.status, **e.detail)

    # -- results ----------------------------------------------------------------------

    def results(self) -> dict:
        by_type = {k: v for k, v in self.ledger.cost_by_type().items() if v}
        spent = self.ledger.total_cost()
        performance = self.task.metrics()
        violations = list(self.task.constraint_violations())
        exhausted = [f for f in self.flags if f.endswith("budget_exhausted")]
        violations.extend(exhausted)
        return {
            "run_id": self.cfg.run_id,
            "flags": self.flags,
            "performance": performance,
            "resources": {
                "budget_usd": self.cfg.budget_usd,
                "domain_budgets": dict(self.cfg.domain_budgets),
                "spent_usd": spent,
                "llm_usd": round(self.llm_spend, 8),
                "llm_provider_usd": round(self.llm_provider_spend, 8),
                "fees_usd": round(spent - self.llm_spend, 8),
                "spend_by_type": by_type,
                "counts": self.ledger.count_by_type(),
                "rate_limits": {name: lim.stats()
                                for name, lim in sorted(self.limiters.items())},
            },
            "constraints": {
                "within_budget": not exhausted,  # global AND domain caps
                "violations": violations,
            },
            "task": {"name": self.cfg.task_name, **self.task.report()},
            "monthly_spend": self.ledger.monthly_cost_buckets(),
            "config": self.cfg.model_dump(mode="json"),
        }
