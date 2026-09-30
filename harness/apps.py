"""Harness-owned EnvApps: clock, schedule, agent schedules (TM-D), waits,
oracle.

Provisioning (`harness_apps`) derives from the run config alone: every run
gets the clock, the spend dashboard, and the timed sleep tool; TM-B cells
additionally get the authored-wait-program affordances (WorkspaceApp +
ProgramApp — the A/B contrast is "may the actor delegate watching to code
it writes"). The oracle app exists only in sig=oracle runs; the wait-party roster
tool exists only for task-authored scaffolds (scaffold "task:<name>",
waitparty.py). The task adds its own apps via Task.env_apps().
"""

from __future__ import annotations

from datetime import datetime

from harness.env_tools import EnvApp, ToolError, tool
from harness.timeutil import iso, parse_iso


def _parse_time(s, field: str) -> datetime:
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {field!r}: {s!r}")


class ClockApp(EnvApp):
    @tool("get_time() -> current simulated time and sim_end",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def get_time(self, args: dict) -> dict:
        sim = self.sim
        return {"now": iso(sim.clock.now), "sim_end": iso(sim.cfg.sim_end)}


class ScheduleApp(EnvApp):
    @tool("get_crontab() -> your recurring schedule "
          "[{id, cron_expr}, ...] (lives in the environment; survives your "
          "process exiting)",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def get_crontab(self, args: dict) -> list[dict]:
        return self.sim.schedule.get_crontab()

    @tool("set_crontab(entries: [{id, cron_expr}, ...]) -> replace your "
          "recurring schedule (idempotent: re-declaring the same entries "
          "does not re-fire them)",
          schema={"additionalProperties": False, "properties": {"entries": {"items": {"additionalProperties": False, "properties": {"agent_owned": {"type": "boolean"}, "cron_expr": {"type": "string"}, "id": {"type": "string"}, "target": {"type": "string"}}, "required": ["id", "cron_expr"], "type": "object"}, "type": "array"}}, "required": ["entries"], "type": "object"})
    async def set_crontab(self, args: dict) -> dict:
        """Program plumbing (never in an agent's registry). An entry may
        carry `agent_owned: true` (+ optional `target`): a TM-D DEFAULT
        schedule, installed once as an agent-owned row the agent may then
        update or delete — re-declaring it never brings it back
        (harness/schedule.py)."""
        sim = self.sim
        entries = args.get("entries")
        if not isinstance(entries, list):
            raise ToolError("set_crontab needs 'entries': a list of "
                            "{id, cron_expr}")
        norm = []
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str) \
                    or not isinstance(e.get("cron_expr"), str):
                raise ToolError(f"bad crontab entry: {e!r}")
            row = {"id": e["id"], "cron_expr": e["cron_expr"]}
            if e.get("agent_owned"):  # a default schedule (TM-D)
                row["agent_owned"] = True
                if e.get("target") is not None:
                    row["target"] = e["target"]
            norm.append(row)
        async with sim.lock:
            try:
                seeded = sim.schedule.set_crontab(norm, sim.clock.now)
            except ValueError as e:
                raise ToolError(str(e))
            sim.ledger.append("crontab_put", sim.clock.now, entries=norm)
            for view in seeded:  # once per default schedule, ever
                sim.ledger.append("schedule_seed", sim.clock.now,
                                  **{("sched_type" if k == "type" else k): v
                                     for k, v in view.items()})
        return {"status": "ok"}

    @tool("run_at(id: str, at: iso datetime) -> one-shot wake-up at `at` "
          "(an `at` in the past fires immediately after you exit)",
          schema={"additionalProperties": False, "properties": {"at": {"format": "date-time", "type": "string"}, "id": {"type": "string"}}, "required": ["id", "at"], "type": "object"})
    async def run_at(self, args: dict) -> dict:
        sim = self.sim
        job_id = args.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ToolError("run_at needs a string 'id'")
        at = _parse_time(args.get("at"), "at")
        async with sim.lock:
            sim.schedule.run_at(job_id, at)
            sim.ledger.append("run_at", sim.clock.now, id=job_id, at=iso(at))
        return {"status": "ok"}


class AgentScheduleApp(EnvApp):
    """TM-D: the agent's
    REST-style CRUD over its OWN wake-up schedules. Crontab rows (owner
    `system`) are read-only by construction — the store keeps agent rows
    apart from the crontab — and `learn` is hidden from the listing
    (learning plumbing is never agent-visible, like the feedback tools
    the cron mains pop). A main may install its base
    cadence as a DEFAULT schedule instead (a `set_crontab` entry flagged
    `agent_owned`): an agent-owned row like any other, which the agent may
    update or delete; mains whose base is shared plumbing keep a `system` row. All free:
    scheduling costs nothing in reality; each FIRING costs whatever the
    episode then spends. Mutations are ledgered (schedule_create/update/
    delete) so the agent's schedule-shaping is the arm's primary read."""

    HIDDEN_IDS = frozenset({"learn"})

    @tool("list_schedules() -> free: your schedules [{id, owner, type, "
          "cron_expr|at, note, next_fire}]; an entry whose owner is "
          "`system` is fixed, every other one is yours to change",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def list_schedules(self, args: dict) -> dict:
        sim = self.sim
        async with sim.lock:
            now = sim.clock.now
            rows = [r for r in sim.schedule.system_view(now)
                    if r["id"] not in self.HIDDEN_IDS
                    and not r["id"].startswith("__")]
            rows += sim.schedule.agent_list(now)
            rows.sort(key=lambda r: (r["next_fire"], r["id"]))
            return {"now": iso(now), "schedules": rows}

    @tool("create_schedule(id: str, at: iso | cron_expr: str, note?: str) -> "
          "free: add a wake-up schedule — one-time (`at`) or recurring "
          "(`cron_expr`); the note is delivered to you when it fires",
          schema={"additionalProperties": False, "properties": {"at": {"format": "date-time", "type": "string"}, "cron_expr": {"type": "string"}, "id": {"type": "string"}, "note": {"type": "string"}}, "required": ["id"], "type": "object"})
    async def create_schedule(self, args: dict) -> dict:
        return await self._put(args, create=True)

    @tool("update_schedule(id: str, at: iso | cron_expr: str, note?: str) -> "
          "free: replace one of your schedules (a `system` entry is "
          "read-only)",
          schema={"additionalProperties": False, "properties": {"at": {"format": "date-time", "type": "string"}, "cron_expr": {"type": "string"}, "id": {"type": "string"}, "note": {"type": "string"}}, "required": ["id"], "type": "object"})
    async def update_schedule(self, args: dict) -> dict:
        return await self._put(args, create=False)

    @tool("delete_schedule(id: str) -> free: remove one of your schedules "
          "(a one-time schedule is removed automatically after it fires)",
          schema={"additionalProperties": False, "properties": {"id": {"type": "string"}}, "required": ["id"], "type": "object"})
    async def delete_schedule(self, args: dict) -> dict:
        sim = self.sim
        sched_id = args.get("id")
        if not isinstance(sched_id, str) or not sched_id:
            raise ToolError("delete_schedule needs a string 'id'")
        async with sim.lock:
            try:
                sim.schedule.agent_delete(sched_id)
            except ValueError as e:
                raise ToolError(str(e))
            sim.ledger.append("schedule_delete", sim.clock.now, id=sched_id)
        return {"status": "ok", "id": sched_id}

    async def _put(self, args: dict, *, create: bool) -> dict:
        sim = self.sim
        verb = "create_schedule" if create else "update_schedule"
        sched_id = args.get("id")
        if not isinstance(sched_id, str) or not sched_id:
            raise ToolError(f"{verb} needs a string 'id'")
        at = args.get("at")
        cron_expr = args.get("cron_expr")
        if at is not None:
            at = _parse_time(at, "at")
        note = args.get("note")
        # `target` is the per-entity routing tag: accepted
        # here, documented only by the mains that expose it
        target = args.get("target")
        async with sim.lock:
            fn = sim.schedule.agent_create if create \
                else sim.schedule.agent_update
            try:
                row = fn(sched_id, now=sim.clock.now, at=at,
                         cron_expr=cron_expr, note=note, target=target)
            except ValueError as e:
                raise ToolError(str(e))
            sim.ledger.append("schedule_create" if create
                              else "schedule_update", sim.clock.now,
                              **{("sched_type" if k == "type" else k): v
                                 for k, v in row.items()})
        return {"status": "ok", "schedule": row}


class SleepApp(EnvApp):
    @tool("sleep(until: iso datetime) -> block (in simulated time) until "
          "`until` or your next scheduled trigger", tags=("wait",),
          schema={"additionalProperties": False, "properties": {"until": {"format": "date-time", "type": "string"}, "waiter_id": {"type": "string"}}, "required": ["until"], "type": "object"})
    async def sleep(self, args: dict, brief: bool = True) -> dict:
        """Sim-time blocking only: advances the clock to min(until, next due
        trigger, sim_end) and returns immediately in real time. A trigger
        that comes due during the sleep is delivered through the response
        (`woke_for: "trigger"`) and consumed, instead of spawning a second
        agent process. Only REAL triggers count (`peek_due`): the
        synthesized midnight fallback is the supervisor's re-invocation
        rule and never ends a wait. `brief=False` (authored programs' internal waits):
        the return is not an agent wake, so it never carries or consumes
        the daily cost brief."""
        sim = self.sim
        until = _parse_time(args.get("until"), "until")
        if sim.party is not None and sim.party.roster:
            return await sim.party.wait(args.get("waiter_id"), until,
                                        brief=brief)
        async with sim.lock:
            if sim.clock.finished:
                return {"experiment_over": True, "now": iso(sim.clock.now)}
            now = sim.clock.now
            if until <= now:
                return {"now": iso(now), "woke_for": "sleep"}
            trig = sim.schedule.peek_due(now)
            wake = min(until, sim.cfg.sim_end)
            if trig is not None:
                wake = min(wake, trig.due_time)
            sim.clock.advance_to(wake)
            sim.book_due_outcomes()
            if sim.clock.finished:
                log_sleep(sim, args.get("waiter_id"), until, now,
                          "experiment_over")
                return {"experiment_over": True, "now": iso(sim.clock.now)}
            if trig is not None and wake == trig.due_time:
                sim.schedule.consume(trig)
                sim.ledger.append("trigger", sim.clock.now, id=trig.id,
                                  kind=trig.kind, via="sleep")
                log_sleep(sim, args.get("waiter_id"), until, now, "trigger")
                return {
                    "now": iso(sim.clock.now),
                    "woke_for": "trigger",
                    "trigger": trig.payload(),
                    **(_costs(sim, sim.clock.now) if brief else {}),
                }
            log_sleep(sim, args.get("waiter_id"), until, now, "sleep")
            return {"now": iso(sim.clock.now), "woke_for": "sleep",
                    **(_costs(sim, sim.clock.now) if brief else {})}


def _costs(sim, now: datetime) -> dict:
    """`{"costs": brief}` on the first wake of a sim date, else `{}`
    ."""
    brief = sim.daily_brief(now)
    return {"costs": brief} if brief is not None else {}


def log_sleep(sim, waiter, until: datetime, armed_at: datetime,
              woke_for: str) -> None:
    """The additive `sleep` ledger event: every wait that moved (or tried
    to move) the clock is recorded
    for every cell — armed at / asked until / woke at / why. Cost-free;
    caller holds sim.lock and has already advanced the clock."""
    sim.ledger.append("sleep", sim.clock.now, waiter=waiter,
                      armed_at=iso(armed_at), until=iso(until),
                      woke_for=woke_for)


class PartyApp(EnvApp):
    @tool("set_party(waiter_ids: [str], trigger_waiter: str) -> declare "
          "this process's concurrent-waiter roster (program-level plumbing "
          "for multi-agent programs: each sleep call then carries a "
          "waiter_id; the clock advances when every roster member is "
          "waiting; scheduled triggers wake trigger_waiter)",
          schema={"additionalProperties": False, "properties": {"trigger_waiter": {"type": "string"}, "waiter_ids": {"items": {"type": "string"}, "type": "array"}}, "required": ["waiter_ids", "trigger_waiter"], "type": "object"})
    async def set_party(self, args: dict) -> dict:
        """Wait-party roster (harness/waitparty.py). Provisioned only for
        task-authored scaffolds (scaffold 'task:<name>'); replacement
        semantics like set_crontab — a finished agent leaves the party by
        re-declaring the roster without itself, from its own turn."""
        from harness.waitparty import WaitParty

        sim = self.sim
        ids = args.get("waiter_ids")
        tw = args.get("trigger_waiter")
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(i, str) and i for i in ids)):
            raise ToolError("set_party needs 'waiter_ids': a non-empty "
                            "list of strings")
        if not isinstance(tw, str) or not tw:
            raise ToolError("set_party needs 'trigger_waiter': a string")
        async with sim.lock:
            if sim.party is None:
                sim.party = WaitParty(sim)
            sim.party.set_roster(ids, tw)
            sim.ledger.append("set_party", sim.clock.now, waiter_ids=ids,
                              trigger_waiter=tw)
        return {"status": "ok"}


class CostsApp(EnvApp):
    @tool("get_costs() -> free: your cumulative spend so far, by category "
          "(LLM tokens + API fees — everything billed against your budget), "
          "and your budgets (total and LLM) with what remains",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def get_costs(self, args: dict) -> dict:
        """Universal billing dashboard. Spend is self-observable
        information — the agent already knows the price of every call it
        makes — so exposing the running total is free and
        sig-axis-neutral. Rate-limit consumption is deliberately NOT
        shown anywhere: the limit is advertised, tracking usage is the
        agent's own job."""
        sim = self.sim
        async with sim.lock:
            status = sim.budget_status()
            spend = status["spend_by_type"]
            return {"now": iso(sim.clock.now), **status,
                    "spend_total": round(sum(spend.values()), 8)}


class OracleApp(EnvApp):
    @tool("get_feedback(since?: iso) -> settled ground-truth outcomes since "
          "`since` (free oracle)", tags=("feedback",),
          schema={"additionalProperties": False, "properties": {"since": {"format": "date-time", "type": "string"}}, "required": [], "type": "object"})
    async def get_feedback(self, args: dict) -> dict:
        """Sig-oracle: settled outcomes with t_settled in (since, now].
        Free, universal. This app is not provisioned in other cells."""
        sim = self.sim
        since = args.get("since")
        s = _parse_time(since, "since") if since is not None else None
        async with sim.lock:
            now = sim.clock.now
            outcomes = sim.task.oracle_outcomes(s, now)
            return {
                "now": iso(now),
                "outcomes": outcomes,
                "spend_total": sim.ledger.total_cost(),
            }


def harness_apps(sim) -> list[EnvApp]:
    """Provisioning table for the harness-owned apps."""
    from harness.contract import is_adhoc, is_external

    apps: list[EnvApp] = [ClockApp(sim), CostsApp(sim)]
    cell = sim.cfg.cell
    external = is_external(sim.cfg)
    if cell.tm != "B" or external:
        # TM-B has no blind sleep tool: waiting happens only through
        # run_program (the seeded sleep.py is the plain-sleep floor), so
        # the timing mechanism is the program affordance itself. An
        # external program
        # waits by client-side authored code whose only clock is this
        # sleep endpoint — its tm=B discipline is the program's own
        apps.append(SleepApp(sim))
    if cell.tm in ("C", "D") or (cell.tlrn_spec or {}).get("source") == "sim":
        # Program-managed schedules: acting (TM-C/D base cron) or a
        # sim-scheduled learning trigger (tlrn source=sim) — the program
        # installs its crontab at start. Pure agent-owned-timing cells
        # get no schedule tools: the actor's wake is its wait tool, never
        # a schedule it maintains (and the react renderer pops schedule
        # tools from the actor's registry even when provisioned).
        apps.append(ScheduleApp(sim))
    if cell.tm == "D":
        # TM-D: the agent's own schedule CRUD beside the read-only base
        # cron. Server-side
        # state only — fine detached.
        apps.append(AgentScheduleApp(sim))
    if cell.tm == "B" and not external:
        # extended TM-B: the
        # author + run-and-wait affordances are the WHOLE A/B contrast —
        # all watching executes real billed calls. Never
        # for ext: — that would be agent code executing in the server
        from harness.authored import ProgramApp, WorkspaceApp

        apps += [WorkspaceApp(sim), ProgramApp(sim)]
    if is_adhoc(sim.cfg):
        # task-authored / external multi-agent programs get the wait-party
        # roster tool; generated mono cells never see it in their manifest
        apps.append(PartyApp(sim))
    if cell.sig == "oracle":
        apps.append(OracleApp(sim))
    return apps
