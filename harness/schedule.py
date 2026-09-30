"""Persistent schedule store: the helper-side 'crontab'.

Holds recurring cron entries, one-shot `at` jobs, and synthesizes the fallback
midnight trigger when both are empty. The store never inspects agent code; it is
mutated only through the HTTP API and consumed by the supervisor.

The fallback is a LIFECYCLE rule only: `peek_next` — the
supervisor's view — synthesizes it so a program that exited with nothing
pending is re-invoked at the next midnight; `peek_due` — what a wait (sleep,
wait party) sees — never does, so a live agent's wait ends at its own `until`,
a real trigger, or sim_end, and nothing else.

Semantics:
- Cron entries fire strictly *after* their install time / last fire (standard cron:
  installing "0 23 * * *" at 23:00 fires tomorrow, not instantly).
- One-shot jobs fire at their `at`; an `at` in the past is overdue and fires "now".
  `run_at` with an existing id replaces that job.
- `set_crontab` is an idempotent full replacement (PUT): entries whose id and
  expression are unchanged keep their last-fire state, so a re-declaring scaffold
  does not re-fire.
- Agent-owned schedules are a
  THIRD entry class, kept apart from the program crontab and program one-shots
  so ownership holds by construction: `set_crontab`'s full replacement can never
  touch an agent row, and the agent CRUD (`agent_*`) can never touch a program
  row — that is what makes the base schedule read-only. A `once` row fires at
  its `at` and is removed; a `recurring` row follows its cron expression with
  the same install/last-fire semantics as a crontab entry. Ids share ONE
  namespace across all three classes (an agent cannot shadow `learn`).
- A DEFAULT schedule (TM-D) is an agent-owned row the PROGRAM
  installs: a `set_crontab` entry flagged `agent_owned` is created once as a
  recurring agent row and never again — the store remembers the id as seeded —
  so the agent may update or delete it like any row of its own and a
  re-declaring program cannot bring it back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from croniter import CroniterBadDateError, croniter

from harness.timeutil import as_utc, iso, next_midnight, parse_iso

TriggerKind = Literal["cron", "at", "fallback", "crash_recovery"]
Owner = Literal["system", "agent"]

FALLBACK_ID = "__fallback_midnight__"

# agent-owned schedule limits
MAX_AGENT_SCHEDULES = 200
NOTE_MAX_CHARS = 1000
TARGET_MAX_CHARS = 200
_AGENT_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
# the cron mains' program-entry ids: reserved even when not installed in
# this cell (a no-learning cell has no `learn` row, but the main still
# dispatches that id to learn(); `act` is the base / default schedule's id —
# an agent may update or delete a default `act` row, never create one)
RESERVED_IDS = frozenset({"act", "learn"})


@dataclass(frozen=True)
class Trigger:
    id: str
    kind: TriggerKind
    due_time: datetime
    owner: Owner = "system"      # "agent" for TM-D agent-owned schedules
    note: str | None = None      # agent-authored, delivered on fire
    target: str | None = None    # opaque routing tag (per-entity mains)

    def payload(self) -> dict:
        """The wire shape shared by the lifecycle route, the sleep
        response and ENV_TRIGGER: {id, kind, due_time} always; owner/
        note/target only when set, so every pre-TM-D payload is
        byte-identical to before."""
        d = {"id": self.id, "kind": self.kind, "due_time": iso(self.due_time)}
        if self.owner != "system":
            d["owner"] = self.owner
        if self.note is not None:
            d["note"] = self.note
        if self.target is not None:
            d["target"] = self.target
        return d

    def to_env(self) -> str:
        """JSON payload for the ENV_TRIGGER env var."""
        return json.dumps(self.payload())


class _CronEntry:
    __slots__ = ("expr", "installed_at", "last_fire")

    def __init__(self, expr: str, installed_at: datetime,
                 last_fire: datetime | None = None):
        if not croniter.is_valid(expr):
            raise ValueError(f"invalid cron expression: {expr!r}")
        try:
            # satisfiability, not just syntax: "0 0 31 2 *" parses but never
            # fires, and an unsatisfiable entry must be a rejected PUT (free
            # 400), never a scheduler crash at peek time
            croniter(expr, installed_at).get_next(datetime)
        except CroniterBadDateError:
            raise ValueError(f"cron expression never fires: {expr!r}")
        self.expr = expr
        self.installed_at = installed_at
        self.last_fire = last_fire

    def next_fire(self, now: datetime) -> datetime:
        # First occurrence strictly after the entry's own last fire (or install
        # time), NOT after `now`: an occurrence equal to `now` that has not fired
        # yet must still fire. Otherwise two entries due at the same instant
        # starve each other — only one can be consumed per peek, and basing the
        # next fire on `now` would silently skip the other's occurrence (real
        # cron runs both jobs; we run them sequentially at the same sim time).
        base = self.last_fire if self.last_fire is not None else self.installed_at
        t = croniter(self.expr, base).get_next(datetime)
        return max(t, now)


class _AgentSchedule:
    """One agent-owned row: `once` (at) or `recurring` (cron entry)."""
    __slots__ = ("id", "at", "cron", "note", "target", "created_at")

    def __init__(self, sched_id: str, *, at: datetime | None,
                 cron: _CronEntry | None, note: str | None,
                 target: str | None, created_at: datetime):
        self.id = sched_id
        self.at = at
        self.cron = cron
        self.note = note
        self.target = target
        self.created_at = created_at

    @property
    def kind(self) -> str:
        return "once" if self.at is not None else "recurring"

    def next_fire(self, now: datetime) -> datetime:
        if self.at is not None:
            return max(self.at, now)
        assert self.cron is not None
        return self.cron.next_fire(now)

    def view(self, now: datetime) -> dict:
        d = {"id": self.id, "owner": "agent", "type": self.kind,
             "next_fire": iso(self.next_fire(now))}
        if self.at is not None:
            d["at"] = iso(self.at)
        else:
            d["cron_expr"] = self.cron.expr
        if self.note is not None:
            d["note"] = self.note
        if self.target is not None:
            d["target"] = self.target
        return d

    def to_json(self) -> dict:
        return {"id": self.id, "at": iso(self.at) if self.at else None,
                "cron_expr": self.cron.expr if self.cron else None,
                "installed_at": iso(self.cron.installed_at) if self.cron else None,
                "last_fire": (iso(self.cron.last_fire)
                              if self.cron and self.cron.last_fire else None),
                "note": self.note, "target": self.target,
                "created_at": iso(self.created_at)}

    @classmethod
    def from_json(cls, d: dict) -> "_AgentSchedule":
        cron = None
        if d.get("cron_expr"):
            cron = _CronEntry(d["cron_expr"], parse_iso(d["installed_at"]),
                              parse_iso(d["last_fire"]) if d.get("last_fire")
                              else None)
        return cls(d["id"], at=parse_iso(d["at"]) if d.get("at") else None,
                   cron=cron, note=d.get("note"), target=d.get("target"),
                   created_at=parse_iso(d["created_at"]))


class ScheduleStore:
    def __init__(self, persist_path: Path | None = None):
        self._cron: dict[str, _CronEntry] = {}
        self._oneshots: dict[str, datetime] = {}
        self._agent: dict[str, _AgentSchedule] = {}
        self._seeded: set[str] = set()  # default-schedule ids installed so far
        self._persist_path = persist_path

    # -- mutation (via HTTP API) ------------------------------------------------

    def set_crontab(self, entries: list[dict], now: datetime) -> list[dict]:
        """Full replacement; ids seen before with the same expr keep their fire
        state (idempotent PUT), new or changed entries start from `now`.

        An entry flagged `agent_owned` (optional `target`) is a DEFAULT
        schedule: not part of the crontab, installed ONCE as an agent-owned
        recurring row. Re-declaring it is a no-op even after the agent changed
        or deleted it. Returns the rows seeded by this call."""
        new: dict[str, _CronEntry] = {}
        defaults: dict[str, _AgentSchedule] = {}
        for e in entries:
            entry_id, expr = e["id"], e["cron_expr"]
            if entry_id in new or entry_id in defaults:
                raise ValueError(f"duplicate crontab id: {entry_id!r}")
            if e.get("agent_owned"):
                if entry_id in self._oneshots:
                    raise ValueError(
                        f"default schedule id {entry_id!r} is a one-shot job")
                if entry_id not in self._seeded and entry_id not in self._agent:
                    defaults[entry_id] = self._agent_row(
                        entry_id, None, expr, None, e.get("target"), now, now)
                continue
            if entry_id in self._agent:
                raise ValueError(
                    f"crontab id {entry_id!r} is an agent-owned schedule")
            old = self._cron.get(entry_id)
            if old is not None and old.expr == expr:
                new[entry_id] = _CronEntry(expr, old.installed_at, old.last_fire)
            else:
                new[entry_id] = _CronEntry(expr, now)
        if len(self._agent) + len(defaults) > MAX_AGENT_SCHEDULES:
            raise ValueError(f"at most {MAX_AGENT_SCHEDULES} schedules may "
                             "be pending at once")
        self._cron = new
        self._agent.update(defaults)
        self._seeded.update(e["id"] for e in entries if e.get("agent_owned"))
        self._save()
        return [r.view(now) for r in defaults.values()]

    def get_crontab(self) -> list[dict]:
        return [{"id": i, "cron_expr": e.expr} for i, e in self._cron.items()]

    def run_at(self, job_id: str, at: datetime) -> None:
        if job_id in self._agent:
            raise ValueError(f"job id {job_id!r} is an agent-owned schedule")
        self._oneshots[job_id] = as_utc(at)
        self._save()

    # -- agent-owned schedules (TM-D) ---------------------------------------------

    def system_view(self, now: datetime) -> list[dict]:
        """Program-owned rows as the agent may see them (read-only)."""
        rows = [{"id": i, "owner": "system", "type": "recurring",
                 "cron_expr": e.expr, "next_fire": iso(e.next_fire(now))}
                for i, e in self._cron.items()]
        rows += [{"id": i, "owner": "system", "type": "once",
                  "at": iso(at), "next_fire": iso(max(at, now))}
                 for i, at in self._oneshots.items()]
        return rows

    def agent_list(self, now: datetime) -> list[dict]:
        return [r.view(now) for r in self._agent.values()]

    def agent_get(self, sched_id: str, now: datetime) -> dict | None:
        r = self._agent.get(sched_id)
        return r.view(now) if r is not None else None

    def _agent_row(self, sched_id, at, cron_expr, note, target, now,
                   created_at) -> _AgentSchedule:
        if not isinstance(sched_id, str) or not _AGENT_ID_RE.match(sched_id) \
                or sched_id.startswith("__"):
            raise ValueError(
                "schedule id must match [a-z0-9_-]{1,64} (not starting "
                "with __)")
        if (at is None) == (cron_expr is None):
            raise ValueError("give exactly one of 'at' (one-time) or "
                             "'cron_expr' (recurring)")
        if note is not None:
            if not isinstance(note, str):
                raise ValueError("note must be a string")
            if len(note) > NOTE_MAX_CHARS:
                raise ValueError(f"note longer than {NOTE_MAX_CHARS} chars")
        if target is not None:
            if not isinstance(target, str) or not target:
                raise ValueError("target must be a non-empty string")
            if len(target) > TARGET_MAX_CHARS:
                raise ValueError(
                    f"target longer than {TARGET_MAX_CHARS} chars")
        cron = None
        if cron_expr is not None:
            if not isinstance(cron_expr, str):
                raise ValueError("cron_expr must be a string")
            old = self._agent.get(sched_id)
            if old is not None and old.cron is not None \
                    and old.cron.expr == cron_expr:
                cron = old.cron  # unchanged expression keeps its fire state
            else:
                cron = _CronEntry(cron_expr, now)
        return _AgentSchedule(sched_id, at=as_utc(at) if at else None,
                              cron=cron, note=note, target=target,
                              created_at=created_at)

    def agent_create(self, sched_id: str, *, now: datetime,
                     at: datetime | None = None, cron_expr: str | None = None,
                     note: str | None = None,
                     target: str | None = None) -> dict:
        if sched_id in self._cron or sched_id in self._oneshots \
                or sched_id in RESERVED_IDS:
            raise ValueError(f"schedule id {sched_id!r} is reserved")
        if sched_id in self._agent:
            raise ValueError(f"schedule {sched_id!r} already exists "
                             "(use update_schedule)")
        if len(self._agent) >= MAX_AGENT_SCHEDULES:
            raise ValueError(f"at most {MAX_AGENT_SCHEDULES} schedules may "
                             "be pending at once")
        row = self._agent_row(sched_id, at, cron_expr, note, target, now, now)
        self._agent[sched_id] = row
        self._save()
        return row.view(now)

    def agent_update(self, sched_id: str, *, now: datetime,
                     at: datetime | None = None, cron_expr: str | None = None,
                     note: str | None = None,
                     target: str | None = None) -> dict:
        old = self._agent.get(sched_id)
        if old is None:
            self._not_agent_owned(sched_id)
        row = self._agent_row(sched_id, at, cron_expr, note, target, now,
                              old.created_at)
        self._agent[sched_id] = row
        self._save()
        return row.view(now)

    def agent_delete(self, sched_id: str) -> None:
        if sched_id not in self._agent:
            self._not_agent_owned(sched_id)
        del self._agent[sched_id]
        self._save()

    def _not_agent_owned(self, sched_id: str) -> None:
        if sched_id in self._cron or sched_id in self._oneshots:
            raise ValueError(f"schedule {sched_id!r} is part of the base "
                             "schedule, which is read-only")
        raise ValueError(f"no such schedule: {sched_id!r}")

    # -- consumption (by the supervisor / sleep handler) ------------------------

    def peek_next(self, now: datetime) -> Trigger:
        """The SUPERVISOR's view: the earliest pending trigger, or — with an
        empty store — the fallback trigger at the next midnight, so a program
        that exited with nothing pending (a crashed or guard-exited resident
        agent, a cron program that wiped its schedule) is re-invoked and no
        run strands itself. Never used by a wait: see `peek_due`."""
        trig = self.peek_due(now)
        if trig is None:
            return Trigger(FALLBACK_ID, "fallback", next_midnight(now))
        return trig

    def peek_due(self, now: datetime) -> Trigger | None:
        """Earliest pending trigger at-or-after `now`; overdue one-shots fire
        now; None when nothing is scheduled. This is what a WAIT sees (sleep,
        wait party): no synthesized fallback, so nothing but the agent's own
        `until`, a real trigger or sim_end ends a wait.

        Ties break deterministically: one-shots < cron, then by id
        (agent rows rank with their kind: once with one-shots, recurring with
        cron — ids are unique across classes, so the tuple order is total).
        """
        candidates: list[tuple[datetime, int, str, TriggerKind]] = []
        for job_id, at in self._oneshots.items():
            candidates.append((max(at, now), 0, job_id, "at"))
        for entry_id, entry in self._cron.items():
            candidates.append((entry.next_fire(now), 1, entry_id, "cron"))
        for sched_id, row in self._agent.items():
            candidates.append((row.next_fire(now),
                               0 if row.at is not None else 1,
                               sched_id, "at" if row.at is not None else "cron"))
        if not candidates:
            return None
        due, _, trig_id, kind = min(candidates)
        row = self._agent.get(trig_id)
        if row is not None:
            return Trigger(trig_id, kind, due, owner="agent", note=row.note,
                           target=row.target)
        return Trigger(trig_id, kind, due)

    def consume(self, trigger: Trigger) -> None:
        """Mark a trigger as fired (one-shots are removed; cron records last fire)."""
        if trigger.owner == "agent":
            row = self._agent.get(trigger.id)
            if row is not None:
                if row.at is not None:
                    del self._agent[trigger.id]  # once: gone after firing
                else:
                    row.cron.last_fire = trigger.due_time
        elif trigger.kind == "at":
            self._oneshots.pop(trigger.id, None)
        elif trigger.kind == "cron":
            entry = self._cron.get(trigger.id)
            if entry is not None:
                entry.last_fire = trigger.due_time
        # fallback / crash_recovery: nothing stored
        self._save()

    # -- persistence -------------------------------------------------------------

    def _save(self) -> None:
        if self._persist_path is None:
            return
        data = {
            "crontab": [
                {"id": i, "cron_expr": e.expr, "installed_at": iso(e.installed_at),
                 "last_fire": iso(e.last_fire) if e.last_fire else None}
                for i, e in self._cron.items()
            ],
            "oneshots": {i: iso(at) for i, at in self._oneshots.items()},
            "agent": [r.to_json() for r in self._agent.values()],
            "seeded": sorted(self._seeded),
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self._persist_path)

    @classmethod
    def load(cls, persist_path: Path) -> "ScheduleStore":
        store = cls(persist_path)
        if persist_path.exists():
            data = json.loads(persist_path.read_text())
            for e in data["crontab"]:
                entry = _CronEntry(e["cron_expr"], parse_iso(e["installed_at"]),
                                   parse_iso(e["last_fire"]) if e["last_fire"] else None)
                store._cron[e["id"]] = entry
            store._oneshots = {i: parse_iso(at) for i, at in data["oneshots"].items()}
            for d in data.get("agent", []):  # absent in pre-TM-D run dirs
                store._agent[d["id"]] = _AgentSchedule.from_json(d)
            store._seeded = set(data.get("seeded", []))
        return store
