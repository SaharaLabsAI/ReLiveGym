"""World mechanics shared by web tasks: the three layers of state under the one sim clock.

- Layer 1, exogenous replay, needs nothing here: a time-indexed store
  queried as of `now` (see `asof_index`).
- Layer 2, agent writes: `Journal` — every accepted write is one record
  (the task ledgers it as a `notify` row, so `Task.restore` replays it);
  the current state of a writable entity is the fold over its records.
  Single-use form tokens make a double submit a no-op; optimistic
  versions let a world amendment refuse a stale edit with a visible
  message.
- Layer 3, derived state: `Timeline` — the ONLY "event queue", and a lazy
  one: scripted world events, timers and grader probes on one heap,
  drained in (time, rank, seq) order whenever the clock has advanced
  (the task calls `drain` from `close_due(now)`). Never a background
  task; never wall time. `SessionStore` — cookie sessions whose idle and
  absolute expiry, one-time codes and lockouts are evaluated lazily in
  sim time at lookup.

Everything is deterministic given the config, the seed salt and the
sequence of writes: ids and tokens are counters hashed with the salt,
never `secrets`.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterator

from harness.task import NotificationError
from harness.timeutil import iso, parse_iso

# ranks: what applies first at an equal instant
RANK_REPLAY = 0     # replay-derived effects (a fill at a candle instant)
RANK_SCRIPTED = 1   # scripted world events (a notice posted, a roster change)
RANK_TIMER = 2      # derived timers (an expiry, a lockout clearing)
RANK_PROBE = 3      # grader checkpoints — see the world BEFORE any action at t


class StaleVersion(NotificationError):
    """The record changed since the form was rendered (world amendment or
    a concurrent write) — the portal's "reload and retry" message."""


class UnknownToken(NotificationError):
    """A write carrying a form token that was never minted for this run."""


def _digest(salt: str, *parts: object) -> str:
    return hashlib.sha256(":".join([salt, *map(str, parts)]).encode()).hexdigest()


# -- layer 2: the journal ------------------------------------------------------------


@dataclass
class Record:
    seq: int
    t: datetime
    actor: str          # "agent" | "world"
    app: str
    kind: str           # "set" | "update" | "delete" | anything the task folds itself
    entity: str | None
    fields: dict
    session: str | None
    form_token: str | None
    version: int        # the entity's version AFTER this record
    duplicate: bool = False  # a re-submitted form token: recorded, not applied

    def payload(self) -> dict:
        """The `notify` payload this record is replayed from."""
        d = {"actor": self.actor, "app": self.app, "kind": self.kind,
             "entity": self.entity, "fields": dict(self.fields)}
        if self.session is not None:
            d["session"] = self.session
        if self.form_token is not None:
            d["form_token"] = self.form_token
        return d

    def to_dict(self) -> dict:
        return {"seq": self.seq, "t": iso(self.t), **self.payload(),
                "version": self.version, "duplicate": self.duplicate}


class Journal:
    """Append-only writes + the fold. `write` is the one entry point for
    agent AND world writes (actor tags them); the task calls it from its
    web handlers (agent) and its timeline handlers (world), and ledgers
    each returned record as a `notify` row with `record.payload()`.
    `replay` is `write` for the restore path (tokens are trusted)."""

    def __init__(self, salt: str, fold: Callable[[dict, Record], None] | None = None):
        self._salt = salt
        self._fold = fold or default_fold
        self.records: list[Record] = []
        self.entities: dict[tuple[str, str], dict] = {}
        self._minted: dict[str, datetime] = {}
        self._consumed: set[str] = set()
        self._n_tokens = 0

    # -- form tokens ------------------------------------------------------------

    def mint_token(self, now: datetime) -> str:
        self._n_tokens += 1
        tok = _digest(self._salt, "form", self._n_tokens)[:12]
        self._minted[tok] = now
        return tok

    def token_minted(self, tok: str) -> bool:
        return tok in self._minted

    # -- writes -----------------------------------------------------------------

    def write(self, t: datetime, *, actor: str, app: str, kind: str,
              entity: str | None, fields: dict | None = None,
              session: str | None = None, form_token: str | None = None,
              expect_version: int | None = None,
              replay: bool = False) -> Record:
        fields = dict(fields or {})
        key = (app, entity) if entity is not None else None
        current = self.entities.get(key) if key else None
        version = int(current["version"]) if current else 0
        duplicate = False
        if form_token is not None:
            if form_token in self._consumed:
                duplicate = True
            elif not replay and form_token not in self._minted:
                raise UnknownToken("this form is no longer valid — reload the "
                                   "page and submit again")
            self._consumed.add(form_token)
        if not duplicate and expect_version is not None and expect_version != version:
            raise StaleVersion("this record changed since you opened it — "
                               "reload the page and submit again")
        rec = Record(seq=len(self.records) + 1, t=t, actor=actor, app=app,
                     kind=kind, entity=entity, fields=fields, session=session,
                     form_token=form_token, version=version, duplicate=duplicate)
        if not duplicate and key is not None:
            state = self.entities.setdefault(
                key, {"app": app, "entity": entity, "fields": {}, "version": 0,
                      "created_at": t, "updated_at": t, "updated_by": actor,
                      "deleted": False})
            self._fold(state, rec)
            state["version"] = version + 1
            state["updated_at"] = t
            state["updated_by"] = actor
            rec.version = state["version"]
        self.records.append(rec)
        return rec

    def replay(self, t: datetime, payload: dict) -> Record:
        """Restore path: apply a ledgered `notify` payload verbatim."""
        p = dict(payload)
        return self.write(t, actor=p.get("actor", "agent"), app=p["app"],
                          kind=p["kind"], entity=p.get("entity"),
                          fields=p.get("fields"), session=p.get("session"),
                          form_token=p.get("form_token"), replay=True)

    # -- reads ------------------------------------------------------------------

    def get(self, app: str, entity: str) -> dict | None:
        st = self.entities.get((app, entity))
        return None if st is None or st["deleted"] else st

    def items(self, app: str, include_deleted: bool = False) -> list[dict]:
        out = [st for (a, _), st in self.entities.items()
               if a == app and (include_deleted or not st["deleted"])]
        return sorted(out, key=lambda s: s["entity"])

    def history(self, app: str, entity: str) -> list[Record]:
        return [r for r in self.records if r.app == app and r.entity == entity]


def default_fold(state: dict, rec: Record) -> None:
    """set/update merge fields; delete tombstones; other kinds are
    recorded against the entity without touching its fields."""
    if rec.kind in ("set", "update"):
        if rec.kind == "set":
            state["fields"] = {}
        state["fields"].update(rec.fields)
        state["deleted"] = False
    elif rec.kind == "delete":
        state["deleted"] = True


# -- layer 3: the timeline -----------------------------------------------------------


@dataclass
class Event:
    id: str
    t: datetime
    rank: int
    kind: str
    payload: dict = field(default_factory=dict)
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "t": iso(self.t), "rank": self.rank,
                "kind": self.kind, "payload": dict(self.payload)}


class Timeline:
    """One heap of (t, rank, seq) events; `drain(now)` yields every event
    due by `now` in order and moves `t_synced` along. Handlers may add
    events at or after the event being handled (never before
    `t_synced`). `scan(t_from, t_to)` — optional, from the task — finds
    the earliest replay×state interaction in (t_from, t_to] and adds it
    (rank RANK_REPLAY), so a fill at 03:15 is handled before the notice
    at 03:20 and before the probe at 08:00, with the state each of them
    should see."""

    def __init__(self, t0: datetime):
        self.t_synced = t0
        self._heap: list[tuple[datetime, int, int, str]] = []
        self._events: dict[str, Event] = {}
        self._seq = 0

    def add(self, t: datetime, kind: str, payload: dict | None = None, *,
            rank: int = RANK_TIMER, id: str | None = None) -> Event:
        if t < self.t_synced:
            raise ValueError(f"timeline: cannot add {kind!r} at {iso(t)} "
                             f"before t_synced {iso(self.t_synced)}")
        self._seq += 1
        eid = id or f"{kind}#{self._seq}"
        if eid in self._events and not self._events[eid].cancelled:
            raise ValueError(f"timeline: duplicate event id {eid!r}")
        ev = Event(eid, t, rank, kind, dict(payload or {}))
        self._events[eid] = ev
        heapq.heappush(self._heap, (t, rank, self._seq, eid))
        return ev

    def cancel(self, eid: str) -> bool:
        ev = self._events.get(eid)
        if ev is None or ev.cancelled:
            return False
        ev.cancelled = True
        return True

    def pending(self) -> list[Event]:
        return sorted((e for e in self._events.values() if not e.cancelled),
                      key=lambda e: (e.t, e.rank, e.id))

    def _peek(self) -> Event | None:
        while self._heap:
            t, rank, seq, eid = self._heap[0]
            ev = self._events[eid]
            if ev.cancelled:
                heapq.heappop(self._heap)
                self._events.pop(eid, None)
                continue
            return ev
        return None

    def drain(self, now: datetime,
              scan: Callable[[datetime, datetime], datetime | None] | None = None
              ) -> Iterator[Event]:
        """Yield due events in order up to `now`, then set t_synced = now.
        The caller handles each yielded event before the next is chosen,
        so handler-added events take their place in the order."""
        if now < self.t_synced:
            raise ValueError(f"timeline: now {iso(now)} < t_synced "
                             f"{iso(self.t_synced)}")
        while True:
            nxt = self._peek()
            t = nxt.t if nxt is not None and nxt.t <= now else now
            if scan is not None and t > self.t_synced:
                first = scan(self.t_synced, t)
                if first is not None:
                    if not (self.t_synced < first <= t):
                        raise ValueError("timeline: scan returned an instant "
                                         "outside (t_synced, t]")
                    t = first
                    nxt = self._peek()
            if nxt is None or nxt.t > t:
                self.t_synced = t
                if t >= now:
                    return
                continue
            heapq.heappop(self._heap)
            self._events.pop(nxt.id, None)
            self.t_synced = nxt.t
            yield nxt


# -- sessions ----------------------------------------------------------------------


@dataclass
class Session:
    id: str
    created: datetime
    last_seen: datetime
    user: str | None = None
    authenticated_at: datetime | None = None
    pending_user: str | None = None      # password ok, code outstanding
    code: str | None = None
    code_issued_at: datetime | None = None
    ended_at: datetime | None = None
    end_reason: str | None = None        # idle | cutoff | logout

    @property
    def authenticated(self) -> bool:
        return self.authenticated_at is not None and self.ended_at is None


class SessionStore:
    """Cookie sessions in sim time. `get(sid, now)` is the one lookup:
    it applies idle expiry (now - last_seen >= idle), the daily absolute
    cutoff (the first cutoff instant after authentication ends the
    session) and returns the session — ended or not — so a handler can
    render "session expired" instead of silently dropping a write.
    Lockouts are per user: `max_failed` consecutive bad passwords lock
    the account for `lockout_hours` (catastrophic by design)."""

    def __init__(self, salt: str, *, idle_minutes: float = 30.0,
                 cutoff_utc: tuple[int, int] | None = (0, 0),
                 code_valid_minutes: float = 10.0, max_failed: int = 5,
                 lockout_hours: float = 24.0):
        self._salt = salt
        self.idle = timedelta(minutes=idle_minutes)
        self.cutoff_utc = cutoff_utc
        self.code_valid = timedelta(minutes=code_valid_minutes)
        self.max_failed = max_failed
        self.lockout = timedelta(hours=lockout_hours)
        self.sessions: dict[str, Session] = {}
        self.failed: dict[str, int] = {}
        self.lock_until: dict[str, datetime] = {}
        self.lockouts: list[tuple[str, datetime, datetime]] = []  # (user, from, until)
        self._n = 0

    # -- ids ---------------------------------------------------------------------

    def new_session(self, now: datetime) -> Session:
        self._n += 1
        sid = "s" + _digest(self._salt, "session", self._n)[:16]
        s = Session(sid, now, now)
        self.sessions[sid] = s
        return s

    def ensure(self, sid: str, now: datetime) -> Session:
        """The session with this id, created if unknown (restore: ids come
        from the ledger, not from the counter)."""
        s = self.sessions.get(sid)
        if s is None:
            s = Session(sid, now, now)
            self.sessions[sid] = s
        return s

    def authenticate(self, s: Session, user: str, now: datetime) -> None:
        s.user, s.pending_user = user, None
        s.code, s.code_issued_at = None, None
        s.authenticated_at = now
        s.last_seen = now
        s.ended_at, s.end_reason = None, None

    def next_cutoff_after(self, t: datetime) -> datetime | None:
        if self.cutoff_utc is None:
            return None
        h, m = self.cutoff_utc
        c = t.replace(hour=h, minute=m, second=0, microsecond=0)
        return c if c > t else c + timedelta(days=1)

    # -- lookup with lazy expiry ------------------------------------------------

    def get(self, sid: str | None, now: datetime) -> Session | None:
        if not sid:
            return None
        s = self.sessions.get(sid)
        if s is None:
            return None
        if s.authenticated:
            cutoff = self.next_cutoff_after(s.authenticated_at)
            if cutoff is not None and now >= cutoff:
                s.ended_at, s.end_reason = cutoff, "cutoff"
            elif now - s.last_seen >= self.idle:
                s.ended_at, s.end_reason = s.last_seen + self.idle, "idle"
        return s

    def touch(self, sid: str, now: datetime) -> None:
        s = self.sessions.get(sid)
        if s is not None and s.ended_at is None:
            s.last_seen = now

    # -- login flow ----------------------------------------------------------------

    def locked_until(self, user: str, now: datetime) -> datetime | None:
        until = self.lock_until.get(user)
        return until if until is not None and now < until else None

    def password_ok(self, s: Session, user: str, now: datetime) -> None:
        """Password accepted: the session awaits the one-time code."""
        self.failed[user] = 0
        s.pending_user = user
        s.code = _digest(self._salt, "code", s.id, len(self.lockouts),
                         iso(now))[:6].upper()
        s.code_issued_at = now

    def password_failed(self, user: str, now: datetime) -> datetime | None:
        """Returns the lock-until instant when this failure locks the
        account, else None."""
        n = self.failed.get(user, 0) + 1
        self.failed[user] = n
        if n >= self.max_failed:
            until = now + self.lockout
            self.lock_until[user] = until
            self.lockouts.append((user, now, until))
            self.failed[user] = 0
            return until
        return None

    def code_ok(self, s: Session, code: str, now: datetime) -> bool:
        if not s.pending_user or s.code is None:
            return False
        if now - s.code_issued_at >= self.code_valid:
            return False
        if code.strip().upper() != s.code:
            return False
        self.authenticate(s, s.pending_user, now)
        return True

    def logout(self, s: Session, now: datetime) -> None:
        if s.ended_at is None:
            s.ended_at, s.end_reason = now, "logout"


# -- layer 1 helper --------------------------------------------------------------------


def asof_index(times: list[float], now: datetime) -> int:
    """Number of entries of a sorted unix-time list visible at `now`
    (t <= now) — the as-of cut of any time-indexed store."""
    return bisect.bisect_right(times, now.timestamp())


def parse_dt(s: str) -> datetime:
    return parse_iso(s)
