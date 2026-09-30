"""The broker_ops world: a levered
crypto account whose book the PM runs, and two event-triggered duties for
the analyst — a margin-call response within `call_response_hours`, a
protective stop within `stop_after_fill_hours` of each fill.

The ground truth is FIXED: `canonical_schedule` computes, from the config
and the candles alone, every PM placement, expiry, fill, maintenance hike,
margin call, treasury settlement and grader probe. The world adds them
all to the timeline at construction (RANK_SCRIPTED) and its handlers
apply the payload deltas — never recompute — so the live account equals
the precompute whatever the agent does. Agent actions (login, a funding
instruction, a protective-stop registration) are journaled and probed but
move neither positions nor cash: the treasury wires the required amount
at the call deadline in every run, and protective stops are risk-desk
instructions, not orders in the book.

Probes (credit 1/0, primary = mean over the fixed roster):
  call:<nid>   the notice was responded to before its deadline
  stop:<oid>   a protective stop is registered for that fill
A probe whose instant falls inside an account lockout scores 0
(status locked_out) — the catastrophic single-wake mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from harness.task import NotificationError, OutcomeEvent
from harness.timeutil import iso
from harness.webworld import (
    RANK_PROBE,
    RANK_SCRIPTED,
    Journal,
    Record,
    SessionStore,
    Timeline,
)
from tasks.broker_ops.candles import CandleStore

SIDES = ("buy", "sell")


@dataclass
class Scheduled:
    t: datetime
    kind: str
    payload: dict = field(default_factory=dict)
    id: str | None = None

    def to_dict(self) -> dict:
        return {"t": iso(self.t), "kind": self.kind, "id": self.id, **self.payload}


def _account(positions: dict, cash: float, marks: dict) -> tuple[float, float, float | None]:
    value = sum(positions[s] * marks[s] for s in positions)
    exposure = sum(abs(positions[s]) * marks[s] for s in positions)
    equity = cash + value
    return equity, exposure, (equity / exposure if exposure > 0 else None)


def canonical_schedule(tcfg, candles: CandleStore, t0: datetime,
                       t1: datetime) -> tuple[list[Scheduled], dict]:
    """Every world event of the run in (t, insertion) order, and the
    final account state {positions, cash, maintenance} it leads to.
    Pure: config + candles in, schedule out."""
    pm = tcfg.pm_orders
    placements: list[datetime] = []
    d = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while d < t1:
        for h in pm.hours_utc:
            tp = d + timedelta(hours=h)
            if t0 <= tp < t1:
                placements.append(tp)
        d += timedelta(days=1)
    placements.sort()
    hikes = sorted(((m.at, m.ratio) for m in tcfg.maintenance_schedule
                    if t0 <= m.at < t1), key=lambda x: x[0])
    instants = sorted({c.close_time for s in candles.symbols
                       for c in candles.between(s, t0, t1)})
    by_t = {s: {c.close_time: c for c in candles.between(s, t0, t1)}
            for s in candles.symbols}

    positions = {s: float(tcfg.positions.get(s, 0.0)) for s in candles.symbols}
    cash = float(tcfg.cash_usd)
    maint = float(tcfg.maintenance_ratio)
    out: list[Scheduled] = []
    orders: list[dict] = []          # open PM orders
    n_orders = n_notices = 0
    pi = hi = 0
    last_call: datetime | None = None
    settle_at: datetime | None = None
    settle_amount = 0.0
    stop_h = timedelta(hours=tcfg.stop_after_fill_hours)
    call_h = timedelta(hours=tcfg.call_response_hours)
    cooldown = timedelta(hours=tcfg.call_cooldown_hours)

    def place(tp: datetime, k: int) -> None:
        nonlocal n_orders, orders
        expired = [o["id"] for o in orders]
        if expired:
            out.append(Scheduled(tp, "pm_expire", {"order_ids": expired}))
        orders = []
        for j, s in enumerate(candles.symbols):
            mark = candles.mark(s, tp)
            side = SIDES[(k + j) % 2]
            price = round(mark * (1 - pm.offset_pct / 100 if side == "buy"
                                  else 1 + pm.offset_pct / 100), 2)
            n_orders += 1
            o = {"id": f"o{n_orders:04d}", "symbol": s, "side": side,
                 "qty": float(pm.qty[s]), "price": price}
            orders.append({**o, "placed": tp})
            out.append(Scheduled(tp, "pm_place", o))

    for t in instants:
        while pi < len(placements) and placements[pi] <= t:
            place(placements[pi], pi)
            pi += 1
        # fills on the candle closing at t (orders placed strictly before it)
        for o in list(orders):
            c = by_t[o["symbol"]].get(t)
            if c is None or o["placed"] >= t:
                continue
            hit = c.low <= o["price"] if o["side"] == "buy" else c.high >= o["price"]
            if hit:
                orders.remove(o)
                sign = 1 if o["side"] == "buy" else -1
                positions[o["symbol"]] += sign * o["qty"]
                cash -= sign * o["qty"] * o["price"]
                ev = Scheduled(t, "fill", {"order_id": o["id"], "symbol": o["symbol"],
                                           "side": o["side"], "qty": o["qty"],
                                           "price": o["price"]})
                if t + stop_h <= t1:
                    ev.payload["probe_at"] = iso(t + stop_h)
                out.append(ev)
        if settle_at is not None and t >= settle_at:
            cash += settle_amount
            settle_at = None
        while hi < len(hikes) and hikes[hi][0] <= t:
            maint = float(hikes[hi][1])
            out.append(Scheduled(hikes[hi][0], "maintenance", {"ratio": maint}))
            hi += 1
        marks = {s: candles.mark(s, t) for s in candles.symbols}
        equity, exposure, ratio = _account(positions, cash, marks)
        if (ratio is not None and ratio < maint and settle_at is None
                and (last_call is None or t - last_call >= cooldown)):
            required = round((maint + tcfg.call_target_margin) * exposure - equity, 2)
            n_notices += 1
            nid = f"mc{n_notices:03d}"
            deadline = t + call_h
            ev = Scheduled(t, "margin_call",
                           {"notice_id": nid, "required_usd": required,
                            "margin_ratio": round(ratio, 4), "maintenance_ratio": maint,
                            "deadline": iso(deadline)})
            out.append(ev)
            last_call = t
            if deadline <= t1:
                ev.payload["probe_at"] = iso(deadline)
                out.append(Scheduled(deadline, "treasury_settle",
                                     {"notice_id": nid, "amount_usd": required}))
                settle_at, settle_amount = deadline, required
            else:
                settle_at = t1  # no further call inside the window
    # stable order: time, then insertion (placements at tp precede fills at tp)
    out = [Scheduled(e.t, e.kind, e.payload, e.id) for _, e in
           sorted(enumerate(out), key=lambda ie: (ie[1].t, ie[0]))]
    final = {"positions": positions, "cash_usd": cash, "maintenance_ratio": maint}
    return out, final


class BrokerOpsWorld:
    def __init__(self, tcfg, candles: CandleStore, sim_start: datetime,
                 sim_end: datetime, salt: str):
        self.tcfg = tcfg
        self.candles = candles
        self.sim_start, self.sim_end = sim_start, sim_end
        self.journal = Journal(salt)
        self.timeline = Timeline(sim_start)
        self.sessions = SessionStore(
            salt, idle_minutes=tcfg.session_idle_minutes,
            cutoff_utc=tuple(tcfg.session_cutoff_utc),
            code_valid_minutes=tcfg.code_valid_minutes,
            max_failed=tcfg.max_failed_logins, lockout_hours=tcfg.lockout_hours)
        self.password = tcfg.password
        self.maintenance = float(tcfg.maintenance_ratio)
        self.cash = float(tcfg.cash_usd)
        self.positions: dict[str, float] = {s: float(tcfg.positions.get(s, 0.0))
                                            for s in candles.symbols}
        self.n_mail = self.n_events = 0
        self.settled: dict[str, dict] = {}
        self.settled_order: list[str] = []
        self.rejections: dict[str, int] = {}
        self._replaying = False
        self.on_write: Callable[[Record], None] | None = None  # ledger hook (task)

        for s in candles.symbols:
            lo, hi = candles.coverage(s)
            if lo > sim_start or hi < sim_end:
                raise ValueError(f"candles for {s} cover {iso(lo)}..{iso(hi)}, "
                                 f"the run needs {iso(sim_start)}..{iso(sim_end)}")
        self.schedule, self.final_state = canonical_schedule(tcfg, candles, sim_start, sim_end)
        self.roster: list[dict] = []   # the fixed duty instances, in deadline order
        for ev in self.schedule:
            if ev.kind == "fill" and "probe_at" in ev.payload:
                self.roster.append({"id": f"stop:{ev.payload['order_id']}",
                                    "routine": "stop_after_fill", "t_event": iso(ev.t),
                                    "deadline": ev.payload["probe_at"]})
            elif ev.kind == "margin_call" and "probe_at" in ev.payload:
                self.roster.append({"id": f"call:{ev.payload['notice_id']}",
                                    "routine": "margin_call", "t_event": iso(ev.t),
                                    "deadline": ev.payload["probe_at"]})
            self.timeline.add(ev.t, ev.kind, ev.payload, rank=RANK_SCRIPTED)
        self.roster.sort(key=lambda r: (r["deadline"], r["id"]))
        self._mail(sim_start, "Desk arrangements while I am away",
                   "The desk algorithm works my book: it enters one day-order per "
                   "symbol at " + ", ".join(f"{h:02d}:00" for h in tcfg.pm_orders.hours_utc)
                   + " UTC, about " + f"{tcfg.pm_orders.offset_pct:g}% off the market, "
                   "and lets unfilled ones lapse at the next entry. You do not trade.\n\n"
                   "Your duties: respond to every margin call on the portal before its "
                   "deadline, and register a protective stop with the risk desk after "
                   "every fill. The broker's account API reports each fill and each "
                   "call the moment it happens.\n\nThanks — PM",
                   kind="instructions")

    # -- account maths ---------------------------------------------------------------

    def marks(self, now: datetime) -> dict[str, float]:
        return {s: self.candles.mark(s, now) or 0.0 for s in self.candles.symbols}

    def account(self, now: datetime) -> dict:
        marks = self.marks(now)
        equity, exposure, ratio = _account(self.positions, self.cash, marks)
        return {"cash_usd": self.cash, "positions": dict(self.positions),
                "marks": marks, "value_usd": equity - self.cash,
                "exposure_usd": exposure, "equity_usd": equity,
                "margin_ratio": ratio, "maintenance_ratio": self.maintenance}

    # -- world writes ------------------------------------------------------------------

    def _write(self, t: datetime, actor: str, app: str, kind: str,
               entity: str | None, fields: dict | None = None, **kw) -> Record:
        rec = self.journal.write(t, actor=actor, app=app, kind=kind,
                                 entity=entity, fields=fields, **kw)
        if actor == "agent" and self.on_write is not None and not self._replaying:
            self.on_write(rec)
        return rec

    def _mail(self, t: datetime, subject: str, body: str, kind: str) -> str:
        self.n_mail += 1
        mid = f"m{self.n_mail:04d}"
        self._write(t, "world", "mail", "set", mid,
                    {"at": iso(t), "subject": subject, "body": body, "kind": kind})
        return mid

    def _event(self, t: datetime, kind: str, fields: dict) -> str:
        """One row of the broker's account-events API (fill, margin_call,
        policy, treasury) — the machine channel a watcher polls."""
        self.n_events += 1
        eid = f"e{self.n_events:04d}"
        self._write(t, "world", "events", "set", eid, {"at": iso(t), "kind": kind, **fields})
        return eid

    # -- sync ------------------------------------------------------------------------

    def sync(self, now: datetime) -> list[OutcomeEvent]:
        out: list[OutcomeEvent] = []
        for ev in self.timeline.drain(now):
            res = getattr(self, f"_on_{ev.kind}")(ev.t, ev.payload)
            if res is not None:
                out.append(res)
        return out

    def open_orders(self) -> list[dict]:
        return [st for st in self.journal.items("orders")
                if st["fields"]["status"] == "open"]

    # -- scripted handlers (payload deltas only) -----------------------------------------

    def _on_pm_place(self, t: datetime, p: dict) -> None:
        self._write(t, "world", "orders", "set", p["id"],
                    {"symbol": p["symbol"], "side": p["side"], "type": "limit",
                     "qty": p["qty"], "price": p["price"], "status": "open",
                     "placed_at": iso(t), "owner": "PM"})

    def _on_pm_expire(self, t: datetime, p: dict) -> None:
        for oid in p["order_ids"]:
            self._write(t, "world", "orders", "update", oid,
                        {"status": "expired", "expired_at": iso(t)})

    def _on_fill(self, t: datetime, p: dict) -> None:
        sign = 1 if p["side"] == "buy" else -1
        self.positions[p["symbol"]] += sign * p["qty"]
        self.cash -= sign * p["qty"] * p["price"]
        self._write(t, "world", "orders", "update", p["order_id"],
                    {"status": "filled", "filled_at": iso(t), "fill_price": p["price"]})
        self._event(t, "fill", {"order_id": p["order_id"], "symbol": p["symbol"],
                                "side": p["side"], "qty": p["qty"], "price": p["price"],
                                "mark": self.candles.mark(p["symbol"], t)})
        if "probe_at" in p:
            self.timeline.add(self._t(p["probe_at"]), "probe_stop",
                              {"order_id": p["order_id"], "symbol": p["symbol"],
                               "side": p["side"], "t_event": iso(t)},
                              rank=RANK_PROBE, id=f"stop:{p['order_id']}")

    def _on_maintenance(self, t: datetime, p: dict) -> None:
        self.maintenance = float(p["ratio"])
        self._event(t, "policy", {"maintenance_ratio": self.maintenance,
                                  "effective_at": iso(t)})

    def _on_margin_call(self, t: datetime, p: dict) -> None:
        nid = p["notice_id"]
        self._write(t, "world", "notices", "set", nid,
                    {"kind": "margin_call", "issued_at": iso(t), "deadline": p["deadline"],
                     "required_usd": p["required_usd"], "margin_ratio": p["margin_ratio"],
                     "maintenance_ratio": p["maintenance_ratio"], "status": "open"})
        self._event(t, "margin_call", {"notice_id": nid, "required_usd": p["required_usd"],
                                       "deadline": p["deadline"],
                                       "margin_ratio": p["margin_ratio"],
                                       "maintenance_ratio": p["maintenance_ratio"]})
        if "probe_at" in p:
            self.timeline.add(self._t(p["probe_at"]), "probe_call",
                              {"notice_id": nid, "t_event": iso(t)},
                              rank=RANK_PROBE, id=f"call:{nid}")

    def _on_treasury_settle(self, t: datetime, p: dict) -> None:
        self.cash += float(p["amount_usd"])
        self._event(t, "treasury", {"notice_id": p["notice_id"],
                                    "amount_usd": float(p["amount_usd"])})

    @staticmethod
    def _t(s: str) -> datetime:
        from harness.timeutil import parse_iso
        return parse_iso(s)

    # -- probes ---------------------------------------------------------------------

    def _locked(self, t: datetime) -> bool:
        return self.sessions.locked_until(self.tcfg.username, t) is not None

    def _settle(self, t: datetime, pid: str, routine: str, ok: bool,
                t_event: str, done_at: str | None, detail: dict) -> OutcomeEvent:
        status = "ok" if ok else "miss"
        if self._locked(t):
            ok, status = False, "locked_out"
        d = {"id": pid, "routine": routine, "t_event": t_event, "deadline": iso(t),
             "done_at": done_at if ok else None, "credit": 1.0 if ok else 0.0, **detail}
        self.settled[pid] = {"status": status, **d}
        self.settled_order.append(pid)
        return OutcomeEvent(pid, status, dict(d))

    def _on_probe_call(self, t: datetime, p: dict) -> OutcomeEvent:
        st = self.journal.get("notices", p["notice_id"])
        f = st["fields"]
        ok = f["status"] == "responded"
        if not ok:
            self._write(t, "world", "notices", "update", p["notice_id"],
                        {"status": "expired"})
        return self._settle(t, f"call:{p['notice_id']}", "margin_call", ok,
                            p["t_event"], f.get("responded_at"),
                            {"notice_id": p["notice_id"], "required_usd": f["required_usd"],
                             "amount_usd": f.get("amount_usd")})

    def _on_probe_stop(self, t: datetime, p: dict) -> OutcomeEvent:
        st = self.journal.get("stops", p["order_id"])
        return self._settle(t, f"stop:{p['order_id']}", "stop_after_fill", st is not None,
                            p["t_event"], st["fields"]["registered_at"] if st else None,
                            {"order_id": p["order_id"], "symbol": p["symbol"],
                             "fill_side": p["side"]})

    # -- agent actions (live from the web handlers; replayed on restore) ----------------

    def _reject(self, reason: str, msg: str):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        raise NotificationError(msg)

    def session_for(self, sid: str | None, now: datetime, create: bool = False):
        if create and sid and sid not in self.sessions.sessions:
            self.sessions.ensure(sid, now)
        return self.sessions.get(sid, now)

    def act(self, now: datetime, kind: str, session: str | None, fields: dict,
            form_token: str | None = None, replay: bool = False) -> Record:
        """One agent action: validated against the state at `now`, applied,
        journaled — the same path live and on restore."""
        fields = dict(fields or {})
        s = self.session_for(session, now, create=replay)
        if s is None:
            self._reject("no_session", "no session — open the portal first")
        u = self.tcfg.username
        if kind == "login_password":
            until = self.sessions.locked_until(u, now)
            if until is not None:
                fields["ok"] = False
                self._record(now, kind, s.id, fields, form_token, replay)
                self._reject("locked", f"account locked until {iso(until)}")
            if replay:  # the recorded outcome is the truth; the password is never journaled
                ok = bool(fields.get("ok"))
            else:
                ok = (fields.get("username") == u and fields.get("password") == self.password)
            fields = {"username": fields.get("username"), "ok": ok}
            if ok:
                self.sessions.password_ok(s, u, now)
                self._mail(now, "Your one-time login code",
                           f"Your code is {s.code}. It expires in "
                           f"{self.tcfg.code_valid_minutes:g} minutes.", kind="code")
            else:
                lock = self.sessions.password_failed(u, now)
                if lock is not None:
                    fields["locked_until"] = iso(lock)
            rec = self._record(now, kind, s.id, fields, form_token, replay)
            if not ok:
                self._reject("bad_password", "invalid username or password"
                             + (f"; account locked until {fields['locked_until']}"
                                if fields.get("locked_until") else ""))
            return rec
        if kind == "login_code":
            code = str(fields.get("code", ""))
            ok = self.sessions.code_ok(s, code, now)
            if replay and fields.get("ok") and not ok:
                self.sessions.authenticate(s, u, now)
                ok = True
            rec = self._record(now, kind, s.id, {"code": code, "ok": ok},
                               form_token, replay)
            if not ok:
                self._reject("bad_code", "invalid or expired code — log in again")
            return rec
        if not s.authenticated:
            self._reject("not_authenticated", "your session is not signed in "
                         "(expired or never logged in); the action was not applied")
        self.sessions.touch(s.id, now)
        if kind == "logout":
            self.sessions.logout(s, now)
            return self._record(now, kind, s.id, {}, form_token, replay)
        if kind == "notice_respond":
            nid = fields.get("notice_id")
            st = self.journal.get("notices", nid or "")
            if st is None or st["fields"]["status"] != "open":
                self._reject("bad_notice", "no open notice with that id")
            try:
                amount = float(fields.get("amount_usd"))
            except (TypeError, ValueError):
                self._reject("bad_amount", "amount must be a number")
            if amount < st["fields"]["required_usd"]:
                self._reject("short_amount", "the funding instruction must cover at least "
                             f"the required ${st['fields']['required_usd']:,.2f}")
            rec = self._record(now, kind, s.id, {"notice_id": nid, "amount_usd": amount},
                               form_token, replay)
            self._write(now, "world", "notices", "update", nid,
                        {"status": "responded", "responded_at": iso(now),
                         "amount_usd": amount})
            return rec
        if kind == "stop_register":
            oid = fields.get("order_id")
            st = self.journal.get("orders", oid or "")
            if st is None or st["fields"]["status"] != "filled":
                self._reject("bad_fill", "no filled order with that id")
            if self.journal.get("stops", oid) is not None:
                self._reject("duplicate_stop", f"a protective stop for {oid} is already registered")
            side = fields.get("side")
            want = "sell" if st["fields"]["side"] == "buy" else "buy"
            if side != want:
                self._reject("wrong_side", f"{oid} was a {st['fields']['side']} fill: "
                             f"its protective stop is a {want} stop")
            try:
                qty, trig = float(fields.get("qty")), float(fields.get("trigger_price"))
            except (TypeError, ValueError):
                self._reject("bad_stop", "qty and trigger price must be numbers")
            if qty <= 0 or trig <= 0:
                self._reject("bad_stop", "qty and trigger price must be positive")
            mark = self.candles.mark(st["fields"]["symbol"], now) or 0.0
            if (side == "sell" and trig >= mark) or (side == "buy" and trig <= mark):
                self._reject("bad_trigger", f"a {side} stop triggers "
                             f"{'below' if side == 'sell' else 'above'} the market "
                             f"(mark {mark:,.2f})")
            rec = self._record(now, kind, s.id,
                               {"order_id": oid, "side": side, "qty": qty, "trigger_price": trig},
                               form_token, replay)
            self._write(now, "world", "stops", "set", oid,
                        {"order_id": oid, "symbol": st["fields"]["symbol"], "side": side,
                         "qty": qty, "trigger_price": trig, "registered_at": iso(now)})
            return rec
        self._reject("unknown_action", f"unknown action {kind!r}")

    def _record(self, now, kind, sid, fields, form_token, replay) -> Record:
        return self._write(now, "agent", "actions", kind, None, fields,
                           session=sid, form_token=form_token, replay=replay)

    def replay(self, t: datetime, payload: dict) -> None:
        self._replaying = True
        try:
            try:
                self.act(t, payload["kind"], payload.get("session"),
                         payload.get("fields") or {}, payload.get("form_token"),
                         replay=True)
            except NotificationError:
                pass  # a rejected attempt was journaled as such; state unchanged
        finally:
            self._replaying = False

    # -- views ------------------------------------------------------------------------

    def orders_view(self) -> list[dict]:
        return [{"id": st["entity"], **st["fields"]}
                for st in sorted(self.journal.items("orders"),
                                 key=lambda s: s["entity"], reverse=True)]

    def notices_view(self) -> list[dict]:
        return [{"id": st["entity"], **st["fields"]}
                for st in sorted(self.journal.items("notices"),
                                 key=lambda s: s["entity"], reverse=True)]

    def stops_view(self) -> list[dict]:
        return [{"id": st["entity"], **st["fields"]}
                for st in sorted(self.journal.items("stops"),
                                 key=lambda s: s["fields"]["registered_at"], reverse=True)]

    def fills_without_stop(self) -> list[dict]:
        return [o for o in self.orders_view() if o["status"] == "filled"
                and self.journal.get("stops", o["id"]) is None]

    def events_view(self) -> list[dict]:
        return [{"id": st["entity"], **st["fields"]}
                for st in sorted(self.journal.items("events"),
                                 key=lambda s: (s["fields"]["at"], s["entity"]),
                                 reverse=True)]

    def mail_view(self) -> list[dict]:
        return [{"id": st["entity"], **st["fields"]}
                for st in sorted(self.journal.items("mail"),
                                 key=lambda s: (s["fields"]["at"], s["entity"]),
                                 reverse=True)]


def main(argv: list[str] | None = None) -> None:
    """Print the canonical schedule of a run yaml:
    python -m tasks.broker_ops.world tasks/broker_ops/configs/cells/tmA-tlrnnone-signone-algnone.yaml"""
    import sys
    from pathlib import Path

    from harness.config import load_config
    from tasks.broker_ops.task import TASK

    path = Path((argv or sys.argv[1:])[0])
    root = Path(__file__).resolve().parents[2]
    task = TASK.from_run_config(load_config(path), root)
    w = task.world
    for ev in w.schedule:
        extra = {k: v for k, v in ev.payload.items() if k not in ("symbol",)}
        print(f"{iso(ev.t)}  {ev.kind:16s} {extra}")
    by = {}
    for r in w.roster:
        by[r["routine"]] = by.get(r["routine"], 0) + 1
    print(f"\nroster: {len(w.roster)} instances {by}; final {w.final_state}")


if __name__ == "__main__":
    main()
