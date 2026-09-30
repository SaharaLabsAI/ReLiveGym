"""broker_ops: the analyst's duties on a levered crypto account whose book
the PM runs. Two HTTP hosts, no task
tools: `mail` (read-only inbox: the PM's note and login codes), `api`
(the broker's read-only account API: events — fills, margin calls,
policy, treasury — account, orders, notices) and `broker` (writable,
login-gated: positions, the PM's orders, notices, the risk desk).

Ground truth is fixed: `world.canonical_schedule` derives every fill,
hike, margin call, treasury wire and grader probe from the config and the
candles alone; nothing the agent does changes the roster. Scoring: one
probe per duty instance at its deadline — margin-call response,
protective stop after a fill — credit 1/0; primary routine_score = mean
over the roster (max). Lockout zeroes the probes inside it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import as_utc, iso, parse_iso
from harness.webworld import Record
from tasks.broker_ops.world import BrokerOpsWorld
from tasks.broker_ops.candles import CandleStore

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent
# the replayed 5-minute candles of crypto_price_consistency (DATA.md)
DEFAULT_DATASETS = TASK_DIR.parent / "crypto_price_consistency" / "data" / "datasets"


class MaintenanceStep(BaseModel):
    at: datetime
    ratio: float

    @model_validator(mode="after")
    def _utc(self) -> "MaintenanceStep":
        self.at = as_utc(self.at)
        return self


class PmOrders(BaseModel):
    """The desk algorithm: one day-order per symbol at each hour, offset_pct
    off the mark, side alternating by placement; unfilled ones lapse at the
    next placement."""
    hours_utc: list[int] = [1, 9, 17]
    offset_pct: float = 1.0
    qty: dict[str, float] = {"BTC": 0.25, "ETH": 5.0}


class BrokerOpsConfig(BaseModel):
    username: str = "ops"
    password: str = "Spring-Portfolio-26!"
    symbols: list[str] = ["BTC", "ETH"]
    datasets_dir: Path | None = None
    dataset_suffix: str = "mar_jul"
    venue: str = "binance"
    positions: dict[str, float] = {"BTC": 3.0, "ETH": 40.0}
    cash_usd: float = -190000.0
    maintenance_ratio: float = 0.30
    maintenance_schedule: list[MaintenanceStep] = []
    call_target_margin: float = 0.05     # the wire brings the ratio to maintenance + this
    call_response_hours: float = 4.0
    call_cooldown_hours: float = 12.0
    pm_orders: PmOrders = PmOrders()
    stop_after_fill_hours: float = 2.0
    session_idle_minutes: float = 30.0
    session_cutoff_utc: tuple[int, int] = (0, 0)
    code_valid_minutes: float = 10.0
    max_failed_logins: int = 5
    lockout_hours: float = 24.0

    @model_validator(mode="after")
    def _consistent(self) -> "BrokerOpsConfig":
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be non-empty and unique")
        for s in self.symbols:
            if s not in self.pm_orders.qty:
                raise ValueError(f"pm_orders.qty has no entry for {s}")
        if not self.pm_orders.hours_utc or any(not 0 <= h < 24 for h in self.pm_orders.hours_utc):
            raise ValueError("pm_orders.hours_utc must be hours within a day")
        if self.pm_orders.offset_pct <= 0:
            raise ValueError("pm_orders.offset_pct must be positive")
        if self.call_target_margin <= 0:
            raise ValueError("call_target_margin must be positive")
        return self

    def resolve_datasets(self, base: Path) -> Path:
        p = self.datasets_dir or DEFAULT_DATASETS
        return p if p.is_absolute() else base / p


class BrokerOpsTask(Task):
    name = "broker_ops"
    # two-hourly: a firing at every even hour reaches every 4 h margin-call
    # deadline and every 2 h stop deadline except a fill on the hour of a
    # firing; tighter coverage is the agent's schedule to make
    episode_act_cron = "0 */2 * * *"

    def __init__(self, tcfg: BrokerOpsConfig, world: BrokerOpsWorld,
                 sim_start: datetime, sim_end: datetime):
        self.tcfg = tcfg
        self.world = world
        self.sim_start, self.sim_end = sim_start, sim_end
        self._sim = None
        world.on_write = self._ledger

    @classmethod
    def from_run_config(cls, cfg: "RunConfig", repo_root: Path) -> "BrokerOpsTask":
        tcfg = BrokerOpsConfig(**cfg.task_params)
        candles = CandleStore(tcfg.resolve_datasets(repo_root), tcfg.symbols,
                              tcfg.dataset_suffix, tcfg.venue)
        world = BrokerOpsWorld(tcfg, candles, cfg.sim_start, cfg.sim_end,
                               salt=f"{cls.name}:{cfg.seed}")
        return cls(tcfg, world, cfg.sim_start, cfg.sim_end)

    # -- binding + hosts -------------------------------------------------------------

    def bind(self, sim: "Sim") -> None:
        self._sim = sim

    def env_apps(self, sim: "Sim") -> list:
        return []

    def web_apps(self, sim: "Sim") -> list:
        from harness.web import WebHostSpec
        from tasks.broker_ops.web.api_app import make_api_app
        from tasks.broker_ops.web.broker_app import make_broker_app
        from tasks.broker_ops.web.mail_app import make_mail_app

        return [WebHostSpec("mail", make_mail_app(sim, self), writable=False),
                WebHostSpec("api", make_api_app(sim, self), writable=False),
                WebHostSpec("broker", make_broker_app(sim, self), writable=True)]

    # -- writes -----------------------------------------------------------------------

    def _ledger(self, rec: Record) -> None:
        if self._sim is not None:
            self._sim.ledger.append("notify", rec.t, payload=rec.payload())

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        if not isinstance(payload, dict) or payload.get("app") != "actions":
            raise NotificationError("malformed action payload")
        self.world.replay(sim_time, payload)

    def restore(self, events: list[dict], now: datetime) -> int:
        n = 0
        for e in events:
            t = parse_iso(e["sim_time"])
            if e.get("type") == "web" and e.get("session"):
                self.close_due(t)
                s = self.world.session_for(e["session"], t, create=True)
                if s is not None:
                    self.world.sessions.touch(s.id, t)
            elif e.get("type") == "notify":
                self.close_due(t)
                self.record_notification(t, dict(e.get("payload") or {}))
                n += 1
        self.close_due(now)
        return n

    # -- scoring ----------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.world.sync(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.world.sync(self.sim_end)

    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        out = []
        for pid in self.world.settled_order:
            d = self.world.settled[pid]
            t = parse_iso(d["deadline"])
            if (since is None or t > since) and t <= now:
                out.append({"kind": "routine", "t_settled": iso(t), **d})
        return out

    def metrics(self) -> dict:
        w = self.world
        settled = [w.settled[p] for p in w.settled_order]
        credits = [s["credit"] for s in settled]

        def sub(routine):
            xs = [s["credit"] for s in settled if s["routine"] == routine]
            return {"n": len(xs), "score": (round(sum(xs) / len(xs), 4) if xs else None)}

        lat = []
        for s in settled:
            if s["credit"] and s.get("done_at"):
                t0, t1, td = parse_iso(s["t_event"]), parse_iso(s["deadline"]), parse_iso(s["done_at"])
                lat.append((td - t0) / (t1 - t0))
        recs = w.journal.records
        logins = [r for r in recs if r.kind == "login_password"]
        return {
            "primary": {"name": "routine_score",
                        "value": (round(sum(credits) / len(credits), 4)
                                  if credits else None), "direction": "max"},
            "instances": len(settled), "roster": len(w.roster),
            "credited": int(sum(credits)),
            "by_routine": {r: sub(r) for r in ("margin_call", "stop_after_fill")},
            "latency_frac": (round(sum(lat) / len(lat), 4) if lat else None),
            "status_counts": {k: sum(1 for s in settled if s["status"] == k)
                              for k in ("ok", "miss", "locked_out")},
            "logins_ok": sum(1 for r in logins if r.fields.get("ok")),
            "logins_failed": sum(1 for r in logins if not r.fields.get("ok")),
            "lockouts": len(w.sessions.lockouts),
            "stops_registered": len(w.journal.items("stops")),
            "rejections": sum(w.rejections.values()),
        }

    def report(self) -> dict:
        w = self.world
        end = self.sim_end
        return {"outcomes": {p: w.settled[p] for p in w.settled_order},
                "roster": list(w.roster),
                "schedule": [e.to_dict() for e in w.schedule],
                "account_end": w.account(end),
                "orders": w.orders_view(), "notices": w.notices_view(),
                "stops": w.stops_view(),
                "lockouts": [(u, iso(a), iso(b)) for u, a, b in w.sessions.lockouts],
                "actions": [r.to_dict() for r in w.journal.records
                            if r.actor == "agent"][-500:]}

    # -- authored wait programs (TM-B) ---------------------------------------------------

    def authored_example(self) -> str | None:
        return (TASK_DIR / "agent" / "example_gatekeeper.py").read_text(encoding="utf-8")

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        t = self.tcfg
        acct = self.world.account(self.sim_start)
        pos = "\n".join(f"| {s} | {q:g} |" for s, q in acct["positions"].items())
        return {
            "username": t.username, "password": t.password,
            "symbols": ", ".join(t.symbols), "positions_table": pos,
            "cash_usd": f"{acct['cash_usd']:,.0f}",
            "maintenance_pct": f"{t.maintenance_ratio:.0%}",
            "pm_hours": ", ".join(f"{h:02d}:00" for h in t.pm_orders.hours_utc),
            "call_hours": f"{t.call_response_hours:g}",
            "stop_hours": f"{t.stop_after_fill_hours:g}",
            "idle_minutes": f"{t.session_idle_minutes:g}",
            "cutoff_utc": f"{t.session_cutoff_utc[0]:02d}:{t.session_cutoff_utc[1]:02d}",
            "code_minutes": f"{t.code_valid_minutes:g}",
            "max_failed": t.max_failed_logins,
            "lockout_hours": f"{t.lockout_hours:g}",
            "sim_start": iso(self.sim_start), "sim_end": iso(self.sim_end),
        }


TASK = BrokerOpsTask
