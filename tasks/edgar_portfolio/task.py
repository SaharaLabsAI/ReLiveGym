"""edgar_portfolio: keep the fund's filings sheet current from a replayed
EDGAR.

The world is two HTTP hosts and no task tools. `sec` (read-only) mirrors
data.sec.gov as of the sim instant; `portal` (writable) is the filings
sheet the agent fills in through a form. For every held ticker, the
earnings 8-K (item 2.02) and the 10-Q/10-K accepted inside the run are
the scorable events; each is a grader probe at its deadline — the next
NYSE open after acceptance — that reads the sheet.

Credit per event (1/0): the record keyed by (ticker, accession) exists,
matches the XBRL gold (form, acceptance time, fy/fp, revenue, diluted
EPS where the filing reports them), its last write lies in
[accepted_at, deadline), and the ledger shows a fetch of that CIK on the
`sec` host between acceptance and the write (source-visible-before-write:
no fabrication, no premature guessing). Primary metric filing_score =
mean credit over events (max). Everything is deterministic given the
snapshot, the config and the agent's writes.

State model: the journal (harness/webworld.py) holds the holdings sheet
(world-written: the initial book and the scripted blotter) and the
filings sheet (agent-written); the timeline runs the blotter and the
probes inside close_due(now). World writes are regenerated from config
on restore; agent writes replay from the `notify` rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import as_utc, iso, parse_iso
from harness.webworld import (
    RANK_PROBE,
    RANK_SCRIPTED,
    Journal,
    Record,
    Timeline,
)
from tasks.edgar_portfolio.edgar import EdgarStore, FilingEvent, cik_path

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = TASK_DIR / "data" / "raw"


class BlotterEntry(BaseModel):
    at: datetime
    ticker: str
    action: Literal["buy", "sell"]
    shares: int = 0

    @model_validator(mode="after")
    def _utc(self) -> "BlotterEntry":
        self.at = as_utc(self.at)
        self.ticker = self.ticker.upper()
        return self


class EdgarPortfolioConfig(BaseModel):
    holdings: dict[str, int]                 # ticker -> shares at sim_start
    blotter: list[BlotterEntry] = []         # scripted roster changes
    raw_dir: Path | None = None              # default: data/raw snapshot
    portal_access: Literal["browser", "api"] = "browser"
    # grading tolerances (INSTRUCTION states them)
    acceptance_tolerance_s: float = 60.0
    revenue_tolerance_frac: float = 0.0005   # rounding to millions passes
    eps_tolerance: float = 0.005
    # the mirror's rate limit; none by default (design D12)
    sec_rate_limit: dict = {"window": "none"}

    @model_validator(mode="after")
    def _consistent(self) -> "EdgarPortfolioConfig":
        if not self.holdings:
            raise ValueError("holdings must be non-empty")
        self.holdings = {t.upper(): int(n) for t, n in self.holdings.items()}
        for e in self.blotter:
            if e.action == "buy" and e.shares <= 0:
                raise ValueError(f"blotter buy of {e.ticker} needs shares > 0")
        return self

    def resolve_raw_dir(self, base: Path) -> Path:
        p = self.raw_dir or DEFAULT_RAW_DIR
        return p if p.is_absolute() else base / p


# -- the world ---------------------------------------------------------------------------


class PortfolioWorld:
    def __init__(self, tcfg: EdgarPortfolioConfig, store: EdgarStore,
                 sim_start: datetime, sim_end: datetime, salt: str):
        self.tcfg = tcfg
        self.store = store
        self.sim_start, self.sim_end = sim_start, sim_end
        self.journal = Journal(salt)
        self.timeline = Timeline(sim_start)
        self.tickers = sorted(set(tcfg.holdings)
                              | {e.ticker for e in tcfg.blotter})
        for t in self.tickers:
            if t not in store.cik:
                raise ValueError(f"ticker {t} is not in the EDGAR snapshot")
        # the initial book: world writes at t0 (regenerated on restore)
        for t, n in sorted(tcfg.holdings.items()):
            self.journal.write(sim_start, actor="world", app="holdings",
                               kind="set", entity=t,
                               fields={"shares": n, "status": "held",
                                       "since": iso(sim_start)})
        for e in sorted(tcfg.blotter, key=lambda e: (e.at, e.ticker)):
            if e.at < sim_start:
                raise ValueError(f"blotter entry {e.ticker}@{iso(e.at)} lies "
                                 "before sim_start")
            if e.at >= sim_end:
                continue  # never happens in this run (e.g. a stretched window)
            self.timeline.add(e.at, "blotter", {"ticker": e.ticker,
                                                "action": e.action,
                                                "shares": e.shares},
                              rank=RANK_SCRIPTED)
        self.events: list[FilingEvent] = store.events(
            self.tickers, sim_start, sim_end, self.held_at)
        for ev in self.events:
            ev.deadline = min(ev.deadline, sim_end)
        self.by_id = {ev.id: ev for ev in self.events}
        for ev in self.events:
            self.timeline.add(ev.deadline, "probe", {"event": ev.id},
                              rank=RANK_PROBE, id=f"probe:{ev.id}")
        self.settled: dict[str, dict] = {}
        self.settled_order: list[str] = []
        self.rejections: dict[str, int] = {}
        self.ledger_events = None  # bound by the task (evidence rule)

    # -- roster ---------------------------------------------------------------------

    def held_at(self, ticker: str, t: datetime) -> bool:
        held = ticker in self.tcfg.holdings
        for e in sorted(self.tcfg.blotter, key=lambda e: e.at):
            if e.ticker != ticker or e.at > t:
                continue
            held = e.action == "buy"
        return held

    def known_ticker(self, t: str) -> bool:
        return t in self.tickers

    # -- sync: the lazy timeline ----------------------------------------------------

    def sync(self, now: datetime) -> list[OutcomeEvent]:
        out: list[OutcomeEvent] = []
        for ev in self.timeline.drain(now):
            if ev.kind == "blotter":
                self._apply_blotter(ev.t, ev.payload)
            elif ev.kind == "probe":
                out.append(self._probe(ev.t, self.by_id[ev.payload["event"]]))
        return out

    def _apply_blotter(self, t: datetime, p: dict) -> None:
        if p["action"] == "buy":
            fields = {"shares": p["shares"], "status": "held", "since": iso(t)}
        else:
            fields = {"shares": 0, "status": "sold", "since": iso(t)}
        self.journal.write(t, actor="world", app="holdings", kind="update",
                           entity=p["ticker"], fields=fields)

    # -- grading ----------------------------------------------------------------------

    def _evidence(self, cik: int, lo: datetime, hi: datetime) -> bool:
        if self.ledger_events is None:
            return False
        needle = cik_path(cik)
        lo_s, hi_s = iso(lo), iso(hi)
        for e in self.ledger_events:
            if e.get("type") != "web" or e.get("host") != "sec":
                continue
            if needle not in (e.get("path") or ""):
                continue
            if lo_s <= e["sim_time"] <= hi_s:
                return True
        return False

    def _matches(self, ev: FilingEvent, f: dict) -> list[str]:
        """Field names that disagree with the gold (empty = correct)."""
        bad = []
        if f.get("form") != ev.form:
            bad.append("form")
        try:
            acc = parse_iso(f.get("accepted_at") or "")
            if abs((acc - ev.accepted_at).total_seconds()) > self.tcfg.acceptance_tolerance_s:
                bad.append("accepted_at")
        except (ValueError, TypeError):
            bad.append("accepted_at")
        if ev.fy is not None and f.get("fy") != ev.fy:
            bad.append("fy")
        if ev.fp is not None and f.get("fp") != ev.fp:
            bad.append("fp")
        if ev.revenue_usd is not None:
            v = f.get("revenue_usd")
            tol = abs(ev.revenue_usd) * self.tcfg.revenue_tolerance_frac
            if v is None or abs(float(v) - ev.revenue_usd) > tol:
                bad.append("revenue_usd")
        if ev.eps_diluted is not None:
            v = f.get("eps_diluted")
            if v is None or abs(float(v) - ev.eps_diluted) > self.tcfg.eps_tolerance:
                bad.append("eps_diluted")
        return bad

    def _probe(self, t: datetime, ev: FilingEvent) -> OutcomeEvent:
        st = self.journal.get("filings", ev.id)
        detail: dict = {"id": ev.id, "ticker": ev.ticker, "form": ev.form,
                        "accepted_at": iso(ev.accepted_at), "deadline": iso(t),
                        "credit": 0.0}
        if st is None:
            status = "miss"
        else:
            t_w = st["updated_at"]
            detail["recorded_at"] = iso(t_w)
            detail["delay_h"] = round((t_w - ev.accepted_at).total_seconds() / 3600, 3)
            wrong = self._matches(ev, st["fields"])
            timely = ev.accepted_at <= t_w < t
            evidence = timely and self._evidence(ev.cik, ev.accepted_at, t_w)
            detail.update(wrong_fields=wrong, timely=timely, evidence=evidence)
            if t_w < ev.accepted_at:
                status = "premature"
            elif wrong:
                status = "wrong"
            elif not evidence:
                status = "no_evidence"
            else:
                status = "ok"
                detail["credit"] = 1.0
        self.settled[ev.id] = {"status": status, **detail}
        self.settled_order.append(ev.id)
        return OutcomeEvent(ev.id, status, dict(detail))

    # -- views for the portal -------------------------------------------------------

    def holding_view(self, ticker: str) -> dict:
        st = self.journal.get("holdings", ticker)
        cik = self.store.cik[ticker]
        f = st["fields"] if st else {"shares": 0, "status": "not held", "since": None}
        return {"ticker": ticker, "cik": cik_path(cik),
                "name": self.store.submissions[cik].get("name", ""),
                "shares": f.get("shares", 0), "status": f.get("status"),
                "since": parse_iso(f["since"]) if f.get("since") else None}

    def filing_records(self, ticker: str) -> list[dict]:
        rows = []
        for st in self.journal.items("filings"):
            if st["entity"].split(":")[0] != ticker:
                continue
            rows.append({**st["fields"], "updated_at": st["updated_at"],
                         "version": st["version"]})
        rows.sort(key=lambda r: r.get("accepted_at") or "", reverse=True)
        return rows

    def holdings_view(self) -> list[dict]:
        out = []
        for t in self.tickers:
            h = self.holding_view(t)
            if h["status"] == "not held":
                continue  # bought later: appears once the desk trades
            recs = self.filing_records(t)
            out.append({**h, "n_records": len(recs),
                        "last": recs[0] if recs else None})
        return out

    def filing_rows_json(self) -> list[dict]:
        return [{"ticker": st["entity"].split(":")[0], **st["fields"],
                 "updated_at": iso(st["updated_at"]), "version": st["version"]}
                for st in self.journal.items("filings")]


# -- the task ------------------------------------------------------------------------


class EdgarPortfolioTask(Task):
    name = "edgar_portfolio"
    # daily at midnight UTC (20:00 ET), the cron mains' daily convention: a wake
    # can reach 42 of the smoke roster's 63 filings; the pre-open third
    # needs a schedule of the agent's own
    episode_act_cron = "0 0 * * *"

    def __init__(self, tcfg: EdgarPortfolioConfig, store: EdgarStore,
                 world: PortfolioWorld, sim_start: datetime, sim_end: datetime):
        self.tcfg = tcfg
        self.store = store
        self.world = world
        self.sim_start, self.sim_end = sim_start, sim_end
        self._restoring = False
        self._sim = None

    @classmethod
    def from_run_config(cls, cfg: "RunConfig", repo_root: Path) -> "EdgarPortfolioTask":
        tcfg = EdgarPortfolioConfig(**cfg.task_params)
        raw = tcfg.resolve_raw_dir(repo_root)
        tickers = sorted(set(tcfg.holdings) | {e.ticker for e in tcfg.blotter})
        store = EdgarStore(raw, tickers)
        # tokens/ids hash a salt that depends on the config, never the
        # run id: identical configs give byte-identical ledgers
        world = PortfolioWorld(tcfg, store, cfg.sim_start, cfg.sim_end,
                               salt=f"{cls.name}:{cfg.seed}")
        if not world.events:
            raise ValueError("no scorable filing lies inside the run window "
                             "for these holdings")
        return cls(tcfg, store, world, cfg.sim_start, cfg.sim_end)

    # -- binding + hosts -------------------------------------------------------------

    def bind(self, sim: "Sim") -> None:
        self._sim = sim
        self.world.ledger_events = sim.ledger.events

    def env_apps(self, sim: "Sim") -> list:
        return []  # the world is HTTP ; clock/waits are the harness apps

    def web_apps(self, sim: "Sim") -> list:
        from harness.web import WebHostSpec
        from tasks.edgar_portfolio.web.portal_app import make_portal_app
        from tasks.edgar_portfolio.web.sec_app import make_sec_app

        # the api arm is the scripted control: curl/PUT writes are the
        # point there, so the browser-only refusal (harness/web.py) is off
        return [WebHostSpec("sec", make_sec_app(sim, self), writable=False),
                WebHostSpec("portal", make_portal_app(sim, self), writable=True,
                            browser_only=self.tcfg.portal_access != "api")]

    # -- writes -----------------------------------------------------------------------

    def write(self, t: datetime, **kw) -> Record:
        """One agent write from a portal handler: journal it and ledger the
        `notify` row (the restore path replays exactly this payload)."""
        rec = self.world.journal.write(t, **kw)
        if self._sim is not None:
            self._sim.ledger.append("notify", t, payload=rec.payload())
        return rec

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        # only the restore path arrives here (the live write path is
        # `write`): replay the journal payload verbatim
        if not isinstance(payload, dict) or "app" not in payload:
            raise NotificationError("malformed journal payload")
        self.world.journal.replay(sim_time, payload)

    def restore(self, events: list[dict], now: datetime) -> int:
        self._restoring = True
        try:
            return super().restore(events, now)
        finally:
            self._restoring = False

    # -- scoring ----------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.world.sync(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.world.sync(self.sim_end)

    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        out = []
        for eid in self.world.settled_order:
            d = self.world.settled[eid]
            t_settled = parse_iso(d["deadline"])
            if (since is None or t_settled > since) and t_settled <= now:
                out.append({"kind": "filing", "id": eid, "t_settled": iso(t_settled), **d})
        return out

    def metrics(self) -> dict:
        w = self.world
        settled = [w.settled[e] for e in w.settled_order]
        credits = [s["credit"] for s in settled]
        mean = round(sum(credits) / len(credits), 4) if credits else None

        def sub(pred):
            xs = [s["credit"] for s in settled if pred(s)]
            return round(sum(xs) / len(xs), 4) if xs else None

        sharp = {e.id for e in w.events
                 if (e.deadline - e.accepted_at) < timedelta(hours=6)}
        delays = [s["delay_h"] for s in settled if s["status"] == "ok"]
        return {
            "primary": {"name": "filing_score", "value": mean, "direction": "max"},
            "events": len(w.events), "events_settled": len(settled),
            "credited": int(sum(credits)),
            "score_8k": sub(lambda s: s["form"] == "8-K"),
            "score_periodic": sub(lambda s: s["form"] != "8-K"),
            "score_sharp": sub(lambda s: s["id"] in sharp),
            "score_overnight": sub(lambda s: s["id"] not in sharp),
            "status_counts": {k: sum(1 for s in settled if s["status"] == k)
                              for k in ("ok", "miss", "wrong", "premature",
                                        "no_evidence")},
            "delay_h_mean_ok": (round(sum(delays) / len(delays), 3)
                                if delays else None),
            "records": len(w.journal.items("filings")),
            "rejections": sum(w.rejections.values()),
        }

    def report(self) -> dict:
        w = self.world
        return {"events": [e.to_dict() for e in w.events],
                "outcomes": {e: w.settled[e] for e in w.settled_order},
                "pending": [e.id for e in w.events if e.id not in w.settled],
                "filings_sheet": w.filing_rows_json(),
                "holdings": [{k: (iso(v) if isinstance(v, datetime) else v)
                              for k, v in h.items() if k != "last"}
                             for h in w.holdings_view()]}

    # -- authored wait programs (TM-B) ---------------------------------------------------

    def authored_example(self) -> str | None:
        return (TASK_DIR / "agent" / "example_gatekeeper.py").read_text(
            encoding="utf-8")

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        w = self.world
        rows = "\n".join(f"| {t} | {cik_path(self.store.cik[t])} | {n} |"
                         for t, n in sorted(self.tcfg.holdings.items()))
        return {
            "holdings_table": rows,
            "n_holdings": len(self.tcfg.holdings),
            "sim_start": iso(self.sim_start),
            "sim_end": iso(self.sim_end),
            "acceptance_tolerance_s": f"{self.tcfg.acceptance_tolerance_s:g}",
            "revenue_tolerance_pct": f"{self.tcfg.revenue_tolerance_frac * 100:g}",
            "eps_tolerance": f"{self.tcfg.eps_tolerance:g}",
            "api_arm_section": (API_SECTION if self.tcfg.portal_access == "api"
                                else ""),
        }


API_SECTION = """\
## Sheet API (this run)

Besides the pages, the sheet accepts JSON writes:

- `PUT ${portal_url}/portfolio/api/rows/<TICKER>/<ACCESSION>` with a JSON
  object of the same fields as the form (`form`, `accepted_at`, `fy`, `fp`,
  `revenue_usd`, `eps_diluted`, `note`) saves or replaces that record.
- `GET ${portal_url}/portfolio/api/rows` lists every record as JSON.
"""

TASK = EdgarPortfolioTask
