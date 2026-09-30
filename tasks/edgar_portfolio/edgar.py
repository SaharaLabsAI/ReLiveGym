"""EDGAR replay store + scorable-event builder for edgar_portfolio.

Layer-1 data: one snapshot of `submissions` and `companyfacts` per CIK
(data/fetch_edgar.py). The replayed `sec` host serves the SAME JSON
shapes as data.sec.gov, cut as of the sim instant:

- submissions: `filings.recent` keeps the filings with
  acceptanceDateTime <= now (the arrays stay newest-first; the multi-year
  history stays — last year's filing dates are the agent's legitimate
  planning signal);
- companyfacts / companyconcept: a fact is visible once the filing it
  came from (`accn`) was accepted; facts from filings older than the
  recent index are always visible.

Ground truth is the same files: for each holding, the earnings 8-K
(item 2.02) and the 10-Q/10-K accepted inside the run window, with the
XBRL single-period values of that accession (revenue, diluted EPS) for
the periodic report. Events are a pure function of (raw files, config),
materialized once at task build and listed in results.json.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from harness.timeutil import iso, parse_iso

ET = ZoneInfo("America/New_York")
ACCN_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")

# XBRL tags, in priority order (the first with a matching single-period
# row of the accession wins)
REVENUE_TAGS = ("Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenuesNetOfInterestExpense", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax")
EPS_TAGS = ("EarningsPerShareDiluted",)

# NYSE 2026 full-day closures (pinned; the agent gets the market status
# from the portal header, never this table)
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}
OPEN_ET = (9, 30)
CLOSE_ET = (16, 0)


# -- exchange calendar ---------------------------------------------------------------


def is_trading_day(d: date, holidays=NYSE_HOLIDAYS_2026) -> bool:
    return d.weekday() < 5 and d not in holidays


def session_bounds(d: date) -> tuple[datetime, datetime]:
    o = datetime(d.year, d.month, d.day, *OPEN_ET, tzinfo=ET)
    c = datetime(d.year, d.month, d.day, *CLOSE_ET, tzinfo=ET)
    return o.astimezone(timezone.utc), c.astimezone(timezone.utc)


def next_market_open(t: datetime, holidays=NYSE_HOLIDAYS_2026) -> datetime:
    """The first NYSE open strictly after t (UTC)."""
    d = t.astimezone(ET).date()
    for _ in range(14):
        if is_trading_day(d, holidays):
            o, _c = session_bounds(d)
            if o > t:
                return o
        d += timedelta(days=1)
    raise ValueError("no trading day within two weeks")


def market_status(t: datetime, holidays=NYSE_HOLIDAYS_2026) -> dict:
    d = t.astimezone(ET).date()
    if not is_trading_day(d, holidays):
        why = "holiday" if d in holidays else "weekend"
        return {"state": "closed", "reason": why,
                "next_open": next_market_open(t, holidays)}
    o, c = session_bounds(d)
    if t < o:
        return {"state": "pre-market", "reason": None, "next_open": o}
    if t < c:
        return {"state": "open", "reason": None, "next_open": next_market_open(t, holidays)}
    return {"state": "closed", "reason": "after hours",
            "next_open": next_market_open(t, holidays)}


# -- the store ---------------------------------------------------------------------------


def cik_path(cik: int) -> str:
    return f"CIK{cik:010d}"


@dataclass
class FilingEvent:
    """One scorable filing: the gold the portal row is compared with."""

    ticker: str
    cik: int
    form: str                       # 8-K | 10-Q | 10-K
    accession: str
    accepted_at: datetime
    deadline: datetime              # next NYSE open after acceptance
    fy: int | None = None
    fp: str | None = None
    revenue_usd: float | None = None
    eps_diluted: float | None = None
    revenue_tag: str | None = None
    period_end: str | None = None

    @property
    def id(self) -> str:
        return f"{self.ticker}:{self.accession}"

    def to_dict(self) -> dict:
        return {"id": self.id, "ticker": self.ticker, "cik": self.cik,
                "form": self.form, "accession": self.accession,
                "accepted_at": iso(self.accepted_at),
                "deadline": iso(self.deadline), "fy": self.fy, "fp": self.fp,
                "revenue_usd": self.revenue_usd, "eps_diluted": self.eps_diluted,
                "revenue_tag": self.revenue_tag, "period_end": self.period_end}


class EdgarStore:
    def __init__(self, raw_dir: Path, tickers: list[str]):
        self.raw_dir = Path(raw_dir)
        tick_path = self.raw_dir / "company_tickers.json"
        if not tick_path.exists():
            raise ValueError(f"no EDGAR snapshot at {self.raw_dir} (run "
                             "tasks/edgar_portfolio/data/fetch_edgar.py)")
        self.tickers_bytes = tick_path.read_bytes()
        by = {v["ticker"]: int(v["cik_str"])
              for v in json.loads(self.tickers_bytes).values()}
        self.cik: dict[str, int] = {}
        self.submissions: dict[int, dict] = {}
        self.facts: dict[int, dict] = {}
        self.accepted: dict[int, dict[str, float]] = {}      # cik -> accn -> ts
        self.accepted_sorted: dict[int, list[float]] = {}
        for t in tickers:
            if t not in by:
                raise ValueError(f"unknown ticker {t!r} in company_tickers.json")
            cik = by[t]
            self.cik[t] = cik
            sub_p = self.raw_dir / "submissions" / f"{cik_path(cik)}.json"
            fac_p = self.raw_dir / "companyfacts" / f"{cik_path(cik)}.json"
            if not sub_p.exists() or not fac_p.exists():
                raise ValueError(f"missing snapshot files for {t} ({cik_path(cik)})")
            sub = json.loads(sub_p.read_text())
            self.submissions[cik] = sub
            self.facts[cik] = json.loads(fac_p.read_text())
            rec = sub["filings"]["recent"]
            acc = {a: parse_iso(ts).timestamp()
                   for a, ts in zip(rec["accessionNumber"], rec["acceptanceDateTime"])}
            self.accepted[cik] = acc
            self.accepted_sorted[cik] = sorted(acc.values())
        self.ticker_of = {c: t for t, c in self.cik.items()}

    # -- as-of views ----------------------------------------------------------------

    def has_cik(self, cik: int) -> bool:
        return cik in self.submissions

    def submissions_asof(self, cik: int, now: datetime) -> dict:
        sub = self.submissions[cik]
        rec = sub["filings"]["recent"]
        now_ts = now.timestamp()
        keep = [i for i, ts in enumerate(rec["acceptanceDateTime"])
                if parse_iso(ts).timestamp() <= now_ts]
        out = {k: v for k, v in sub.items() if k != "filings"}
        out["filings"] = {"recent": {k: [v[i] for i in keep] for k, v in rec.items()},
                          "files": copy.deepcopy(sub["filings"].get("files", []))}
        return out

    def _visible(self, cik: int, accn: str, now_ts: float) -> bool:
        ts = self.accepted[cik].get(accn)
        return ts is None or ts <= now_ts  # older than the recent index: visible

    def companyfacts_asof(self, cik: int, now: datetime) -> dict:
        cf = self.facts[cik]
        now_ts = now.timestamp()
        facts = {}
        for taxonomy, tags in cf["facts"].items():
            tout = {}
            for tag, entry in tags.items():
                units = {u: [r for r in rows if self._visible(cik, r["accn"], now_ts)]
                         for u, rows in entry["units"].items()}
                units = {u: rows for u, rows in units.items() if rows}
                if units:
                    tout[tag] = {"label": entry.get("label"),
                                 "description": entry.get("description"),
                                 "units": units}
            if tout:
                facts[taxonomy] = tout
        return {"cik": cf["cik"], "entityName": cf["entityName"], "facts": facts}

    def companyconcept_asof(self, cik: int, taxonomy: str, tag: str,
                            now: datetime) -> dict | None:
        entry = (self.facts[cik]["facts"].get(taxonomy) or {}).get(tag)
        if entry is None:
            return None
        now_ts = now.timestamp()
        units = {u: [r for r in rows if self._visible(cik, r["accn"], now_ts)]
                 for u, rows in entry["units"].items()}
        units = {u: rows for u, rows in units.items() if rows}
        if not units:
            return None
        cf = self.facts[cik]
        return {"cik": cf["cik"], "taxonomy": taxonomy, "tag": tag,
                "label": entry.get("label"), "description": entry.get("description"),
                "entityName": cf["entityName"], "units": units}

    # -- ground truth -----------------------------------------------------------------

    def _period_row(self, cik: int, accn: str, tags: tuple[str, ...],
                    form: str) -> tuple[dict | None, str | None]:
        """The single-period XBRL row of `accn` for the first tag that has
        one: a quarter (70–100 days) for a 10-Q, a year (350–380) for a
        10-K, latest period end wins."""
        lo, hi = (70, 100) if form == "10-Q" else (350, 380)
        gaap = self.facts[cik]["facts"].get("us-gaap") or {}
        for tag in tags:
            entry = gaap.get(tag)
            if not entry:
                continue
            best = None
            for rows in entry["units"].values():
                for r in rows:
                    if r.get("accn") != accn or not r.get("start"):
                        continue
                    days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
                    if not lo <= days <= hi:
                        continue
                    if best is None or r["end"] > best["end"]:
                        best = r
            if best is not None:
                return best, tag
        return None, None

    def events(self, tickers: list[str], window_start: datetime,
               window_end: datetime, held_at) -> list[FilingEvent]:
        """Scorable filings of `tickers` accepted in [window_start,
        window_end) while held (`held_at(ticker, t) -> bool`), sorted by
        acceptance. 8-K events need items 2.02 (results of operations);
        periodic reports carry the XBRL gold."""
        out: list[FilingEvent] = []
        for t in tickers:
            cik = self.cik[t]
            rec = self.submissions[cik]["filings"]["recent"]
            for i, form in enumerate(rec["form"]):
                if form not in ("8-K", "10-Q", "10-K"):
                    continue
                acc_at = parse_iso(rec["acceptanceDateTime"][i])
                if not (window_start <= acc_at < window_end):
                    continue
                if form == "8-K" and "2.02" not in (rec["items"][i] or ""):
                    continue
                if not held_at(t, acc_at):
                    continue
                accn = rec["accessionNumber"][i]
                ev = FilingEvent(t, cik, form, accn, acc_at, next_market_open(acc_at))
                if form != "8-K":
                    rev, rtag = self._period_row(cik, accn, REVENUE_TAGS, form)
                    eps, _ = self._period_row(cik, accn, EPS_TAGS, form)
                    row = rev or eps
                    if row is not None:
                        ev.fy = row.get("fy")
                        ev.fp = row.get("fp")
                        ev.period_end = row.get("end")
                    if rev is not None:
                        ev.revenue_usd = float(rev["val"])
                        ev.revenue_tag = rtag
                    if eps is not None:
                        ev.eps_diluted = float(eps["val"])
                out.append(ev)
        out.sort(key=lambda e: (e.accepted_at, e.ticker, e.accession))
        return out
