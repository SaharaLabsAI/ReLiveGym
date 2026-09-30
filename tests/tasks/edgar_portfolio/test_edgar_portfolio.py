"""edgar_portfolio:
as-of leak pins on every scorable filing, the NYSE calendar, scoring
anchors through a SERVED sim with its two web hosts (perfect client
1.000, premature 0.000, a 07:00-ET daily poller = the pre-open miss
rate), ledger determinism, restore, and the non-browser write flag.
Skips without the EDGAR snapshot (data/raw, gitignored)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from harness.config import RunConfig
from harness.mcp import _Http
from harness.runtime import Sim
from harness.timeutil import iso, parse_iso
from tasks.edgar_portfolio.edgar import (
    ET,
    EdgarStore,
    FilingEvent,
    cik_path,
    market_status,
    next_market_open,
)
from tasks.edgar_portfolio.roster import ALL_TICKERS
from tasks.edgar_portfolio.task import TASK
from tests.control_plane.test_detached import Served, ledger_view
from tests.control_plane.test_mcp_episode import settle

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "tasks" / "edgar_portfolio" / "data" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "manifest.json").exists(),
                                reason="EDGAR snapshot not fetched (data/raw)")
UTC = timezone.utc
START = datetime(2026, 4, 1, tzinfo=UTC)
END = datetime(2026, 6, 1, tzinfo=UTC)
HOLD = {"AAPL": 100, "IBM": 200, "JPM": 300, "MSFT": 50}
BLOTTER = [{"at": "2026-04-20T15:00:00Z", "ticker": "DIS", "action": "buy",
            "shares": 80},
           {"at": "2026-04-20T15:00:00Z", "ticker": "MSFT", "action": "sell"}]
BROWSER = {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document",
           "Sec-Fetch-Site": "same-origin"}


def make_cfg(run_id="edgar-test", **task_over) -> RunConfig:
    return RunConfig(run_id=run_id,
                     task={"name": "edgar_portfolio", "holdings": HOLD,
                           "blotter": BLOTTER, **task_over},
                     sim_start=START, sim_end=END,
                     cell={"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"},
                     budget_usd=20.0, agent={"scaffold": "ext:mcp"})


@pytest.fixture(scope="module")
def gold() -> list[FilingEvent]:
    return TASK.from_run_config(make_cfg(), REPO_ROOT).world.events


# -- calendar + store ---------------------------------------------------------------------


def test_calendar_pins():
    # Good Friday 2026-04-03: a Thursday-evening filing waits for Monday
    thu = datetime(2026, 4, 2, 21, 0, tzinfo=UTC)
    assert next_market_open(thu) == datetime(2026, 4, 6, 13, 30, tzinfo=UTC)
    # a pre-open filing has the same day's open
    pre = datetime(2026, 4, 14, 10, 32, tzinfo=UTC)
    assert next_market_open(pre) == datetime(2026, 4, 14, 13, 30, tzinfo=UTC)
    # Memorial Day 2026-05-25
    fri = datetime(2026, 5, 22, 20, 30, tzinfo=UTC)
    assert next_market_open(fri) == datetime(2026, 5, 26, 13, 30, tzinfo=UTC)
    assert market_status(datetime(2026, 4, 3, 15, 0, tzinfo=UTC))["reason"] == "holiday"
    assert market_status(datetime(2026, 4, 14, 14, 0, tzinfo=UTC))["state"] == "open"


def test_events_and_roster(gold):
    ids = {e.id for e in gold}
    tickers = {e.ticker for e in gold}
    assert {"AAPL", "IBM", "JPM", "DIS"} <= tickers
    assert "MSFT" not in tickers  # sold 2026-04-20, files 2026-04-29
    for e in gold:
        assert e.deadline == min(next_market_open(e.accepted_at), END)
        if e.form == "8-K":
            assert e.fy is None and e.revenue_usd is None
    aapl = [e for e in gold if e.ticker == "AAPL" and e.form == "10-Q"][0]
    assert aapl.fy == 2026 and aapl.fp == "Q2" and aapl.eps_diluted is not None
    assert len(ids) == len(gold)


def test_asof_never_leaks_full_roster():
    """Every scorable filing of the full roster: invisible one second
    before acceptance on submissions AND companyfacts/companyconcept,
    visible at acceptance."""
    store = EdgarStore(RAW, ALL_TICKERS)
    events = store.events(ALL_TICKERS, START, END, lambda t, at: True)
    assert len(events) >= 60
    for e in events:
        before = store.submissions_asof(e.cik, e.accepted_at - timedelta(seconds=1))
        after = store.submissions_asof(e.cik, e.accepted_at)
        assert e.accession not in before["filings"]["recent"]["accessionNumber"]
        assert e.accession in after["filings"]["recent"]["accessionNumber"]
        if e.form != "8-K" and e.eps_diluted is not None:
            tag = "EarningsPerShareDiluted"
            cb = store.companyconcept_asof(e.cik, "us-gaap", tag,
                                           e.accepted_at - timedelta(seconds=1))
            ca = store.companyconcept_asof(e.cik, "us-gaap", tag, e.accepted_at)
            rows_b = [r for u in (cb or {}).get("units", {}).values() for r in u]
            rows_a = [r for u in ca["units"].values() for r in u]
            assert not any(r["accn"] == e.accession for r in rows_b)
            assert any(r["accn"] == e.accession for r in rows_a)
            facts = store.companyfacts_asof(e.cik, e.accepted_at - timedelta(seconds=1))
            eps = facts["facts"]["us-gaap"].get(tag, {"units": {}})
            assert not any(r["accn"] == e.accession
                           for u in eps["units"].values() for r in u)


# -- a scripted actor against a served sim ----------------------------------------------


class Actor:
    def __init__(self, handle: dict, headers: dict | None = BROWSER):
        self.env = _Http(handle["env_url"], handle["agent_token"])
        self.sec = handle["hosts"]["sec"]
        self.portal = handle["hosts"]["portal"]
        self.http = httpx.Client(headers=headers or {}, follow_redirects=False,
                                 timeout=60)

    def now(self) -> datetime:
        return parse_iso(self.env.call("get_time", {})["now"])

    def sleep(self, until: datetime) -> dict:
        """Sleep to `until`: a sleep wakes early at any due trigger, so
        loop as an agent must."""
        while True:
            r = self.env.call("sleep", {"until": iso(until)})
            if r.get("experiment_over") or parse_iso(r["now"]) >= until:
                return r

    def fetch_submissions(self, cik: int) -> dict:
        r = self.http.get(f"{self.sec}/submissions/{cik_path(cik)}.json")
        assert r.status_code == 200, r.text[:200]
        assert "x-sim-time" in r.headers
        return r.json()

    def visible(self, e: FilingEvent) -> bool:
        sub = self.fetch_submissions(e.cik)
        return e.accession in sub["filings"]["recent"]["accessionNumber"]

    def record(self, e: FilingEvent, **override) -> httpx.Response:
        page = self.http.get(f"{self.portal}/portfolio/{e.ticker}")
        assert page.status_code == 200, page.text[:200]
        token = re.search(r'name="form_token" value="([^"]+)"', page.text).group(1)
        version = re.search(r'name="version" value="([^"]+)"', page.text).group(1)
        fields = {"accession": e.accession, "form": e.form,
                  "accepted_at": iso(e.accepted_at),
                  "fy": e.fy if e.fy is not None else "",
                  "fp": e.fp or "",
                  "revenue_usd": f"{e.revenue_usd:.0f}" if e.revenue_usd is not None else "",
                  "eps_diluted": f"{e.eps_diluted:.2f}" if e.eps_diluted is not None else "",
                  "note": "", "form_token": token, "version": version}
        fields.update(override)
        return self.http.post(f"{self.portal}/portfolio/{e.ticker}", data=fields)


def serve(tmp_path, name: str, **task_over):
    cfg = make_cfg(run_id=f"edgar-{name}", **task_over)
    srv = Served(cfg, REPO_ROOT, tmp_path / name / "server")
    h = srv.handle
    ctl = _Http(h["env_url"], h["token"])
    trig = ctl.get("/trigger/next")  # the controller consumes __bootstrap__
    return srv, h, ctl, trig


def finish(srv, ctl, trig) -> dict:
    settle(ctl, trig)
    return srv.join()


def perfect_run(tmp_path, gold, name="perfect"):
    srv, h, ctl, trig = serve(tmp_path, name)
    a = Actor(h)
    for e in gold:
        t = e.accepted_at + timedelta(minutes=1)
        if t > a.now():
            a.sleep(t)
        assert a.visible(e)
        r = a.record(e)
        assert r.status_code == 303, r.text[:300]
    a.sleep(END)
    return finish(srv, ctl, trig), h


def test_perfect_client_scores_one(tmp_path, gold):
    res, h = perfect_run(tmp_path, gold)
    assert res["performance"]["primary"]["value"] == 1.0
    assert res["performance"]["events_settled"] == len(gold)
    assert set(res["task"]["outcomes"][gold[0].id]) >= {"credit", "evidence", "timely"}
    assert "web_nonbrowser" not in res["flags"]
    rows = ledger_view(Path(h["env_url"] and tmp_path / "perfect" / "server" / "ledger.jsonl"))
    kinds = [r["type"] for r in rows]
    assert kinds.count("notify") == len(gold)
    assert any(r["type"] == "web" and r["host"] == "sec" for r in rows)
    assert any(r["type"] == "web" and r["host"] == "portal" and r["method"] == "POST"
               for r in rows)
    assert res["resources"]["spent_usd"] == 0.0  # fetches and pages are free


def test_premature_client_scores_zero(tmp_path, gold):
    srv, h, ctl, trig = serve(tmp_path, "premature")
    a = Actor(h)
    for e in gold:
        a.fetch_submissions(e.cik)  # evidence exists, but before acceptance
        assert not a.visible(e)
        assert a.record(e).status_code == 303
    a.sleep(END)
    res = finish(srv, ctl, trig)
    assert res["performance"]["primary"]["value"] == 0.0
    assert res["performance"]["status_counts"]["premature"] == len(gold)


def test_daily_0600_poller_pins_preopen_misses(tmp_path, gold):
    """A cron at 06:00 ET each day records whatever is visible: credit iff
    a wake falls in [accepted_at, deadline) — computed independently. The
    roster's 06:22/06:32 ET filings (IBM 10-Q, JPM 8-K) are the misses."""
    def wakes():
        d = START.astimezone(ET).date()
        while True:
            w = datetime(d.year, d.month, d.day, 6, 0, tzinfo=ET).astimezone(UTC)
            if w >= END:
                return
            if w >= START:
                yield w
            d += timedelta(days=1)
    expected = {e.id: any(e.accepted_at <= w < e.deadline for w in wakes())
                for e in gold}
    srv, h, ctl, trig = serve(tmp_path, "cron")
    a = Actor(h)
    done: set[str] = set()
    for w in wakes():
        a.sleep(w)
        for e in gold:
            if e.id in done:
                continue
            if a.visible(e):
                assert a.record(e).status_code == 303
                done.add(e.id)
    a.sleep(END)
    res = finish(srv, ctl, trig)
    got = {k: v["credit"] == 1.0 for k, v in res["task"]["outcomes"].items()}
    assert got == expected
    assert 0 < sum(expected.values()) < len(expected)  # a real pre-open miss rate
    assert res["performance"]["primary"]["value"] == pytest.approx(
        sum(expected.values()) / len(expected), abs=1e-4)


def test_ledger_is_deterministic(tmp_path, gold):
    res1, h1 = perfect_run(tmp_path, gold, "det1")
    res2, h2 = perfect_run(tmp_path, gold, "det2")
    l1 = ledger_view(tmp_path / "det1" / "server" / "ledger.jsonl")
    l2 = ledger_view(tmp_path / "det2" / "server" / "ledger.jsonl")
    assert l1 == l2
    assert res1["performance"] == res2["performance"]


def test_restore_replays_the_sheet(tmp_path, gold):
    res, h = perfect_run(tmp_path, gold, "restore")
    events = [json.loads(l) for l in
              (tmp_path / "restore" / "server" / "ledger.jsonl").read_text().splitlines()]
    cfg = make_cfg(run_id="edgar-restore")
    task = TASK.from_run_config(cfg, REPO_ROOT)
    run_dir = tmp_path / "restore2"
    run_dir.mkdir()
    sim = Sim(cfg, run_dir, run_dir / "ws", task)
    sim.ledger.events.extend(events)  # what checkpoint.restore preloads
    n = task.restore(events, END)
    assert n == len(gold)
    assert task.metrics()["primary"]["value"] == 1.0
    assert task.report()["filings_sheet"] == res["task"]["filings_sheet"]


def test_rejections_and_versions(tmp_path, gold):
    srv, h, ctl, trig = serve(tmp_path, "reject")
    a = Actor(h)
    e = gold[0]
    a.sleep(e.accepted_at + timedelta(minutes=1))
    a.fetch_submissions(e.cik)  # the evidence the credit needs
    bad = a.record(e, accession="nope")
    assert bad.status_code == 200 and "accession must look like" in bad.text
    ok = a.record(e)
    assert ok.status_code == 303
    # re-submitting the same token is a no-op, not a second write
    page = a.http.get(f"{a.portal}/portfolio/{e.ticker}?accession={e.accession}")
    assert e.accession in page.text
    stale = a.record(e, version="0")  # the record is now v1: stale edit refused
    assert stale.status_code == 200 and "changed since you opened it" in stale.text
    a.sleep(END)
    res = finish(srv, ctl, trig)
    rows = ledger_view(tmp_path / "reject" / "server" / "ledger.jsonl")
    assert [r for r in rows if r["type"] == "web" and r.get("rejected")]
    assert res["task"]["outcomes"][e.id]["credit"] == 1.0


def test_nonbrowser_write_is_refused(tmp_path, gold):
    """A curl write to the portal (no Sec-Fetch headers) is refused with
    403 before the app sees it, ledgered `nonbrowser`, flags the run, and
    records nothing: the browser is the only write path."""
    srv, h, ctl, trig = serve(tmp_path, "curl")
    a = Actor(h, headers=None)  # curl-like: no Sec-Fetch headers
    e = gold[0]
    a.sleep(e.accepted_at + timedelta(minutes=1))
    a.fetch_submissions(e.cik)
    r = a.record(e)
    assert r.status_code == 403 and "only from the browser" in r.text
    a.sleep(END)
    res = finish(srv, ctl, trig)
    assert "web_nonbrowser" in res["flags"]
    rows = ledger_view(tmp_path / "curl" / "server" / "ledger.jsonl")
    post = [r for r in rows if r["type"] == "web" and r["method"] == "POST"][0]
    assert post["nonbrowser"] is True and post["status"] == 403
    assert not [r for r in rows if r["type"] == "notify"]
    assert res["task"]["outcomes"][e.id]["credit"] == 0.0


def test_token_scopes(tmp_path):
    srv, h, ctl, trig = serve(tmp_path, "scopes")
    agent = _Http(h["env_url"], h["agent_token"])
    assert "sleep" in {t["name"] for t in agent.get("/tools")["tools"]}
    with pytest.raises(Exception) as ei:
        agent.get("/status")
    assert "403" in str(ei.value)
    assert ctl.get("/status")["hosts"] == h["hosts"]
    assert set(ctl.get("/contract")["hosts"]) == {"sec", "portal"}
    agent.call("sleep", {"until": iso(END)})
    finish(srv, ctl, trig)


def test_api_arm(tmp_path, gold):
    srv, h, ctl, trig = serve(tmp_path, "api", portal_access="api")
    a = Actor(h, headers=None)
    e = gold[0]
    a.sleep(e.accepted_at + timedelta(minutes=1))
    a.fetch_submissions(e.cik)
    body = {"form": e.form, "accepted_at": iso(e.accepted_at), "fy": e.fy,
            "fp": e.fp, "revenue_usd": e.revenue_usd, "eps_diluted": e.eps_diluted}
    r = a.http.put(f"{a.portal}/portfolio/api/rows/{e.ticker}/{e.accession}", json=body)
    assert r.status_code == 200 and r.json()["status"] == "saved"
    rows = a.http.get(f"{a.portal}/portfolio/api/rows").json()["rows"]
    assert rows[0]["accession"] == e.accession
    a.sleep(END)
    res = finish(srv, ctl, trig)
    assert res["task"]["outcomes"][e.id]["credit"] == 1.0
