"""The portfolio portal (`portal`, writable): the fund's filings sheet
. Server-rendered pages, no JavaScript; the only write path in
the browser arm is the ticker page's form. Every accepted write is one
`notify` ledger row (the journal record's payload), so restore replays
it; rejections re-render the form with the message, free.

    GET  /portfolio/                     holdings (as of now) + market status
    GET  /portfolio/audit                the sheet's audit log
    GET  /portfolio/<TICKER>             holding + its filing records + the form
    POST /portfolio/<TICKER>             save a filing record (form)
    api arm only:
    PUT  /portfolio/api/rows/<TICKER>/<ACCESSION>   JSON write, same journal
    GET  /portfolio/api/rows             JSON dump of the sheet

Handlers run under sim.lock (harness/web.py) and never move the clock.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from harness.task import NotificationError
from harness.timeutil import iso, parse_iso
from harness.web import html, templates, web_log
from tasks.edgar_portfolio.edgar import ACCN_RE, market_status

if TYPE_CHECKING:
    from tasks.edgar_portfolio.task import EdgarPortfolioTask

TEMPLATES = Path(__file__).resolve().parent / "templates"
FORMS = ("8-K", "10-Q", "10-K")
PERIODS = ("Q1", "Q2", "Q3", "Q4", "FY")
TICKER_RE = re.compile(r"^[A-Z.\-]{1,8}$")


def parse_fields(raw: dict) -> dict:
    """Validate one filing record from form/JSON input. Raises
    NotificationError with the message the page shows."""
    def s(k):
        v = raw.get(k)
        return v.strip() if isinstance(v, str) else v

    accession = s("accession") or ""
    if not ACCN_RE.match(accession):
        raise NotificationError("accession must look like 0000320193-26-000013")
    form = s("form") or ""
    if form not in FORMS:
        raise NotificationError(f"form must be one of {', '.join(FORMS)}")
    acc_raw = s("accepted_at") or ""
    try:
        accepted_at = parse_iso(acc_raw)
    except (ValueError, TypeError):
        raise NotificationError("accepted_at must be an ISO-8601 UTC datetime, "
                                "e.g. 2026-04-30T20:30:41Z")
    fy = s("fy")
    if fy in (None, ""):
        fy_val = None
    else:
        try:
            fy_val = int(fy)
        except (TypeError, ValueError):
            raise NotificationError("fiscal year must be a 4-digit integer or blank")
        if not 1990 <= fy_val <= 2100:
            raise NotificationError("fiscal year out of range")
    fp = s("fp") or None
    if fp not in (None,) + PERIODS:
        raise NotificationError(f"fiscal period must be one of {', '.join(PERIODS)} or blank")

    def num(k, label):
        v = s(k)
        if v in (None, ""):
            return None
        try:
            return float(str(v).replace(",", ""))
        except ValueError:
            raise NotificationError(f"{label} must be a number or blank")

    return {"accession": accession, "form": form, "accepted_at": iso(accepted_at),
            "fy": fy_val, "fp": fp, "revenue_usd": num("revenue_usd", "revenue"),
            "eps_diluted": num("eps_diluted", "diluted EPS"),
            "note": (s("note") or "")[:200]}


def make_portal_app(sim, task: "EdgarPortfolioTask") -> Starlette:
    env = templates(TEMPLATES)
    world = task.world
    api_arm = task.tcfg.portal_access == "api"

    def ctx(request, **extra) -> dict:
        now = sim.clock.now
        ms = market_status(now)
        return {"now": now, "market": ms, "api_arm": api_arm,
                "sim_end": sim.cfg.sim_end, **extra}

    async def holdings(request):
        web_log(request, page="holdings")
        rows = world.holdings_view()
        return html(env, "holdings.html", **ctx(request, rows=rows))

    async def audit(request):
        web_log(request, page="audit")
        recs = [r.to_dict() for r in world.journal.records][-200:][::-1]
        return html(env, "audit.html", **ctx(request, records=recs))

    def _ticker(request) -> str | None:
        t = request.path_params["ticker"].upper()
        return t if TICKER_RE.match(t) and world.known_ticker(t) else None

    def render_ticker(request, ticker: str, *, values: dict | None = None,
                      error: str | None = None, saved: str | None = None,
                      status: int = 200):
        holding = world.holding_view(ticker)
        records = world.filing_records(ticker)
        token = world.journal.mint_token(sim.clock.now)
        values = values or {}
        editing = None
        if values.get("accession"):
            editing = world.journal.get("filings", f"{ticker}:{values['accession']}")
        version = editing["version"] if editing else 0
        return html(env, "ticker.html", status=status,
                    **ctx(request, ticker=ticker, holding=holding, records=records,
                          form_token=token, version=version, values=values,
                          error=error, saved=saved, forms=FORMS, periods=PERIODS))

    async def ticker_page(request):
        t = _ticker(request)
        if t is None:
            web_log(request, page="ticker", unknown=True)
            return html(env, "notfound.html", status=404, **ctx(request))
        web_log(request, page="ticker", ticker=t)
        prefill = {}
        acc = request.query_params.get("accession")
        if acc:
            st = world.journal.get("filings", f"{t}:{acc}")
            if st:
                prefill = dict(st["fields"])
        saved = request.query_params.get("saved")
        return render_ticker(request, t, values=prefill, saved=saved)

    async def ticker_save(request):
        t = _ticker(request)
        if t is None:
            web_log(request, action="save", unknown=True)
            return html(env, "notfound.html", status=404, **ctx(request))
        form = await request.form()
        raw = {k: form.get(k) for k in form.keys()}
        web_log(request, action="save", ticker=t)
        try:
            fields = parse_fields(raw)
            version = raw.get("version")
            expect = int(version) if version not in (None, "") else None
            rec = task.write(sim.clock.now, actor="agent", app="filings",
                             kind="update", entity=f"{t}:{fields['accession']}",
                             fields=fields, form_token=raw.get("form_token"),
                             expect_version=expect, session=None)
        except NotificationError as e:
            web_log(request, rejected=str(e)[:80])
            return render_ticker(request, t, values=raw, error=str(e))
        web_log(request, entity=rec.entity, duplicate=rec.duplicate)
        return RedirectResponse(
            f"/portfolio/{t}?saved={fields['accession']}", status_code=303)

    # -- api arm -------------------------------------------------------------------

    async def api_rows(request):
        web_log(request, page="api_rows")
        return JSONResponse({"now": iso(sim.clock.now),
                             "rows": world.filing_rows_json()})

    async def api_put(request):
        t = request.path_params["ticker"].upper()
        acc = request.path_params["accession"]
        web_log(request, action="api_put", ticker=t)
        if not world.known_ticker(t):
            return JSONResponse({"error": f"unknown ticker {t}"}, status_code=404)
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        try:
            fields = parse_fields({**body, "accession": acc})
            rec = task.write(sim.clock.now, actor="agent", app="filings",
                             kind="update", entity=f"{t}:{acc}", fields=fields,
                             form_token=None, expect_version=body.get("version"),
                             session=None)
        except NotificationError as e:
            web_log(request, rejected=str(e)[:80])
            return JSONResponse({"error": str(e)}, status_code=400)
        web_log(request, entity=rec.entity)
        return JSONResponse({"status": "saved", "entity": rec.entity,
                             "version": rec.version, "at": iso(sim.clock.now)})

    routes = [
        Route("/portfolio/", holdings),
        Route("/portfolio", holdings),
        Route("/portfolio/audit", audit),
    ]
    if api_arm:
        routes += [Route("/portfolio/api/rows", api_rows),
                   Route("/portfolio/api/rows/{ticker}/{accession}", api_put,
                         methods=["PUT"])]
    routes += [
        Route("/portfolio/{ticker}", ticker_page),
        Route("/portfolio/{ticker}", ticker_save, methods=["POST"]),
    ]

    async def root(request):
        return RedirectResponse("/portfolio/", status_code=302)

    async def fallback(request):
        web_log(request, unknown=True)
        return html(env, "notfound.html", status=404, **ctx(request))

    routes = [Route("/", root)] + routes + [Route("/{path:path}", fallback)]
    return Starlette(routes=routes)
