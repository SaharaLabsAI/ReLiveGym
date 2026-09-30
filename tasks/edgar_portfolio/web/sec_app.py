"""The replayed EDGAR host (`sec`, read-only): data.sec.gov's JSON
endpoints as of the sim instant. No tool, no session — the
agent fetches these URLs with curl or the browser exactly as it would
the real site; every request is a free `web` ledger row.

    GET /                                                  plain-text index
    GET /files/company_tickers.json                        static
    GET /submissions/CIK##########.json                    as-of
    GET /api/xbrl/companyfacts/CIK##########.json          as-of
    GET /api/xbrl/companyconcept/CIK##########/<tax>/<tag>.json   as-of
    GET /Archives/...                                      404 (v1 serves no documents)

Handlers run under sim.lock (harness/web.py middleware) and read the
clock; they never move it.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from harness.limits import RateLimited, RateLimiter
from harness.web import web_log

if TYPE_CHECKING:
    from tasks.edgar_portfolio.task import EdgarPortfolioTask

CIK_RE = re.compile(r"^CIK(\d{10})$")

INDEX = """EDGAR mirror — the same paths as data.sec.gov, served as of the current time.

  /files/company_tickers.json
  /submissions/CIK##########.json
  /api/xbrl/companyfacts/CIK##########.json
  /api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json

CIK is 10 digits, zero-padded. Filing documents (/Archives/...) are not
served by this mirror.
"""


def _cik(param: str) -> int | None:
    m = CIK_RE.match(param or "")
    return int(m.group(1)) if m else None


def _json(payload, status: int = 200) -> Response:
    return Response(json.dumps(payload, separators=(",", ":")),
                    status_code=status, media_type="application/json")


def _not_found(what: str) -> Response:
    return _json({"error": f"{what} not found"}, 404)


def make_sec_app(sim, task: "EdgarPortfolioTask") -> Starlette:
    store = task.store
    limiter: RateLimiter | None = None
    spec = task.tcfg.sec_rate_limit
    if spec and spec.get("window") not in (None, "none"):
        limiter = sim.limiters.setdefault("sec", RateLimiter("sec mirror", spec))

    def gate(request) -> Response | None:
        if limiter is None:
            return None
        try:
            limiter.consume(sim.clock.now)
        except RateLimited as e:
            web_log(request, limited=True)
            headers = {}
            if e.send_header and e.retry_after_s is not None:
                headers["Retry-After"] = str(max(1, int(e.retry_after_s + 0.999)))
            return _json({"error": "rate limit exceeded", "detail": str(e)}, 429,) \
                if not headers else Response(
                    json.dumps({"error": "rate limit exceeded"}), 429,
                    headers=headers, media_type="application/json")
        return None

    async def index(request):
        return PlainTextResponse(INDEX)

    async def tickers(request):
        if (r := gate(request)) is not None:
            return r
        web_log(request, endpoint="company_tickers")
        return Response(store.tickers_bytes, media_type="application/json")

    async def submissions(request):
        if (r := gate(request)) is not None:
            return r
        cik = _cik(request.path_params["cik"])
        web_log(request, endpoint="submissions", cik=cik)
        if cik is None or not store.has_cik(cik):
            return _not_found("submissions for that CIK")
        return _json(store.submissions_asof(cik, sim.clock.now))

    async def companyfacts(request):
        if (r := gate(request)) is not None:
            return r
        cik = _cik(request.path_params["cik"])
        web_log(request, endpoint="companyfacts", cik=cik)
        if cik is None or not store.has_cik(cik):
            return _not_found("companyfacts for that CIK")
        return _json(store.companyfacts_asof(cik, sim.clock.now))

    async def companyconcept(request):
        if (r := gate(request)) is not None:
            return r
        cik = _cik(request.path_params["cik"])
        taxonomy = request.path_params["taxonomy"]
        tag = request.path_params["tag"]
        web_log(request, endpoint="companyconcept", cik=cik, tag=tag)
        if cik is None or not store.has_cik(cik):
            return _not_found("companyconcept for that CIK")
        out = store.companyconcept_asof(cik, taxonomy, tag, sim.clock.now)
        if out is None:
            return _not_found("that concept")
        return _json(out)

    async def archives(request):
        web_log(request, endpoint="archives")
        return _not_found("filing documents are not served by this mirror; "
                          "the XBRL endpoints carry the reported numbers")

    async def fallback(request):
        return _not_found("that path")

    return Starlette(routes=[
        Route("/", index),
        Route("/files/company_tickers.json", tickers),
        Route("/submissions/{cik}.json", submissions),
        Route("/api/xbrl/companyfacts/{cik}.json", companyfacts),
        Route("/api/xbrl/companyconcept/{cik}/{taxonomy}/{tag}.json", companyconcept),
        Route("/Archives/{path:path}", archives),
        Route("/{path:path}", fallback),
    ])
