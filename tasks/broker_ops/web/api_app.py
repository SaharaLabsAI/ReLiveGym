"""The broker's read-only account API (`api`; replay data a
watcher program may poll). JSON as of the sim instant; no auth, no writes.

    GET /api/events            [{id, at, kind, ...}] newest first (?since=<iso>)
                               kinds: fill | margin_call | policy | treasury
    GET /api/events/<id>
    GET /api/account           positions, marks, equity, exposure, margin ratio, maintenance
    GET /api/orders            the PM's book (open / filled / expired)
    GET /api/notices           margin calls (open / responded / expired)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from harness.timeutil import iso, parse_iso
from harness.web import web_log

if TYPE_CHECKING:
    from tasks.broker_ops.task import BrokerOpsTask


def make_api_app(sim, task: "BrokerOpsTask") -> Starlette:
    world = task.world

    def now() -> str:
        return iso(sim.clock.now)

    async def events(request):
        since = request.query_params.get("since")
        rows = world.events_view()
        if since:
            try:
                s = iso(parse_iso(since))
            except ValueError:
                return JSONResponse({"error": "since must be ISO-8601"}, status_code=400)
            rows = [e for e in rows if e["at"] > s]
        web_log(request, page="api_events")
        return JSONResponse({"now": now(), "events": rows})

    async def event(request):
        eid = request.path_params["eid"]
        st = world.journal.get("events", eid)
        web_log(request, page="api_event", event=eid)
        if st is None:
            return JSONResponse({"error": "no such event"}, status_code=404)
        return JSONResponse({"id": eid, **st["fields"]})

    async def account(request):
        web_log(request, page="api_account")
        return JSONResponse({"now": now(), **world.account(sim.clock.now)})

    async def orders(request):
        web_log(request, page="api_orders")
        return JSONResponse({"now": now(), "orders": world.orders_view()})

    async def notices(request):
        web_log(request, page="api_notices")
        return JSONResponse({"now": now(), "notices": world.notices_view()})

    async def fallback(request):
        web_log(request, unknown=True)
        return JSONResponse({"error": "not found", "endpoints": [
            "/api/events", "/api/events/<id>", "/api/account", "/api/orders",
            "/api/notices"]}, status_code=404)

    return Starlette(routes=[
        Route("/api/events", events), Route("/api/events/{eid}", event),
        Route("/api/account", account), Route("/api/orders", orders),
        Route("/api/notices", notices), Route("/{path:path}", fallback)])
