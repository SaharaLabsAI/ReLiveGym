"""The ops inbox (`mail`, read-only): standing orders, one-time login
codes, margin-call and policy notices, as of the sim instant. No writes —
a watcher program may poll it.

    GET /mail/                 inbox, newest first
    GET /mail/<id>             one message
    GET /mail/api/messages     JSON [{id, at, subject, kind}] (?since=iso)
    GET /mail/api/messages/<id>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from harness.timeutil import iso, parse_iso
from harness.web import html, templates, web_log

if TYPE_CHECKING:
    from tasks.broker_ops.task import BrokerOpsTask

TEMPLATES = Path(__file__).resolve().parent / "templates"


def make_mail_app(sim, task: "BrokerOpsTask") -> Starlette:
    env = templates(TEMPLATES)
    world = task.world

    def ctx(**extra):
        return {"now": sim.clock.now, "user": None, "host": "mail", **extra}

    async def inbox(request):
        web_log(request, page="inbox")
        return html(env, "inbox.html", **ctx(messages=world.mail_view()))

    async def message(request):
        mid = request.path_params["mid"]
        st = world.journal.get("mail", mid)
        web_log(request, page="message", message=mid)
        if st is None:
            return html(env, "notfound.html", status=404, **ctx())
        return html(env, "message.html", **ctx(m={"id": mid, **st["fields"]}))

    async def api_list(request):
        since = request.query_params.get("since")
        rows = world.mail_view()
        if since:
            try:
                s = iso(parse_iso(since))
                rows = [m for m in rows if m["at"] > s]
            except ValueError:
                return JSONResponse({"error": "since must be ISO-8601"}, status_code=400)
        web_log(request, page="api_messages")
        return JSONResponse({"now": iso(sim.clock.now),
                             "messages": [{k: m[k] for k in ("id", "at", "subject", "kind")}
                                          for m in rows]})

    async def api_one(request):
        mid = request.path_params["mid"]
        st = world.journal.get("mail", mid)
        web_log(request, page="api_message", message=mid)
        if st is None:
            return JSONResponse({"error": "no such message"}, status_code=404)
        return JSONResponse({"id": mid, **st["fields"]})

    async def root(request):
        return RedirectResponse("/mail/", status_code=302)

    async def fallback(request):
        web_log(request, unknown=True)
        return html(env, "notfound.html", status=404, **ctx())

    return Starlette(routes=[
        Route("/", root), Route("/mail", inbox), Route("/mail/", inbox),
        Route("/mail/api/messages", api_list),
        Route("/mail/api/messages/{mid}", api_one),
        Route("/mail/{mid}", message), Route("/{path:path}", fallback)])
