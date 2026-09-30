"""The broker portal for broker_ops (`broker`, writable, login-gated).
Every mutation is a form POST that becomes one agent action (world.act)
and one `notify` ledger row; a POST under an expired session renders the
login page with the message and applies nothing.

    GET/POST /login            username + password  → code mailed
    GET/POST /login/code       one-time code        → signed in
    POST     /logout
    GET      /positions        account, marks, margin ratio
    GET      /orders           the PM's book (read-only)
    GET      /notices          margin calls;  POST /notices/<id>/respond
    GET      /risk             protective stops;  POST /risk (register)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import RedirectResponse
from starlette.routing import Route

from harness.task import NotificationError
from harness.web import html, templates, web_log

if TYPE_CHECKING:
    from tasks.broker_ops.task import BrokerOpsTask

TEMPLATES = Path(__file__).resolve().parent / "templates"


def make_broker_app(sim, task: "BrokerOpsTask") -> Starlette:
    env = templates(TEMPLATES)
    world = task.world
    tcfg = task.tcfg

    def session(request):
        now = sim.clock.now
        sid = request.cookies.get("sid")
        s = world.sessions.get(sid, now)
        if s is None:
            s = world.sessions.new_session(now)
            request.state.new_sid = s.id
        request.state.session = s
        web_log(request, session=s.id)
        return s

    def cookie(request, resp):
        sid = getattr(request.state, "new_sid", None)
        if sid:
            resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp

    def ctx(request, **extra):
        s = request.state.session
        return {"now": sim.clock.now, "host": "broker",
                "user": s.user if s.authenticated else None, "tcfg": tcfg,
                "error": None, "saved": False, "not_applied": False, **extra}

    def page(request, name, status=200, **extra):
        return cookie(request, html(env, name, status=status, **ctx(request, **extra)))

    def login_page(request, status=200, **extra):
        return page(request, "login.html", status=status,
                    form_token=world.journal.mint_token(sim.clock.now), **extra)

    def guard(request):
        s = request.state.session
        if not s.authenticated:
            why = ("your session has expired — please sign in again"
                   if s.ended_at is not None else None)
            if request.method == "GET":
                return cookie(request, RedirectResponse(
                    "/login" + ("?expired=1" if why else ""), status_code=302))
            web_log(request, rejected="not_authenticated")
            return login_page(request, error=why or "please sign in first",
                              not_applied=True)
        return None

    def act(request, kind, fields, token):
        s = request.state.session
        try:
            world.act(sim.clock.now, kind, s.id, fields, form_token=token)
        except NotificationError as e:
            web_log(request, rejected=str(e)[:80])
            return str(e)
        web_log(request, action=kind)
        return None

    async def form(request) -> dict:
        f = await request.form()
        return {k: f.get(k) for k in f.keys()}

    # -- login flow --------------------------------------------------------------------

    async def login_get(request):
        session(request)
        web_log(request, page="login")
        expired = request.query_params.get("expired")
        return login_page(request, error=("your session has expired — please sign "
                                          "in again" if expired else None))

    async def login_post(request):
        session(request)
        raw = await form(request)
        err = act(request, "login_password",
                  {"username": raw.get("username"), "password": raw.get("password")},
                  raw.get("form_token"))
        if err:
            return login_page(request, error=err)
        return cookie(request, RedirectResponse("/login/code", status_code=303))

    async def code_get(request):
        s = session(request)
        web_log(request, page="code")
        if not s.pending_user:
            return login_page(request, error="start by entering your password")
        return page(request, "code.html", form_token=world.journal.mint_token(sim.clock.now))

    async def code_post(request):
        session(request)
        raw = await form(request)
        err = act(request, "login_code", {"code": raw.get("code")}, raw.get("form_token"))
        if err:
            return login_page(request, error=err)
        return cookie(request, RedirectResponse("/positions", status_code=303))

    async def logout(request):
        session(request)
        act(request, "logout", {}, None)
        return cookie(request, RedirectResponse("/login", status_code=303))

    # -- protected pages --------------------------------------------------------------

    async def root(request):
        s = session(request)
        return cookie(request, RedirectResponse(
            "/positions" if s.authenticated else "/login", status_code=302))

    async def positions(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        web_log(request, page="positions")
        return page(request, "positions.html", acct=world.account(sim.clock.now),
                    open_orders=world.open_orders(), pending_stops=world.fills_without_stop(),
                    open_notices=[n for n in world.notices_view() if n["status"] == "open"])

    async def orders_get(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        web_log(request, page="orders")
        return page(request, "orders.html", orders=world.orders_view(),
                    stops={s["id"]: s for s in world.stops_view()})

    async def notices(request, error=None, saved=False):
        web_log(request, page="notices")
        return page(request, "notices.html", notices=world.notices_view(),
                    form_token=world.journal.mint_token(sim.clock.now),
                    error=error, saved=saved)

    async def notices_get(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        return await notices(request, saved=bool(request.query_params.get("saved")))

    async def notice_respond(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        raw = await form(request)
        err = act(request, "notice_respond",
                  {"notice_id": request.path_params["nid"], "amount_usd": raw.get("amount_usd")},
                  raw.get("form_token"))
        if err:
            return await notices(request, error=err)
        return cookie(request, RedirectResponse("/notices?saved=1", status_code=303))

    async def risk(request, error=None, saved=False, prefill=""):
        web_log(request, page="risk")
        return page(request, "risk.html", pending=world.fills_without_stop(),
                    stops=world.stops_view(), marks=world.marks(sim.clock.now),
                    form_token=world.journal.mint_token(sim.clock.now),
                    error=error, saved=saved, prefill=prefill)

    async def risk_get(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        return await risk(request, saved=bool(request.query_params.get("saved")),
                          prefill=request.query_params.get("order_id", ""))

    async def risk_post(request):
        session(request)
        if (r := guard(request)) is not None:
            return r
        raw = await form(request)
        fields = {k: raw.get(k) for k in ("order_id", "side", "qty", "trigger_price")}
        err = act(request, "stop_register", fields, raw.get("form_token"))
        if err:
            return await risk(request, error=err, prefill=raw.get("order_id") or "")
        return cookie(request, RedirectResponse("/risk?saved=1", status_code=303))

    async def fallback(request):
        session(request)
        web_log(request, unknown=True)
        return page(request, "notfound.html", status=404)

    return Starlette(routes=[
        Route("/", root),
        Route("/login", login_get), Route("/login", login_post, methods=["POST"]),
        Route("/login/code", code_get), Route("/login/code", code_post, methods=["POST"]),
        Route("/logout", logout, methods=["POST"]),
        Route("/positions", positions),
        Route("/orders", orders_get),
        Route("/notices", notices_get),
        Route("/notices/{nid}/respond", notice_respond, methods=["POST"]),
        Route("/risk", risk_get), Route("/risk", risk_post, methods=["POST"]),
        Route("/{path:path}", fallback),
    ])
