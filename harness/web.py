"""Web hosts: the browser-facing side of a web task.

A task declares named hosts (`Task.web_apps(sim)` -> [WebHostSpec]);
`start_hosts` binds one uvicorn listener per host on an ephemeral
loopback port, in the same process and event loop as the env API, and
records the URLs in `sim.hosts` (run.json, GET /status, GET /contract,
and the `${<name>_url}` INSTRUCTION placeholders all read from there).

Every request runs under `sim.lock` at the frozen sim instant: the
middleware takes the lock, counts the request as agent activity for the
real-time watchdog, stamps `X-Sim-Time` on the response and appends one
free `web` ledger row {host, method, path, query, status, + whatever the
app put in request.state.web_log}. On a writable host a mutating
request without the headers browsers set automatically (Sec-Fetch-*)
is marked `nonbrowser`, flags the run once (`web_nonbrowser`) and — on a
`browser_only` host, the default for writable hosts — is refused with
403 before the app sees it: the browser is the only write path (an actor whose browser breaks would
otherwise curl every form). A task
arm that is meant to be scripted (edgar's `portal_access: api`) opts out
with browser_only=False and keeps the audit row only.

App handlers therefore MUST NOT take sim.lock themselves (they already
hold it) and must never move the clock: pages are pure functions of
(replay as of now, journal fold, now).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.timeutil import iso

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
NONBROWSER_FLAG = "web_nonbrowser"


@dataclass(frozen=True)
class WebHostSpec:
    """One host: `name` is the placeholder `${<name>_url}` and the ledger
    `host`; `writable` hosts carry sessions/forms and are fenced off
    from authored programs (harness/mcp.py); read-only hosts are replay
    data a watcher program may poll."""

    name: str
    app: Any            # an ASGI app
    writable: bool = False
    browser_only: bool = True  # writable hosts: refuse non-browser writes (403)


class HostMiddleware:
    """Pure ASGI wrapper: lock + activity + ledger + X-Sim-Time."""

    def __init__(self, app, sim, spec: WebHostSpec):
        self.app = app
        self.sim = sim
        self.spec = spec

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sim = self.sim
        async with sim.lock:
            sim.touch()
            now = sim.clock.now
            state = scope.setdefault("state", {})
            state["web_log"] = {}
            status: int | None = None

            async def send_wrapped(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                    headers = list(message.get("headers") or [])
                    headers.append((b"x-sim-time", iso(now).encode()))
                    message = {**message, "headers": headers}
                await send(message)

            method = scope.get("method", "")
            nonbrowser = program = False
            if self.spec.writable and method in MUTATING:
                names = {k.lower() for k, _ in scope.get("headers") or []}
                nonbrowser = b"sec-fetch-mode" not in names
                program = bool(getattr(sim, "program_active", False))
            try:
                if program:
                    await _refuse(send_wrapped, PROGRAM_REFUSAL)
                elif nonbrowser and self.spec.browser_only:
                    await _refuse(send_wrapped)
                else:
                    await self.app(scope, receive, send_wrapped)
            finally:
                row = {"host": self.spec.name, "method": method,
                       "path": scope.get("path", ""),
                       "status": status if status is not None else 500}
                query = (scope.get("query_string") or b"").decode(errors="replace")
                if query:
                    row["query"] = query
                if nonbrowser:
                    row["nonbrowser"] = True
                    if NONBROWSER_FLAG not in sim.flags:
                        sim.flags.append(NONBROWSER_FLAG)
                if program:
                    row["program_write"] = True
                    if PROGRAM_FLAG not in sim.flags:
                        sim.flags.append(PROGRAM_FLAG)
                row.update(state.get("web_log") or {})
                sim.ledger.append("web", now, **row)


# agent-visible (design Appendix A): the one sentence a curl write gets back
NONBROWSER_REFUSAL = (b"403 Forbidden: this site accepts writes only from the "
                      b"browser. Open the page and submit its form.\n")
# ... and the one a write during a program run gets back (a
# watcher program may read, never write; the episode bridge marks runs)
PROGRAM_REFUSAL = (b"403 Forbidden: this site accepts no writes while a "
                   b"program is running.\n")
PROGRAM_FLAG = "web_program_write"


async def _refuse(send, body: bytes = NONBROWSER_REFUSAL) -> None:
    await send({"type": "http.response.start", "status": 403,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def web_log(request, **fields) -> None:
    """From an app handler: add fields (session, entity, action, ...) to
    this request's `web` ledger row."""
    request.state.web_log.update(fields)


async def start_hosts(sim, specs: list[WebHostSpec],
                      host: str = "127.0.0.1") -> list[tuple[Any, asyncio.Task]]:
    """Bind every host; returns [(uvicorn server, its task)] for shutdown
    (set server.should_exit and await the task)."""
    import uvicorn

    out = []
    for spec in specs:
        if spec.name in sim.hosts:
            raise ValueError(f"duplicate web host name {spec.name!r}")
        server = uvicorn.Server(uvicorn.Config(
            HostMiddleware(spec.app, sim, spec), host=host, port=0,
            log_level="warning", access_log=False))
        task = asyncio.create_task(server.serve())
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError(f"web host {spec.name!r} exited before starting")
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        sim.hosts[spec.name] = f"http://{host}:{port}"
        sim.host_specs[spec.name] = spec
        out.append((server, task))
    return out


async def stop_hosts(started: list[tuple[Any, asyncio.Task]]) -> None:
    for server, task in started:
        server.should_exit = True
    for _, task in started:
        try:
            await task
        except Exception:
            pass


def readonly_hosts(sim) -> dict[str, str]:
    """The hosts an authored program may reach."""
    return {n: u for n, u in sim.hosts.items()
            if n in sim.host_specs and not sim.host_specs[n].writable}


# -- templates --------------------------------------------------------------------------


def templates(directory: Path):
    """A Jinja2 environment for a task's server-rendered pages: autoescape
    on, an `iso` filter, and `et`/`fmt` helpers for display clocks."""
    from datetime import timezone as _tz
    from zoneinfo import ZoneInfo

    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(directory)),
                             autoescape=True, undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True)
    et = ZoneInfo("America/New_York")

    def fmt(dt, tz: str = "UTC", pattern: str = "%Y-%m-%d %H:%M"):
        if dt is None:
            return "—"
        zone = et if tz == "ET" else _tz.utc if tz == "UTC" else ZoneInfo(tz)
        label = "ET" if tz == "ET" else tz
        return dt.astimezone(zone).strftime(pattern) + f" {label}"

    env.filters["iso"] = lambda dt: iso(dt) if dt is not None else "—"
    env.filters["fmt"] = fmt
    return env


def html(env, name: str, status: int = 200, **ctx):
    from starlette.responses import HTMLResponse

    return HTMLResponse(env.get_template(name).render(**ctx), status_code=status)
