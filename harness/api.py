"""The environment contract: HTTP API.

| Endpoint          | Method | Meaning                                          |
|-------------------|--------|--------------------------------------------------|
| /tools            | GET    | manifest of the tools this run provisions        |
| /call/{name}      | POST   | invoke one tool; body = its args as a JSON object|
| /llm/*            | POST   | metered LLM proxy (real provider token rates)    |
| /contract         | GET    | the published contract (tools, llm terms,        |
|                   |        | INSTRUCTION.md, cell_config.py, envkit stub)     |
| /trigger/next     | GET    | lifecycle: advance to the next due trigger       |
| /trigger/{id}/exit| POST   | lifecycle: the invocation ended {code, killed,   |
|                   |        | output, head_sha}                                |
| /activity         | GET    | lifecycle: watchdog facts for the runner         |
| /status           | GET    | sim time, done, flags, spend                     |

The data plane (/tools, /call, /llm) is what the actor's program uses; the
lifecycle routes are what its runner (scaffolds/runtime/actor.py) uses —
the server owns sim time, the runner owns the process
(harness/supervisor.py). Only data-plane requests count as agent activity
for the watchdog / crash-streak rule.

Everything the environment offers an agent program is a tool declared on an
EnvApp (harness/apps.py + tasks/<task>/env/apps.py); GET /tools is the
single source of what exists in this run, and POST /call/{name} the single
dispatch point. Malformed or rejected calls are HTTP 4xx — the free
error-handling path. Two enforced constraints surface here:
HTTP 402 when the run's budget_usd is
spent (cost-bearing calls only) and HTTP 429 when a task rate limit is
exceeded (Retry-After only where the real API documents one). Which tools
exist derives from the run config (provisioning), so capability
differences between cells are data, never branches in agent code.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

from harness.apps import harness_apps
from harness.env_tools import PaymentRequired, ToolError, build_registry, manifest
from harness.limits import RateLimited
from harness.runtime import Sim
from harness.supervisor import SchedulerError
from harness.task import NotificationError


def build_tool_registry(sim: Sim):
    """All tools provisioned for this run (harness + task), name -> tool."""
    return build_registry(harness_apps(sim) + sim.task.env_apps(sim))


def tools_manifest(sim: Sim) -> list[dict]:
    """The GET /tools payload — also snapshotted into workspace_manifest."""
    return manifest(build_tool_registry(sim))


# Under the cron arms the clock and the crontab are the driver's: an
# agent-token request for these is refused (403). Constructor-built programs hold the
# controller token and only the tools in their registry; an episode actor
# has a shell, reaches this port, and its token sits in its workspace —
# one `curl .../call/sleep` would move the clock mid-firing.
CRON_ARM_CONTROL_TOOLS = frozenset({
    "sleep", "set_crontab", "run_at", "get_crontab", "set_party"})


def make_app(sim: Sim) -> FastAPI:
    async def auth(request: Request) -> None:
        # two scopes: the controller token opens every route,
        # the agent (data-plane) token only /tools, /call and /llm
        header = request.headers.get("authorization", "")
        if header == f"Bearer {sim.token}":
            request.state.control = True
        elif header == f"Bearer {sim.agent_token}":
            request.state.control = False
        else:
            raise HTTPException(401, "missing or invalid run token")

    async def control_only(request: Request) -> None:
        if not getattr(request.state, "control", False):
            raise HTTPException(403, "this route needs the run's controller token")

    app = FastAPI(title="experiment-env", dependencies=[Depends(auth)])
    registry = build_tool_registry(sim)

    @app.get("/tools")
    async def tools() -> dict:
        sim.touch()
        return {"tools": manifest(registry)}

    @app.post("/call/{name}")
    async def call(name: str, request: Request):
        sim.touch()
        entry = registry.get(name)
        if entry is None:
            raise HTTPException(404, f"no such tool: {name!r}")
        if (name in CRON_ARM_CONTROL_TOOLS and sim.cfg.cell.tm in ("C", "D")
                and not request.state.control):
            raise HTTPException(403, f"{name} is not available to you")
        _, handler = entry
        body = await request.body()
        if not body:
            args: dict = {}
        else:
            try:
                args = await request.json()
                assert isinstance(args, dict)
            except Exception:
                raise HTTPException(
                    400, "request body must be a JSON object of tool args")
        try:
            return await handler(args)
        except PaymentRequired as e:
            raise HTTPException(402, str(e))
        except RateLimited as e:
            headers = {}
            if e.send_header and e.retry_after_s is not None:
                headers["Retry-After"] = str(max(1, int(e.retry_after_s + 0.999)))
            raise HTTPException(429, str(e), headers=headers or None)
        except (ToolError, NotificationError) as e:
            raise HTTPException(400, str(e))

    @app.post("/llm/{path:path}")
    async def llm(path: str, request: Request):
        from harness.llm_proxy import handle_llm

        sim.touch()
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "request body must be JSON")
        return await handle_llm(sim, path, body)

    # -- the published contract -------------------------------------------------------

    @app.get("/contract", dependencies=[Depends(control_only)])
    async def contract() -> dict:
        from harness.contract import contract_payload

        return contract_payload(sim)

    # -- lifecycle (harness/supervisor.py) -------------------------------------------

    def scheduler():
        if sim.scheduler is None:
            raise HTTPException(404, "this sim has no lifecycle scheduler")
        return sim.scheduler

    @app.get("/trigger/next", dependencies=[Depends(control_only)])
    async def trigger_next(code_sha: str | None = None,
                           head_sha: str | None = None) -> dict:
        try:
            return await scheduler().next(code_sha, head_sha)
        except SchedulerError as e:
            raise HTTPException(409, str(e))

    @app.post("/trigger/{trigger_id}/exit", dependencies=[Depends(control_only)])
    async def trigger_exit(trigger_id: str, request: Request) -> dict:
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:
            raise HTTPException(400, "request body must be a JSON object")
        try:
            return await scheduler().report_exit(
                trigger_id, int(body.get("code", 0)),
                bool(body.get("killed", False)),
                str(body.get("output", "")), body.get("head_sha"))
        except SchedulerError as e:
            raise HTTPException(409, str(e))

    @app.post("/program")  # agent-token: the episode bridge marks program runs
    async def program(request: Request):
        """A run_program starts/ends (harness/mcp.py): while active,
        writable web hosts refuse mutating requests (harness/web.py) —
        an authored program may watch, never write."""
        sim.touch()
        try:
            body = await request.json()
            active = bool(body["active"])
        except Exception:
            raise HTTPException(400, 'body must be {"active": bool}')
        async with sim.lock:
            sim.program_active = active
        return {"active": active}

    @app.get("/activity")  # watchdog facts only: agent-token accessible
    async def activity() -> dict:
        return scheduler().activity()

    @app.get("/status", dependencies=[Depends(control_only)])
    async def status() -> dict:
        return scheduler().status()

    return app
