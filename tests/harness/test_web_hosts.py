"""harness/web.py: one listener per declared host in the served sim, URLs
in run.json / /status / /contract, requests under the lock at the frozen
instant, the free `web` ledger row, X-Sim-Time, and the non-browser
write flag on writable hosts."""

from __future__ import annotations

import json

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from harness.mcp import _Http
from harness.web import WebHostSpec, web_log
from tasks.weather_fixture.task import WeatherTask
from tests.control_plane.test_detached import Served, ext_config, ledger_view


def _apps(sim):
    async def now(request):
        web_log(request, page="now")
        return JSONResponse({"now": sim.clock.now.isoformat()})

    async def post(request):
        return JSONResponse({"ok": True})

    ro = Starlette(routes=[Route("/now", now)])
    rw = Starlette(routes=[Route("/write", post, methods=["POST"])])
    return [WebHostSpec("data", ro), WebHostSpec("app", rw, writable=True)]


def test_hosts_served_and_ledgered(tmp_path, monkeypatch):
    monkeypatch.setattr(WeatherTask, "web_apps", lambda self, sim: _apps(sim))
    # tm=A: the agent below waits with `sleep` (refused to the agent token
    # under the cron arms, the config default)
    cfg = ext_config(tmp_path, days=1, cell={"tm": "A", "tlrn": "none",
                                             "sig": "none", "alg": "none"})
    srv = Served(cfg, tmp_path, tmp_path / "server")
    h = srv.handle
    assert set(h["hosts"]) == {"data", "app"} and h["agent_token"] != h["token"]
    run_json = json.loads((tmp_path / "server" / "run.json").read_text())
    assert run_json["hosts"] == h["hosts"]
    ctl = _Http(h["env_url"], h["token"])
    assert ctl.get("/status")["hosts"] == h["hosts"]
    assert ctl.get("/contract")["hosts"] == h["hosts"]
    trig = ctl.get("/trigger/next")

    r = httpx.get(h["hosts"]["data"] + "/now?x=1")
    assert r.status_code == 200 and r.headers["x-sim-time"] == "2021-06-01T00:00:00Z"
    assert r.json()["now"].startswith("2021-06-01T00:00:00")
    # a curl-like write on the writable host is refused (403, browser_only
    # default) and flagged; a browser-like one goes through
    r = httpx.post(h["hosts"]["app"] + "/write")
    assert r.status_code == 403 and "only from the browser" in r.text
    assert httpx.post(h["hosts"]["app"] + "/write",
                      headers={"Sec-Fetch-Mode": "navigate"}).status_code == 200
    assert httpx.get(h["hosts"]["app"] + "/nothing").status_code == 404

    agent = _Http(h["env_url"], h["agent_token"])
    # while the episode bridge marks a program run, even a browser write
    # is refused (a watcher may read, never write); lifted when it ends
    assert agent.post("/program", {"active": True}) == {"active": True}
    r = httpx.post(h["hosts"]["app"] + "/write", headers={"Sec-Fetch-Mode": "navigate"})
    assert r.status_code == 403 and "program is running" in r.text
    assert agent.post("/program", {"active": False}) == {"active": False}
    assert httpx.post(h["hosts"]["app"] + "/write",
                      headers={"Sec-Fetch-Mode": "navigate"}).status_code == 200
    agent.call("sleep", {"until": "2021-06-02T00:00:00Z"})
    ctl.post(f"/trigger/{trig['id']}/exit", {"code": 0, "killed": False, "output": ""})
    assert ctl.get("/trigger/next")["done"]
    res = srv.join()
    assert "web_nonbrowser" in res["flags"] and "web_program_write" in res["flags"]
    rows = [r for r in ledger_view(tmp_path / "server" / "ledger.jsonl")
            if r["type"] == "web"]
    assert rows[0] == {"sim_time": "2021-06-01T00:00:00Z", "type": "web",
                       "cost": 0.0, "host": "data", "method": "GET",
                       "path": "/now", "status": 200, "query": "x=1",
                       "page": "now"}
    assert rows[1]["nonbrowser"] is True and rows[1]["status"] == 403
    assert "nonbrowser" not in rows[2] and rows[2]["status"] == 200
    assert rows[3]["status"] == 404
    assert rows[4]["program_write"] is True and rows[4]["status"] == 403
    assert "program_write" not in rows[5] and rows[5]["status"] == 200
