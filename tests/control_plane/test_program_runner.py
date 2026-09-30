"""Client-side authored programs:
the same gatekeeper run through the server-side ProgramApp (constructor-built tm=B)
and through scaffolds/runtime/program.run_program against a served ext:
tm=B sim produces the same wake times, the same billed fetches and the
same handover — the parity precondition for the two modes."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness.api import build_tool_registry
from harness.runtime import Sim
from scaffolds.runtime import program as rp
from scaffolds.runtime.actor import Runner, _Api
from scaffolds.runtime.env_client import Env
from tests.conftest import make_config, make_weather_task, write_weather_csv
from tests.control_plane.test_detached import Served, ledger_view

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)

GATEKEEPER = '''\
import envkit
from datetime import timedelta

while True:
    now = envkit.now()
    day = now.strftime("%Y-%m-%d")
    res = envkit.get_weather(start=f"{day}T00:00:00Z",
                             end=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    temps = res["hourly"]["temperature_2m"]
    if temps and max(temps) >= 33.0:
        envkit.handover({"day": day, "max": max(temps), "at": now.isoformat()})
    envkit.wait(now + timedelta(hours=1))
'''

TEMPS = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24  # crosses 33 at 14:00 day 1


def cfg_for(tmp_path, scaffold, tm="B"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = write_weather_csv(tmp_path / "w.csv", START, TEMPS)
    return make_config(sim_start=START, sim_end=START + timedelta(days=2),
                       data_cutoff=START, weather_csv=csv,
                       cell={"tm": tm, "tlrn": "none", "sig": "none",
                             "alg": "none"},
                       agent={"scaffold": scaffold})


def events(path_or_sim, types=("weather", "sleep")):
    if isinstance(path_or_sim, Path):
        evs = ledger_view(path_or_sim)
    else:
        evs = [{k: v for k, v in e.items() if k not in ("seq", "real_time")}
               for e in path_or_sim.ledger.events]
    return [(e["type"], e["sim_time"], e.get("until"), e.get("woke_for"))
            for e in evs if e["type"] in types]


def test_client_side_runner_matches_program_app(tmp_path, monkeypatch):
    # -- server side: the constructor's tm=B ProgramApp, in-process ---------------------------
    cfg_s = cfg_for(tmp_path / "s", "react")
    run_dir = tmp_path / "s" / "run"
    ws = run_dir / "workspace"
    ws.mkdir(parents=True)
    sim = Sim(cfg_s, run_dir, ws, make_weather_task(cfg_s))
    reg = build_tool_registry(sim)
    assert "run_program" in reg and "sleep" not in reg

    def call(name, **args):
        return asyncio.run(reg[name][1](args))

    call("write_file", path="gate.py", content=GATEKEEPER)
    server_out = call("run_program", path="gate.py")
    assert server_out["woke_for"] == "handover"
    assert server_out["payload"]["day"] == "2021-06-01"
    assert server_out["now"] == "2021-06-01T15:00:00Z"  # 14:00 hour observable at 15:00
    server_events = events(sim)

    # -- client side: served ext: tm=B sim, program in this process ---------------
    cfg_c = cfg_for(tmp_path / "c", "ext:gate")
    server_dir = tmp_path / "c" / "server"
    srv = Served(cfg_c, tmp_path / "c", server_dir)
    h = srv.handle
    monkeypatch.setenv("ENV_URL", h["env_url"])
    monkeypatch.setenv("ENV_TOKEN", h["token"])
    env = Env()
    assert "run_program" not in {t["name"] for t in env.tools()}
    assert "sleep" in {t["name"] for t in env.tools()}  # the program's only clock
    root = tmp_path / "c" / "actor" / "agents" / "w1"
    root.mkdir(parents=True)
    (root / "gate.py").write_text(GATEKEEPER)
    # as inside a real invocation: the bootstrap trigger has been taken
    api = _Api(h["env_url"], h["token"])
    trig = api.get("/trigger/next")
    client_out = rp.run_program(root / "gate.py", env=env)
    api.post(f"/trigger/{trig['id']}/exit", {"code": 0, "killed": False,
                                            "output": ""})
    assert (root / "envkit.py").exists()  # the sim's published stub
    assert client_out["woke_for"] == "handover", client_out
    assert client_out["payload"] == server_out["payload"]
    assert client_out["now"] == server_out["now"]
    assert client_out["fetches"] == 16 and client_out["waits"] == 15

    # drain the served sim to its end with a no-op program so it finishes
    ws_c = tmp_path / "c" / "actor" / "workspace"
    ws_c.mkdir()
    (ws_c / "main.py").write_text("print('noop')\n")
    Runner(h["env_url"], h["token"], "", ws_c, ws_c.parent, quiet=True).run()
    srv.join()

    client_events = events(server_dir / "ledger.jsonl")
    # every wake time and every billed fetch identical; the server-side
    # path additionally logs its run_program event (client side: none)
    assert client_events == server_events
    assert [t for t, *_ in server_events].count("weather") == 16
    types_c = {e["type"] for e in ledger_view(server_dir / "ledger.jsonl")}
    assert "run_program" not in types_c


def test_validate_and_errors(tmp_path, monkeypatch):
    cfg_c = cfg_for(tmp_path / "c", "ext:gate")
    srv = Served(cfg_c, tmp_path / "c", tmp_path / "c" / "server")
    h = srv.handle
    monkeypatch.setenv("ENV_URL", h["env_url"])
    monkeypatch.setenv("ENV_TOKEN", h["token"])
    env = Env()
    root = tmp_path / "c" / "agents"
    root.mkdir(parents=True)
    (root / "gate.py").write_text(GATEKEEPER)
    api = _Api(h["env_url"], h["token"])
    trig = api.get("/trigger/next")
    v = rp.run_program(root / "gate.py", env=env, validate=True)
    assert v["validate"] == "ok" and v["reached"] == "get_weather"
    (root / "bad.py").write_text("import envkit\nenvkit.wait('2020-01-01T00:00:00Z')\n")
    e = rp.run_program(root / "bad.py", env=env)
    assert e["woke_for"] == "error" and "must be in the future" in e["error"]
    (root / "syn.py").write_text("def (:\n")
    e2 = rp.run_program(root / "syn.py", env=env)
    assert e2["woke_for"] == "error" and "syntax error" in e2["error"]
    (root / "ret.py").write_text("import envkit\nx = envkit.now()\n")
    r = rp.run_program(root / "ret.py", env=env)
    assert r["woke_for"] == "return"
    # nothing above moved the clock or billed anything but the fetch
    st = rp.Env().call("get_time")
    assert st["now"] == "2021-06-01T00:00:00Z"
    api.post(f"/trigger/{trig['id']}/exit", {"code": 0, "killed": False,
                                            "output": ""})
    ws_c = tmp_path / "c" / "ws"
    ws_c.mkdir()
    (ws_c / "main.py").write_text("print('noop')\n")
    Runner(h["env_url"], h["token"], "", ws_c, ws_c.parent, quiet=True).run()
    srv.join()
