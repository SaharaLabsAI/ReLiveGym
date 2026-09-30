"""Server ↔ actor detach: a sim
served by harness.serve, driven by scaffolds/runtime/actor.py from a
workspace the server never sees, produces the same ledger as the combined
harness.run — and the lifecycle routes behave as documented."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.run import run_experiment
from harness.serve import serve
from scaffolds.runtime.actor import Runner, _Api, check
from tests.conftest import make_config, write_weather_csv

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)
FIXTURES = Path(__file__).resolve().parents[1] / "fixture_agents"


def ext_config(tmp_path, days=2, temps=None, **over):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = write_weather_csv(tmp_path / "w.csv", START,
                            temps if temps is not None else [20.0] * 24 * days)
    return make_config(sim_start=START, sim_end=START + timedelta(days=days),
                       data_cutoff=START, weather_csv=csv,
                       agent={"scaffold": "ext:fixture"}, **over)


class Served:
    """harness.serve in a background thread; `.handle` once bound."""

    def __init__(self, cfg, repo_root: Path, run_dir: Path):
        self.handle = None
        self.results = None
        self.error = None
        self._ready = threading.Event()

        def target():
            try:
                self.results = asyncio.run(serve(
                    cfg, repo_root=repo_root, run_dir=run_dir,
                    ready=self._on_ready))
            except BaseException as e:  # surfaced by join()
                self.error = e
                self._ready.set()

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        self._ready.wait(60)
        if self.error:
            raise self.error
        assert self.handle, "server never announced its handle"

    def _on_ready(self, handle):
        self.handle = handle
        self._ready.set()

    def join(self, timeout=120):
        self.thread.join(timeout)
        if self.error:
            raise self.error
        assert self.results is not None, "server did not finish"
        return self.results


def ledger_view(path: Path) -> list[dict]:
    return [{k: v for k, v in json.loads(l).items()
             if k not in ("seq", "real_time")}
            for l in path.read_text().splitlines() if l.strip()]


def test_detached_equals_combined(tmp_path):
    temps = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24
    # combined mode (harness.run) on the fixture, tagged ext:
    cfg = ext_config(tmp_path / "a", temps=temps)
    combined = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path / "a", run_dir=tmp_path / "a" / "run",
        scaffold_src=FIXTURES / "poller"))

    # detached: serve, then a runner on a copy of the same program.
    # Same csv FILE as the combined run: cell_config.py embeds the task
    # params (TASK_PARAMS), so
    # per-side csv paths would break the byte-equality pin below.
    cfg2 = ext_config(tmp_path / "b", temps=temps)
    cfg2.task["weather_csv"] = cfg.task["weather_csv"]
    server_dir = tmp_path / "b" / "server"
    srv = Served(cfg2, tmp_path / "b", server_dir)
    ws = tmp_path / "b" / "actor" / "workspace"
    shutil.copytree(FIXTURES / "poller", ws)
    h = srv.handle
    out = Runner(h["env_url"], h["token"], h["model"], ws, ws.parent,
                 quiet=True).run()
    assert out["done"] is True
    detached = srv.join()

    # same ledger (modulo bookkeeping), same score
    assert ledger_view(server_dir / "ledger.jsonl") == \
        ledger_view(tmp_path / "a" / "run" / "ledger.jsonl")
    assert detached["performance"] == combined["performance"]
    assert detached["resources"]["spent_usd"] == combined["resources"]["spent_usd"]

    # the runner wrote the sim's contract into ITS workspace...
    assert (ws / "cell_config.py").read_text() == \
        (tmp_path / "a" / "run" / "workspace" / "cell_config.py").read_text()
    assert (ws / "INSTRUCTION.md").exists()
    assert (ws.parent / "code_history").is_dir()
    assert json.loads((ws.parent / "workspace_manifest.json").read_text())["files"]
    # ...and the server dir holds no copy of the program
    assert not (server_dir / "workspace").exists()
    assert (server_dir / "results.json").exists()
    assert json.loads((server_dir / "workspace_manifest.json").read_text())["files"] is None
    assert (server_dir / "run.json").exists()


def test_lifecycle_routes(tmp_path):
    cfg = ext_config(tmp_path, days=1, watchdog_seconds=5)
    srv = Served(cfg, tmp_path, tmp_path / "server")
    h = srv.handle
    api = _Api(h["env_url"], h["token"])
    contract = api.get("/contract")
    assert {t["name"] for t in contract["tools"]} >= {
        "get_time", "get_costs", "sleep", "set_party", "set_crontab",
        "get_weather", "notify"}
    assert "TM = " in contract["cell_config_py"]
    assert "def get_weather" in contract["envkit_py"]
    assert "def notify" in contract["envkit_py"]
    assert "def sleep" not in contract["envkit_py"]
    st = api.get("/status")
    assert st["done"] is False and st["sim_now"] == "2021-06-01T00:00:00Z"
    act = api.get("/activity")
    assert act["watchdog_seconds"] == 5 and act["active_trigger"] is None

    # the clock only moves on `next`; a second `next` while active is 409
    trig = api.get("/trigger/next")
    assert trig["id"] == "__bootstrap__" and trig["kind"] == "at"
    with pytest.raises(Exception, match="409"):
        api.get("/trigger/next")
    resp = api.post(f"/trigger/{trig['id']}/exit",
                    {"code": 1, "killed": False, "output": "boom"})
    assert resp["crashed"] is True and resp["failed_degenerate"] is False
    # after a crash the next trigger is crash_recovery (the fallback midnight
    # == sim_end here, so the run is done instead)
    nxt = api.get("/trigger/next")
    assert nxt.get("done") is True
    results = srv.join()
    assert "failed-degenerate" not in results["flags"]
    assert [e["type"] for e in ledger_view(tmp_path / "server" / "ledger.jsonl")
            if e["type"] in ("trigger", "agent_exit")] == ["trigger", "agent_exit"]


def test_check_offline(tmp_path):
    ws = tmp_path / "cand"
    shutil.copytree(FIXTURES / "poller", ws)
    problems = check(ws)
    assert problems == ["no runtime/ library (copy runtime/ into the program)"]
    shutil.copytree(Path(__file__).resolve().parents[2] / "scaffolds" / "runtime",
                    ws / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    (ws / "bad.py").write_text("def (:\n")
    problems = check(ws)
    assert problems and problems[0].startswith("bad.py: syntax error")
