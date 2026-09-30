"""The launcher hands out sims, never code: POST /runs spawns `harness.serve`
processes and returns run.json handles; a candidate copied to a workspace
and run with `python runtime/actor.py run` completes the run; the server
run dir contains no copy of the candidate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.launcher import Launcher, make_app
from tests.conftest import write_weather_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixture_agents"
RUNTIME = REPO_ROOT / "scaffolds" / "runtime"
START = datetime(2021, 6, 1, tzinfo=timezone.utc)


def write_base(tmp_path: Path, name="wx", days=2, temps=None) -> Path:
    csv = write_weather_csv(tmp_path / f"{name}.csv", START,
                            temps if temps is not None else [20.0] * 24 * days)
    yaml = tmp_path / f"{name}.yaml"
    yaml.write_text(textwrap.dedent(f"""\
        run_id: {name}
        task:
          name: weather_fixture
          location: LA
          weather_csv: {csv}
          data_cutoff: 2021-06-01T00:00:00Z
          threshold_c: 33.0
        sim_start: 2021-06-01T00:00:00Z
        sim_end: 2021-06-0{1 + days}T00:00:00Z
        agent:
          scaffold: "ext:fixture"
        """))
    return yaml


def make_candidate(tmp_path: Path, name="cand") -> Path:
    cand = tmp_path / name
    shutil.copytree(FIXTURES / "poller", cand)
    shutil.copytree(RUNTIME, cand / "runtime",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return cand


def run_actor(cand: Path, handle: dict, where: Path) -> subprocess.CompletedProcess:
    ws = where / "workspace"
    shutil.copytree(cand, ws)
    env = {**os.environ, "ENV_URL": handle["env_url"],
           "ENV_TOKEN": handle["token"], "ENV_MODEL": handle["model"] or ""}
    return subprocess.run([sys.executable, "runtime/actor.py", "run"],
                          cwd=str(ws), env=env, capture_output=True,
                          text=True, timeout=300)


@pytest.fixture
def launcher(tmp_path):
    temps = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24
    base = write_base(tmp_path, temps=temps)
    l = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    yield l
    for rid in list(l.runs):
        l.kill(rid)


def test_launch_run_and_query(launcher, tmp_path):
    client = TestClient(make_app(launcher))
    assert client.get("/bases").json() == {"bases": ["wx"], "heldout": []}
    r = client.post("/runs", json={"base": "wx", "seeds": 2})
    assert r.status_code == 200, r.text
    handles = r.json()["runs"]
    assert [h["run_id"] for h in handles] == ["wx-L001-s0", "wx-L001-s1"]
    assert all(h["env_url"].startswith("http://127.0.0.1:") for h in handles)
    assert client.get("/runs/wx-L001-s0/status").json()["sim"]["done"] is False
    assert client.get("/runs/wx-L001-s0/results").status_code == 409  # not yet

    cand = make_candidate(tmp_path)
    for i, h in enumerate(handles):
        cp = run_actor(cand, h, tmp_path / f"actor{i}")
        assert cp.returncode == 0, cp.stdout + cp.stderr
        assert "done at" in cp.stdout
    # the servers finish on their own once done
    for _ in range(100):
        if all(launcher.runs[h["run_id"]]["proc"].poll() is not None for h in handles):
            break
        time.sleep(0.1)
    res = client.get("/runs/wx-L001-s0/results").json()
    assert res["run_id"] == "wx-L001-s0"
    assert res["task"]["days"][0]["status"] == "ok"
    outs = client.get("/runs/wx-L001-s0/outcomes").json()["outcomes"]
    assert len(outs) == 2 and all(o["type"] == "outcome" for o in outs)
    assert "trigger" in client.get("/runs/wx-L001-s0/ledger").json()["ledger_jsonl"]
    st = client.get("/runs/wx-L001-s0/status").json()
    assert st["results_written"] is True and st["server_alive"] is False
    # the server run dir has no copy of the candidate
    run_dir = tmp_path / "runs" / "wx-L001-s0"
    assert not (run_dir / "workspace").exists()
    assert not (run_dir / "main.py").exists()
    assert (run_dir / "run.json").exists() and (run_dir / "ledger.jsonl").exists()
    # the actor side has the program, its memory/logs, its code_history
    assert (tmp_path / "actor0" / "workspace" / "main.py").exists()
    assert (tmp_path / "actor0" / "code_history").is_dir()
    assert client.get("/runs").json()["runs"][0]["base"] == "wx"
    assert client.get("/runs/nope/status").status_code == 404


def test_mock_llm_stretch(launcher, tmp_path):
    client = TestClient(make_app(launcher))
    r = client.post("/runs", json={"base": "wx", "stretch": "1d", "mock_llm": True})
    (h,) = r.json()["runs"]
    assert h["run_id"] == "wx-L001"
    assert h["sim_end"].startswith("2021-06-02")
    cand = make_candidate(tmp_path)
    cp = run_actor(cand, h, tmp_path / "actor")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    for _ in range(100):
        if launcher.runs[h["run_id"]]["proc"].poll() is not None:
            break
        time.sleep(0.1)
    res = launcher.results(h["run_id"])
    assert res["resources"]["spent_usd"] == 0.0
    assert client.delete(f"/runs/{h['run_id']}").json()["killed"] is True
