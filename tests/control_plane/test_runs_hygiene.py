"""Runs outlive launchers: launch.json carries pids, a fresh
Launcher adopts runs from disk and can stop them; harness.serve exits on
its own when its run dir disappears or a stop marker lands; harness.runs
inventories and kills from the process table without any launcher."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness import runs as hr
from harness import serve as hs
from harness.launcher import Launcher, make_app, pid_runs
from tests.control_plane.test_detached import ext_config
from tests.control_plane.test_launcher import REPO_ROOT, write_base


def _wait(pred, timeout=15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.2)
    return pred()


# -- launcher: pids on disk, adoption, kill after "restart" -------------------------------


def test_launch_json_pids_and_adoption(tmp_path):
    base = write_base(tmp_path)
    l1 = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    c1 = TestClient(make_app(l1))
    h = c1.post("/runs", json={"base": "wx", "mock_llm": True}).json()["runs"][0]
    rid = h["run_id"]
    run_dir = tmp_path / "runs" / rid
    info = json.loads((run_dir / "launch.json").read_text())
    assert info["server_pid"] and info["server_pgid"] == info["server_pid"]
    assert pid_runs(info["server_pid"], rid)
    assert json.loads((run_dir / "run.json").read_text())["pid"] == info["server_pid"]

    # a second launcher over the same root knows the run without having spawned it
    l2 = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    assert rid in l2.runs and l2.runs[rid]["adopted"]
    c2 = TestClient(make_app(l2))
    listed = {r["run_id"]: r for r in c2.get("/runs").json()["runs"]}
    assert listed[rid]["adopted"] and listed[rid]["server_alive"]
    st = c2.get(f"/runs/{rid}/status").json()
    assert st["server_alive"] and st["adopted"] and st["server_pid"] == info["server_pid"]
    assert st["sim"]["done"] is False  # the handle came from run.json
    # stop it from the launcher that did not start it
    res = c2.delete(f"/runs/{rid}").json()
    assert res["killed"] and res["server_killed"]
    assert _wait(lambda: not pid_runs(info["server_pid"], rid))
    assert (run_dir / "killed.json").exists()
    assert c2.get(f"/runs/{rid}/status").json()["server_alive"] is False
    # the first launcher's own handle agrees
    assert l1.runs[rid]["proc"].poll() is not None


def test_unknown_id_is_adopted_on_demand(tmp_path):
    base = write_base(tmp_path)
    l1 = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    l2 = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)  # scanned an empty root
    h = TestClient(make_app(l1)).post("/runs", json={"base": "wx", "mock_llm": True}).json()["runs"][0]
    try:
        st = TestClient(make_app(l2)).get(f"/runs/{h['run_id']}/status").json()
        assert st["adopted"] and st["server_alive"]
    finally:
        l1.kill(h["run_id"])


# -- serve: self-termination ----------------------------------------------------------------


class _Serve:
    def __init__(self, cfg, tmp_path, run_dir):
        self.results = "unset"
        self.error = None
        self.handle = None
        ev = threading.Event()

        def ready(h):
            self.handle = h
            ev.set()

        def target():
            try:
                self.results = asyncio.run(hs.serve(cfg, repo_root=tmp_path,
                                                    run_dir=run_dir, ready=ready))
            except BaseException as e:
                self.error = e
            ev.set()

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        ev.wait(60)


def test_serve_stop_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "WATCH_INTERVAL_S", 0.2)
    run_dir = tmp_path / "server"
    s = _Serve(ext_config(tmp_path, days=1), tmp_path, run_dir)
    assert s.handle and s.handle["pid"] == os.getpid()
    (run_dir / hs.STOP_MARKER).touch()
    s.thread.join(20)
    assert not s.thread.is_alive() and s.error is None
    assert s.results is None
    k = json.loads((run_dir / "killed.json").read_text())
    assert k["reason"] == "stop marker" and k["server_killed"]
    assert not (run_dir / "results.json").exists()


def test_serve_exits_when_run_dir_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "WATCH_INTERVAL_S", 0.2)
    run_dir = tmp_path / "server"
    s = _Serve(ext_config(tmp_path, days=1), tmp_path, run_dir)
    assert s.handle
    shutil.rmtree(run_dir)
    s.thread.join(20)
    assert not s.thread.is_alive()
    assert isinstance(s.error, hs.RunDirGone)


# -- harness.runs: inventory + kill from the process table ----------------------------------


def test_parse_ps_rows():
    lines = [
        "  100  100 /usr/bin/python -m harness.serve --config c.yaml --run-dir /r/task_x/m/w10-L004-s0 --repo-root /r --run-id w10-L004 --model openai:luna --seed 0",
        "  200  200 /usr/bin/python /r/task_x/m/w10-L004-s0/workspace/runtime/actor.py run --workspace /r/task_x/m/w10-L004-s0/workspace --out /r/task_x/m/w10-L004-s0",
        "  300  300 grep harness.serve",
        "  400  400 /usr/bin/python -m harness.runs ps",
    ]
    rows = hr.parse_ps(lines)
    assert [(r["role"], r["pid"], r["run_id"]) for r in rows] == [
        ("server", 100, "w10-L004-s0"), ("actor", 200, "w10-L004-s0")]
    assert rows[0]["model"] == "openai:luna"
    inv = hr.inventory(roots=[], procs=rows)
    assert len(inv) == 1 and inv[0]["state"] == "orphan"  # no such dir on disk


def test_matches():
    rid = "w13full-tmB-reflection-only-L005-qwen-qwen3.7-plus-s2"
    assert hr.matches(rid, rid) and hr.matches(rid, "L005") and hr.matches(rid, "qwen")
    assert hr.matches(rid, "/tmB-.*-s2$/")
    assert not hr.matches(rid, "L004") and not hr.matches(rid, "luna")


def test_runs_ps_and_kill_live(tmp_path):
    base = write_base(tmp_path)
    l = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    hs_ = TestClient(make_app(l)).post("/runs", json={"base": "wx", "mock_llm": True,
                                                       "seeds": 2}).json()["runs"]
    ids = [h["run_id"] for h in hs_]
    try:
        rows = hr.inventory(roots=[tmp_path / "runs"])
        by = {r["run_id"]: r for r in rows}
        assert set(ids) <= set(by)
        assert all(by[i]["state"] == "idle" and by[i]["server_pid"] for i in ids)
        table = hr.format_table(rows)
        assert ids[0] in table and "idle" in table
        # dry run touches nothing
        out = hr.kill_run(by[ids[0]], dry_run=True)
        assert out["server_killed"] and pid_runs(by[ids[0]]["server_pid"], ids[0])
        # the CLI kills by launch number and writes killed.json
        rc = hr.main(["kill", "L001", "--root", str(tmp_path / "runs")])
        assert rc == 0
        for i in ids:
            assert _wait(lambda i=i: not pid_runs(by[i]["server_pid"], i))
            assert json.loads((tmp_path / "runs" / i / "killed.json").read_text())["server_killed"]
        after = {r["run_id"]: r for r in hr.inventory(roots=[tmp_path / "runs"], include_finished=True)}
        assert all(after[i]["state"] == "killed" for i in ids)
        assert hr.main(["kill", "L001", "--root", str(tmp_path / "runs")]) == 1  # nothing alive
    finally:
        for i in ids:
            l.kill(i)


def test_runs_stop_marker_via_cli(tmp_path):
    base = write_base(tmp_path)
    l = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    h = TestClient(make_app(l)).post("/runs", json={"base": "wx", "mock_llm": True}).json()["runs"][0]
    rid, pid = h["run_id"], h["pid"]
    try:
        assert hr.main(["stop", rid, "--root", str(tmp_path / "runs")]) == 0
        assert _wait(lambda: not pid_runs(pid, rid), timeout=20)
        k = json.loads((tmp_path / "runs" / rid / "killed.json").read_text())
        assert k["reason"] == "stop marker"
    finally:
        l.kill(rid)


def test_cli_module_runs():
    out = subprocess.run([sys.executable, "-m", "harness.runs", "ps", "--root",
                          "/nonexistent"], capture_output=True, text=True, cwd=REPO_ROOT)
    # processes are joined regardless of --root, so the table may be non-empty
    assert out.returncode == 0 and (out.stdout.startswith("state") or "(no runs)" in out.stdout)
