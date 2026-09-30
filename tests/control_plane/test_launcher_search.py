"""Launcher session extensions: launcher-spawned actors,
held-out bases (validate route, trimmed results, hidden traces, summary
files), a launch counter derived from disk, the leak audit and the spend
caps, and clock independence of concurrent runs."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.launcher import (Launcher, leak_audit, load_registry, make_app,
                              resolve_bind)
from tests.control_plane.test_launcher import (REPO_ROOT, make_candidate,
                                               write_base)

TEMPS = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24


def _wait_done(launcher: Launcher, run_ids, timeout=120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        recs = [launcher.runs[r] for r in run_ids]
        if all(r["proc"].poll() is not None for r in recs) and \
                all(r["actor"] is None or r["actor"].poll() is not None for r in recs):
            return
        time.sleep(0.1)
    raise AssertionError("runs did not finish in time")


@pytest.fixture
def world(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    cand = make_candidate(session, "candidates/v1")
    search = write_base(tmp_path, "wx", temps=TEMPS)
    heldout = write_base(tmp_path, "hx", temps=TEMPS)
    launcher = Launcher({"wx": {"yaml": search, "heldout": False},
                         "hx": {"yaml": heldout, "heldout": True}},
                        session / "runs", REPO_ROOT,
                        session_dir=session, heldout_root=tmp_path / "heldout")
    yield session, cand, launcher
    for rid in list(launcher.runs):
        launcher.kill(rid)


def test_launcher_spawns_actor_and_run_lands_in_session(world):
    session, cand, launcher = world
    client = TestClient(make_app(launcher))
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "seeds": 1})
    assert r.status_code == 200, r.text
    (h,) = r.json()["runs"]
    assert h["run_id"] == "wx-v1-L001-s0" and "actor_pid" in h
    _wait_done(launcher, [h["run_id"]])
    rd = session / "runs" / h["run_id"]
    assert (rd / "workspace" / "main.py").exists()
    assert (rd / "workspace" / "cell_config.py").exists()  # from the sim
    assert (rd / "actor.log").exists() and "done at" in (rd / "actor.log").read_text()
    assert (rd / "launch.json").exists() and (rd / "results.json").exists()
    assert (rd / "ledger.jsonl").exists() and (rd / "config.json").exists()
    st = client.get(f"/runs/{h['run_id']}/status").json()
    assert st["actor_alive"] is False and st["results_written"] is True
    assert st["candidate"] == "v1" and st["heldout"] is False
    assert set(st["results_summary"]) == {"primary", "spent_usd", "flags"}
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "run_root": "validation"})
    assert r.status_code == 400 and "run_root" in r.text
    res = client.get(f"/runs/{h['run_id']}/results").json()
    assert "task" in res  # search runs disclose everything
    assert client.get(f"/runs/{h['run_id']}/ledger").status_code == 200


def test_heldout_base_only_via_validate_and_traces_hidden(world):
    session, cand, launcher = world
    client = TestClient(make_app(launcher))
    assert client.get("/bases").json() == {"bases": ["hx", "wx"], "heldout": ["hx"]}
    r = client.post("/runs", json={"base": "hx", "candidate": "candidates/v1"})
    assert r.status_code == 403
    r = client.post("/validate", json={"base": "wx", "candidate": "candidates/v1"})
    assert r.status_code == 400  # search base is not validated
    r = client.post("/validate", json={"base": "hx", "candidate": "candidates/v1",
                                       "seeds": 1})
    assert r.status_code == 200, r.text
    (h,) = r.json()["runs"]
    assert set(h) == {"run_id", "sim_start", "sim_end", "heldout"}  # no url/token
    rid = h["run_id"]
    _wait_done(launcher, [rid])
    # the run lives outside the session dir, traces included
    assert (launcher.heldout_root / rid / "workspace" / "main.py").exists()
    assert not (session / "runs" / rid).exists()
    assert client.get(f"/runs/{rid}/ledger").status_code == 403
    assert client.get(f"/runs/{rid}/outcomes").status_code == 403
    res = client.get(f"/runs/{rid}/results").json()
    assert "task" not in res and res["performance"]["primary"]["name"]
    st = client.get(f"/runs/{rid}/status").json()
    assert st["heldout"] is True and "run_dir" not in st
    # the summary file appears in the session's validation/ dir
    for _ in range(100):
        if (session / "validation" / f"{rid}.json").exists():
            break
        time.sleep(0.1)
    summary = json.loads((session / "validation" / f"{rid}.json").read_text())
    assert summary["heldout"] is True and summary["candidate"] == "v1"
    assert set(summary) >= {"performance", "resources", "flags", "base", "seed"}
    assert "task" not in summary
    pub = client.get("/runs").json()["runs"][0]
    assert pub["heldout"] is True and "handle" not in pub and "run_dir" not in pub


def test_backfill_summary_and_counter_survive_restart(world, tmp_path):
    session, cand, launcher = world
    client = TestClient(make_app(launcher))
    r = client.post("/validate", json={"base": "hx", "candidate": "candidates/v1"})
    rid = r.json()["runs"][0]["run_id"]
    _wait_done(launcher, [rid])
    for _ in range(100):
        if (session / "validation" / f"{rid}.json").exists():
            break
        time.sleep(0.1)
    (session / "validation" / f"{rid}.json").unlink()
    # a fresh launcher over the same roots: summary backfilled, counter continues
    fresh = Launcher(launcher.bases | {}, session / "runs", REPO_ROOT,
                     heldout={"hx"}, session_dir=session,
                     heldout_root=launcher.heldout_root)
    assert (session / "validation" / f"{rid}.json").exists()
    (h,) = fresh.launch("wx", None, None, False, candidate=cand)
    assert h["run_id"] == "wx-v1-L002"
    _wait_done(fresh, [h["run_id"]])


def test_leak_audit_and_candidate_placement(world, tmp_path):
    session, cand, launcher = world
    launcher.forbidden = {"refuse": ["deadbeefcafe0001"], "warn": ["616902"]}
    (cand / "notes.md").write_text("watch article deadbeefcafe0001 and market 616902\n")
    client = TestClient(make_app(launcher))
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1"})
    assert r.status_code == 400 and "leak audit" in r.text and "notes.md" in r.text
    hits = leak_audit(cand, launcher.forbidden)
    assert hits["refuse"][0]["token"] == "deadbeefcafe0001"
    assert hits["warn"][0]["token"] == "616902"
    (cand / "notes.md").write_text("market 616902\n")  # warn only: launches, reports
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "mock_llm": True})
    assert r.status_code == 200, r.text
    assert r.json()["warnings"][0]["token"] == "616902"
    _wait_done(launcher, [r.json()["runs"][0]["run_id"]])
    # a candidate outside the session dir is refused
    outside = make_candidate(tmp_path, "outside")
    r = client.post("/runs", json={"base": "wx", "candidate": str(outside)})
    assert r.status_code == 400 and "inside the session dir" in r.text
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/nope"})
    assert r.status_code == 400


def test_spend_and_validation_caps(world):
    session, cand, launcher = world
    client = TestClient(make_app(launcher))
    # a shipped past run: results.json but no launch.json -> never counted
    past = launcher.runs_root / "smoke-old-run-s0"
    past.mkdir(parents=True)
    (past / "results.json").write_text(json.dumps(
        {"resources": {"spent_usd": 999.0}, "performance": {}, "flags": []}))
    # an unfinished run of an earlier launcher of this session: reserved
    live = launcher.runs_root / "wx-v0-L000-s0"
    live.mkdir()
    (live / "launch.json").write_text(json.dumps({"mock_llm": False}))
    (live / "config.json").write_text(json.dumps({"budget_usd": 50.0}))
    acct = client.get("/budget").json()
    assert acct["finished"] == [] and acct["live"] == [
        {"run_id": "wx-v0-L000-s0", "reserved_usd": 50.0}]
    assert acct["committed_usd"] == 50.0 and acct["remaining_usd"] is None
    launcher.max_spend_usd = 110.0  # the fixture base's wallet is 50
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "seeds": 2})
    assert r.status_code == 402 and "spend cap" in r.text and "remaining $60.00" in r.text
    assert client.get("/budget").json()["remaining_usd"] == 60.0
    (live / "results.json").write_text(json.dumps(
        {"resources": {"spent_usd": 3.25}, "performance": {}, "flags": []}))
    acct = client.get("/budget").json()
    assert acct["finished"] == [{"run_id": "wx-v0-L000-s0", "spent_usd": 3.25}]
    assert acct["remaining_usd"] == 106.75
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "seeds": 2, "mock_llm": True})  # free
    assert r.status_code == 200, r.text
    _wait_done(launcher, [h["run_id"] for h in r.json()["runs"]])
    assert launcher.committed_usd() == 3.25  # mocked runs add nothing
    launcher.max_spend_usd = None
    launcher.max_validation_runs = 1
    r = client.post("/validate", json={"base": "hx", "candidate": "candidates/v1",
                                       "seeds": 2})
    assert r.status_code == 402 and "held-out cap" in r.text


def test_concurrent_runs_have_independent_clocks(world):
    session, cand, launcher = world
    handles = launcher.launch("wx", 2, None, False, candidate=cand)
    ids = [h["run_id"] for h in handles]
    _wait_done(launcher, ids)
    seqs = []
    for rid in ids:
        rows = [json.loads(l) for l in
                (session / "runs" / rid / "ledger.jsonl").read_text().splitlines()]
        times = [r["sim_time"] for r in rows]
        assert times == sorted(times), "a run's clock never goes backwards"
        assert (session / "runs" / rid / "results.json").exists()
        seqs.append([r["sim_time"] for r in rows if r["type"] == "trigger"])
    assert seqs[0] == seqs[1]  # same fixture, same schedule, both complete


def test_registry_v2_loads(tmp_path):
    base = write_base(tmp_path, "wx", temps=TEMPS)
    session = tmp_path / "improver" / "s01"
    session.mkdir(parents=True)
    forb = tmp_path / "improver" / "s01.forbidden.json"
    forb.write_text(json.dumps({"refuse": ["x"], "warn": []}))
    (tmp_path / "improver" / "s01.launcher.json").write_text(json.dumps({
        "session": "s01", "task": "weather_fixture", "session_dir": str(session),
        "bases": {"wx": {"yaml": str(base), "heldout": False},
                  "hx": {"yaml": str(base), "heldout": True}},
        "runs_root": str(session / "runs"), "heldout_root": str(tmp_path / "ho"),
        "max_spend_usd": 12.5, "max_validation_runs": 3,
        "forbidden_tokens": str(forb)}))
    reg = load_registry(str(session), [])
    assert set(reg["bases"]) == {"wx", "hx"} and reg["heldout"] == {"hx"}
    assert reg["max_spend_usd"] == 12.5 and reg["forbidden"]["refuse"] == ["x"]
    assert Path(reg["session_dir"]) == session
    # the flat v1 form still loads
    (tmp_path / "improver" / "s02.launcher.json").write_text(json.dumps({
        "session": "s02", "bases": {"wx": str(base)}}))
    reg = load_registry(str(tmp_path / "improver" / "s02"), [])
    assert set(reg["bases"]) == {"wx"} and reg["heldout"] == set()


def test_launcher_binds_where_the_session_expects(tmp_path):
    """One launcher per session, one port per launcher: with no --port the
    launcher listens on the registry's launcher_url, the URL baked into
    the session's HOWTO at init — two sessions on two ports never collide
    or cross-talk by a forgotten flag."""
    base = write_base(tmp_path, "wx", temps=TEMPS)
    session = tmp_path / "improver" / "s07"
    session.mkdir(parents=True)
    (tmp_path / "improver" / "s07.launcher.json").write_text(json.dumps({
        "session": "s07", "task": "weather_fixture", "session_dir": str(session),
        "bases": {"wx": {"yaml": str(base), "heldout": False}},
        "launcher_url": "http://127.0.0.1:8766"}))
    reg = load_registry(str(session), [])
    assert reg["launcher_url"] == "http://127.0.0.1:8766"
    assert resolve_bind(reg, None, None) == ("127.0.0.1", 8766)
    assert resolve_bind(reg, None, 9000) == ("127.0.0.1", 9000)  # explicit wins
    assert resolve_bind(reg, "0.0.0.0", None) == ("0.0.0.0", 8766)
    assert resolve_bind({}, None, None) == ("127.0.0.1", 8765)  # no registry: defaults
    assert resolve_bind({"launcher_url": "http://localhost"}, None, None) == ("localhost", 8765)


def test_stopped_runs_settle_to_their_ledger_and_get_the_repo_runner(world):
    session, cand, launcher = world
    client = TestClient(make_app(launcher))
    # a run an earlier launcher stopped: no results.json, a killed marker,
    # a ledger -> finished at the ledger's spend, never live
    dead = launcher.runs_root / "wx-v0-L001-s0"
    dead.mkdir(parents=True)
    (dead / "launch.json").write_text(json.dumps({"mock_llm": False}))
    (dead / "config.json").write_text(json.dumps({"budget_usd": 50.0}))
    (dead / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"type": "llm", "cost": 0.5}, {"type": "news_search", "cost": 0.25},
        {"type": "trigger"}]))
    (dead / "killed.json").write_text(json.dumps({"killed_at": "2026-09-08T00:00:00Z"}))
    acct = client.get("/budget").json()
    assert acct["live"] == []
    assert acct["finished"] == [{"run_id": "wx-v0-L001-s0", "killed": True,
                                 "spent_usd": 0.75}]
    # a real stop: launch, kill at once
    r = client.post("/runs", json={"base": "wx", "candidate": "candidates/v1",
                                   "mock_llm": True})
    assert r.status_code == 200, r.text
    rid = r.json()["runs"][0]["run_id"]
    rd = launcher.runs_root / rid
    # the run copy carries the environment's current runner, byte for byte
    assert (rd / "workspace" / "runtime" / "actor.py").read_bytes() == \
        (REPO_ROOT / "scaffolds" / "runtime" / "actor.py").read_bytes()
    out = client.delete(f"/runs/{rid}").json()
    assert out["killed"] is True and "spent_usd" in out
    st = client.get(f"/runs/{rid}/status").json()
    assert st["server_alive"] is False and st["actor_alive"] is False
    finished_first = (rd / "results.json").exists()  # a mocked run can be that quick
    assert (rd / "killed.json").exists() != finished_first
    assert st["killed"] == (not finished_first)
    listed = {x["run_id"]: x for x in client.get("/runs").json()["runs"]}
    assert listed[rid]["killed"] == st["killed"]
