"""Launcher side of an online-update stage (harness/launcher.py `launch(...,
pause_at=, resume_from=)`): a paused stage lands as a checkpoint, a resume copies the whole
snapshot (state included) and continues under a new seed, the chain's
score equals a continuous run's, paused runs book their ledger spend,
kill leaves a checkpoint alone, and the guards: not-a-checkpoint,
per-launch forbidden tokens, a session launcher's resume fence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.launcher import Launcher, LeakError, make_app
from tests.control_plane.test_launcher import REPO_ROOT, make_candidate, write_base
from tests.control_plane.test_launcher_search import _wait_done

TEMPS3 = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24 * 2
CUT = "2021-06-02T00:00:00Z"


@pytest.fixture
def launcher(tmp_path):
    base = write_base(tmp_path, "wx", days=3, temps=TEMPS3)
    l = Launcher({"wx": base}, tmp_path / "runs", REPO_ROOT)
    yield l
    for rid in list(l.runs):
        l.kill(rid)


@pytest.mark.slow
def test_stage_chain_through_the_launcher(launcher, tmp_path):
    cand = make_candidate(tmp_path, "A1")
    (h1,) = launcher.launch("wx", [0], None, False, candidate=cand,
                            pause_at=CUT, extra={"stage": 1})
    assert h1["run_id"] == "wx-A1-L001-s0" and h1["pause_at"] == "2021-06-02T00:00:00+00:00"
    _wait_done(launcher, [h1["run_id"]])
    rd1 = launcher.runs_root / h1["run_id"]
    assert (rd1 / "checkpoint.json").exists() and not (rd1 / "results.json").exists()
    launch = json.loads((rd1 / "launch.json").read_text())
    assert launch["stage"] == 1 and launch["pause_at"] == CUT and launch["resume_from"] is None
    st = launcher.status(h1["run_id"])
    assert st["paused"] is True and st["results_written"] is False and st["killed"] is False
    assert launcher.public(h1["run_id"])["paused"] is True
    acct = launcher.budget()  # a paused run books its ledger spend, reserves nothing
    assert acct["live"] == [] and acct["finished"][0]["run_id"] == h1["run_id"]
    assert acct["finished"][0]["paused"] is True
    # kill on a paused run: nothing to stop, no killed.json
    assert launcher.kill(h1["run_id"])["server_killed"] is False
    assert not (rd1 / "killed.json").exists()

    # the snapshot: the stage-1 workspace, with state the improver added
    snap = tmp_path / "A2"
    shutil.copytree(rd1 / "workspace", snap, ignore=shutil.ignore_patterns("__pycache__"))
    (snap / "memory" / "note.txt").write_text("carried across the cut\n")
    (snap / "state.json").write_text('{"k": 1}')
    (h2,) = launcher.launch("wx", [1], None, False, candidate=snap, resume_from=rd1,
                            extra={"stage": 2})
    assert h2["run_id"] == "wx-A2-L002-s1" and h2["resumed_from"] == str(rd1.resolve())
    _wait_done(launcher, [h2["run_id"]])
    rd2 = launcher.runs_root / h2["run_id"]
    assert (rd2 / "results.json").exists() and not (rd2 / "checkpoint.json").exists()
    assert json.loads((rd2 / "resume.json").read_text())["parent"] == str(rd1.resolve())
    assert json.loads((rd2 / "launch.json").read_text())["resume_from"] == str(rd1.resolve())
    # the whole snapshot travelled, state included; the runner is the repo's
    assert (rd2 / "workspace" / "memory" / "note.txt").read_text().startswith("carried")
    assert (rd2 / "workspace" / "state.json").exists()
    assert (rd2 / "workspace" / "runtime" / "actor.py").read_bytes() == \
        (REPO_ROOT / "scaffolds" / "runtime" / "actor.py").read_bytes()
    rows = [json.loads(l) for l in (rd2 / "ledger.jsonl").read_text().splitlines()]
    assert [e["type"] for e in rows if e["type"] == "resume"] == ["resume"]
    assert [e["seq"] for e in rows] == list(range(1, len(rows) + 1))

    # the chain scores exactly as a continuous run of the same program
    (h3,) = launcher.launch("wx", [0], None, False, candidate=cand)
    _wait_done(launcher, [h3["run_id"]])
    cont = json.loads((launcher.runs_root / h3["run_id"] / "results.json").read_text())
    chain = json.loads((rd2 / "results.json").read_text())
    assert chain["performance"] == cont["performance"] and chain["task"] == cont["task"]
    assert chain["resources"]["counts"]["trigger"] == cont["resources"]["counts"]["trigger"]


@pytest.mark.slow
def test_resume_guards(launcher, tmp_path):
    cand = make_candidate(tmp_path, "A1")
    with pytest.raises(FileNotFoundError):
        launcher.launch("wx", [0], None, True, candidate=cand, resume_from=tmp_path / "nope")
    # per-launch forbidden tokens: the online session audits each stage
    (cand / "notes.md").write_text("article deadbeefcafe0001\n")
    with pytest.raises(LeakError):
        launcher.launch("wx", [0], None, True, candidate=cand,
                        forbidden={"refuse": ["deadbeefcafe0001"]})
    assert launcher.forbidden == {}  # the registry's tokens are untouched
    (h,) = launcher.launch("wx", [0], None, True, candidate=cand,
                           forbidden={"refuse": ["somethingelse"]})
    _wait_done(launcher, [h["run_id"]])
    # stretch and resume do not combine
    (h1,) = launcher.launch("wx", [0], None, True, candidate=cand, pause_at=CUT)
    _wait_done(launcher, [h1["run_id"]])
    rd1 = launcher.runs_root / h1["run_id"]
    with pytest.raises(ValueError, match="stretch"):
        launcher.launch("wx", [0], "1d", True, candidate=cand, resume_from=rd1)
    # HTTP: the body carries pause_at / resume_from; bad parents are 400
    client = TestClient(make_app(launcher))
    r = client.post("/runs", json={"base": "wx", "candidate": str(cand),
                                   "mock_llm": True, "resume_from": str(tmp_path / "nope")})
    assert r.status_code == 400 and "not a paused run" in r.text
    r = client.post("/runs", json={"base": "wx", "candidate": str(rd1 / "workspace"),
                                   "mock_llm": True, "resume_from": str(rd1), "seeds": [3]})
    assert r.status_code == 200, r.text
    (h2,) = r.json()["runs"]
    assert h2["run_id"].endswith("-s3") and h2["resumed_from"] == str(rd1.resolve())
    _wait_done(launcher, [h2["run_id"]])
    assert (launcher.runs_root / h2["run_id"] / "results.json").exists()


def test_session_launcher_fences_resume(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    cand = make_candidate(session, "candidates/v1")
    base = write_base(tmp_path, "wx", days=3, temps=TEMPS3)
    l = Launcher({"wx": {"yaml": base, "heldout": False}}, session / "runs", REPO_ROOT,
                 session_dir=session, heldout_root=tmp_path / "heldout")
    try:
        (h1,) = l.launch("wx", [0], None, True, candidate="candidates/v1", pause_at=CUT)
        _wait_done(l, [h1["run_id"]])
        rd1 = l.runs_root / h1["run_id"]
        # a checkpoint outside the session's runs root is out of reach
        outside = tmp_path / "elsewhere"
        shutil.copytree(rd1, outside)
        with pytest.raises(ValueError, match="runs root"):
            l.launch("wx", [0], None, True, candidate="candidates/v1", resume_from=outside)
        # inside it, relative to the session dir, is fine
        (h2,) = l.launch("wx", [0], None, True, candidate="candidates/v1",
                         resume_from=Path("runs") / h1["run_id"])
        _wait_done(l, [h2["run_id"]])
        assert (l.runs_root / h2["run_id"] / "results.json").exists()
    finally:
        for rid in list(l.runs):
            l.kill(rid)


def test_budget_counts_this_session_only_and_chains_once(tmp_path):
    """The online session shares its runs root with every other run of the
    model: budget() must count only launches naming the session, book a
    resumed run's own rows only (its ledger starts with the parent's), and
    reserve for a live resumed run what the chain can still spend."""
    base = write_base(tmp_path, "wx", days=3, temps=TEMPS3)
    root = tmp_path / "runs"
    root.mkdir()

    def run(name, *, session, results=None, ledger=(), resume_spent=None, live=False,
            checkpoint=False):
        d = root / name
        d.mkdir()
        (d / "launch.json").write_text(json.dumps({"mock_llm": False, "session": session}))
        (d / "config.json").write_text(json.dumps({"budget_usd": 70.0}))
        rows = list(ledger)
        if resume_spent is not None:
            rows.insert(0, {"type": "resume", "cost": 0.0, "spent_usd": resume_spent})
            (d / "resume.json").write_text("{}")
        (d / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        if results is not None:
            (d / "results.json").write_text(json.dumps(
                {"resources": {"spent_usd": results}, "performance": {}, "flags": []}))
        if checkpoint:
            (d / "checkpoint.json").write_text("{}")

    # a foreign session's finished run and a foreign dead launch: never counted
    run("w10full-other-L001-s0", session=None, results=45.0)
    run("w10full-other-L002-s0", session="elsewhere", live=True)
    # this session: stage 1 paused ($10 own), stage 2 finished — its results
    # carry the chain's total ($10 + $12) and its ledger the parent's rows
    run("s1", session="me", checkpoint=True, ledger=[{"type": "llm", "cost": 10.0}])
    run("s2", session="me", results=22.0, resume_spent=10.0,
        ledger=[{"type": "llm", "cost": 12.0}])
    # a live resumed seed: reserves the wallet minus what the chain spent at the cut
    run("s3", session="me", resume_spent=10.0, live=True)
    l = Launcher({"wx": base}, root, REPO_ROOT, session_name="me", max_spend_usd=100.0)
    acct = l.budget()
    assert {r["run_id"]: r["spent_usd"] for r in acct["finished"]} == {"s1": 10.0, "s2": 12.0}
    assert acct["live"] == [{"run_id": "s3", "reserved_usd": 60.0}]
    assert acct["committed_usd"] == 82.0 and acct["remaining_usd"] == 18.0
    # without a session name the root's every launch counts (the offline session)
    acct_all = Launcher({"wx": base}, root, REPO_ROOT).budget()
    assert len(acct_all["finished"]) == 3 and len(acct_all["live"]) == 2
