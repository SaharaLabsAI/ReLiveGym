"""The launcher: a small repo-side service that hands out sims and, when
asked, runs an actor on them. What an external program's session talks to — over HTTP only.

    python -m harness.launcher --session sessions/s01 [--port 8765]
    python -m harness.launcher --bases smoke=tasks/x/configs/smoke.yaml ...

| route                                   | meaning                                        |
|-----------------------------------------|------------------------------------------------|
| GET  /bases                             | registered base ids (held-out ones marked)      |
| GET  /budget                            | spend accounting: finished spend, live wallets, |
|                                         | remaining under the cap, held-out runs used     |
| POST /runs {base, seeds?, candidate?,   | one `harness.serve` per seed from the base yaml;|
|   stretch?, mock_llm?, model?,          | with `candidate` the directory is copied to     |
|   pause_at?, resume_from?}              | <run_dir>/workspace and an actor is started on  |
|                                         | it; returns the run.json handles. pause_at /    |
|                                         | resume_from: a stage of an online-update chain  |
|                                         | (harness/checkpoint.py) — the candidate is then |
|                                         | a whole workspace snapshot, copied verbatim     |
| POST /validate {base, candidate,        | the same on a HELD-OUT base: run dirs under the |
|   seeds?, mock_llm?, model?}            | held-out root, results trimmed, traces hidden   |
| GET  /runs                              | every run this launcher spawned                 |
| GET  /runs/<id>/status                  | server /status + process liveness               |
| GET  /runs/<id>/results                 | results.json (once done); held-out: metrics only|
| GET  /runs/<id>/ledger                  | ledger.jsonl (text); 403 for held-out runs      |
| GET  /runs/<id>/outcomes                | settled `outcome` events; 403 for held-out runs |
| DELETE /runs/<id>                       | kill the actor and the server                   |

Isolation: one `serve` process = one run = one Sim = one clock, ephemeral
loopback port, random token; the actor holds one token. N launches are N
independent processes — nothing mutable is shared, so clocks cannot
interfere. The launch counter is derived from the run dirs on disk, so a
restarted launcher never reuses an id.

Runs outlive launchers: `launch.json` records the server's
and the actor's pid + process group, and a launcher adopts every
`launch.json` under its roots at start (and on first mention of an
unknown id), so `/runs`, `/status` and `DELETE` work across launcher
restarts — a pid is trusted only while `ps` still shows that run id on
it. `python -m harness.runs ps|kill|stop` does the same from the shell
with no launcher at all.

Held-out bases (registry `heldout: true`): reachable only through
`/validate`; their run dirs (server ledger AND actor workspace) live under
the held-out root outside the searcher's directory; `/results` returns
`{performance, resources, flags}` only; the same summary is written to
`<session_dir>/validation/<run_id>.json` when the run finishes.

Guards before any spawn: the candidate must lie inside the session dir;
a leak audit (registry `forbidden_tokens`) refuses candidates containing
ground-truth strings; a session spend cap (`max_spend_usd`, committed =
finished runs' spend + live runs' wallets) refuses launches that would
exceed it; `max_validation_runs` caps held-out seeds per session.

Registry (`<session>.launcher.json`, written beside — OUTSIDE — the session dir):

    {"session": "s01", "task": "…", "session_dir": "…",
     "bases": {"<id>": {"yaml": "…", "heldout": false}, …},
     "runs_root": "…", "heldout_root": "…",
     "max_spend_usd": 300.0, "max_validation_runs": 15,
     "forbidden_tokens": "<path>.forbidden.json"}

The older flat form `{"bases": {"<id>": "<yaml path>"}}` still loads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from harness.model_costs import model_slug, parse_model_spec

READY_TIMEOUT_S = 600.0  # a task's data can take a while to load
RESULTS_WAIT_S = 900.0  # after the actor exits, how long results.json may take
_L_RE = re.compile(r"-L(\d{3,})(?:-|$)")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
                  ".toml", ".cfg", ".ini", ".csv", ".html", ".sh", ""}
_AUDIT_MAX_BYTES = 8 << 20


class HeldOutError(PermissionError):
    """A read or launch the held-out protocol forbids."""


class SpendCapError(Exception):
    """The session spend cap would be exceeded."""


class LeakError(ValueError):
    """The candidate contains forbidden ground-truth strings."""


def _slug(s: str) -> str:
    return _SLUG_RE.sub("-", s).strip("-") or "cand"


# -- leak audit ----------------------------------------------------------------------------


def leak_audit(candidate: Path, forbidden: dict) -> dict:
    """Scan every text file of `candidate` for the forbidden strings.
    Returns {"refuse": [{file, line, token}], "warn": [...]}; a non-empty
    `refuse` list blocks the launch. A floor, not the bar. `skip_dirs`
    (top-level directory names) are left out of the scan — an online-update
    snapshot's `logs/` are the run's own records, not the improver's
    writing."""
    refuse = list(forbidden.get("refuse") or [])
    warn = list(forbidden.get("warn") or [])
    skip = set(forbidden.get("skip_dirs") or ())
    hits: dict[str, list[dict]] = {"refuse": [], "warn": []}
    if not refuse and not warn:
        return hits
    for p in sorted(candidate.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if skip and p.relative_to(candidate).parts[0] in skip:
            continue
        if p.suffix.lower() not in _TEXT_SUFFIXES or p.stat().st_size > _AUDIT_MAX_BYTES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(p.relative_to(candidate))
        lines = None
        for kind, toks in (("refuse", refuse), ("warn", warn)):
            for tok in toks:
                if tok and tok in text:
                    if lines is None:
                        lines = text.splitlines()
                    line = next((i for i, l in enumerate(lines, 1) if tok in l), 0)
                    hits[kind].append({"file": rel, "line": line, "token": tok})
    return hits


# -- the launcher ------------------------------------------------------------------------------


class Launcher:
    def __init__(self, bases: dict, runs_root: Path, repo_root: Path,
                 python_exe: str = sys.executable, *,
                 heldout: set[str] | frozenset[str] | None = None,
                 session_dir: Path | None = None,
                 heldout_root: Path | None = None,
                 max_spend_usd: float | None = None,
                 max_validation_runs: int | None = None,
                 forbidden: dict | None = None,
                 default_model: str | None = None,
                 session_name: str | None = None):
        # api_provider:model applied to every launch that names none (the
        # bases carry no model: it is a launch-time argument repo-wide)
        self.default_model = default_model
        # when set, budget() counts only runs whose launch.json names this
        # session — an online-update session shares its runs root with
        # every other run of the model ; the offline session's root holds its runs alone
        self.session_name = session_name
        self.bases: dict[str, Path] = {}
        self.heldout: set[str] = set(heldout or ())
        for k, v in bases.items():
            if isinstance(v, dict):
                self.bases[k] = Path(v["yaml"]).resolve()
                if v.get("heldout"):
                    self.heldout.add(k)
            else:
                self.bases[k] = Path(v).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.python_exe = python_exe
        self.session_dir = Path(session_dir).resolve() if session_dir else None
        self.heldout_root = (Path(heldout_root).resolve() if heldout_root
                             else self.runs_root / "heldout")
        self.max_spend_usd = max_spend_usd
        self.max_validation_runs = max_validation_runs
        self.forbidden = forbidden or {}
        self.runs: dict[str, dict] = {}  # run_id -> {handle, proc, actor?, run_dir, ...}
        self._lock = threading.Lock()
        self._launch_no = 0
        self._backfill_validation_summaries()
        self.adopt_from_disk()

    # -- spawn -----------------------------------------------------------------------

    def launch(self, base: str, seeds, stretch: str | None,
               mock_llm: bool, model: str | None = None,
               candidate: str | Path | None = None,
               run_root: str | Path | None = None,
               validate: bool = False,
               pause_at: str | None = None,
               resume_from: str | Path | None = None,
               forbidden: dict | None = None,
               extra: dict | None = None) -> list[dict]:
        """Spawn one server (+ actor) per seed. `pause_at` / `resume_from`
        make the run a stage of an online-update chain: the server pauses at the
        instant / continues the paused run dir, and the candidate — a
        whole workspace snapshot — is copied verbatim. `forbidden`
        overrides the registry's leak-audit tokens for this launch (the
        online session audits per stage). `extra` is recorded in
        launch.json (e.g. the stage number)."""
        if base not in self.bases:
            raise KeyError(f"unknown base {base!r} (known: {sorted(self.bases)})")
        heldout = base in self.heldout
        if heldout and not validate:
            raise HeldOutError(f"base {base!r} is held out: launch it through "
                               "POST /validate (traces stay hidden)")
        if validate and not heldout:
            raise ValueError(f"base {base!r} is a search base; /validate is for "
                             "held-out bases only")
        if validate and not candidate:
            raise ValueError("/validate needs a candidate directory")
        model = model or self.default_model
        if seeds is None:
            seed_list: list[int | None] = [None]
        elif isinstance(seeds, int):
            seed_list = list(range(seeds))
        else:
            seed_list = [int(s) for s in seeds]

        cand_dir = (self._check_candidate(candidate, forbidden=forbidden)
                    if candidate else None)
        if run_root is not None:
            raise ValueError("run_root is not accepted: search runs land under "
                             "the session's runs root, held-out runs under the "
                             "held-out root")
        parent = self._check_resume_from(resume_from) if resume_from else None
        if parent is not None and stretch:
            raise ValueError("stretch and resume_from do not combine")
        root = self.heldout_root if heldout else self.runs_root
        self._check_caps(base, len(seed_list), heldout, mock_llm)

        # the run dir mirrors the run_id serve derives from --run-id:
        # base[-cand]-L<n>-<model>[-s<seed>] — apply_launch_model suffixes
        # the model slug whether it came from --model or the base yaml
        from harness.config import load_config

        base_model = load_config(str(self.bases[base])).agent.model
        mname = parse_model_spec(model)[1] if model else base_model
        mslug = f"-{model_slug(mname)}" if mname else ""
        cand_name = _slug(cand_dir.name) if cand_dir else None
        with self._lock:
            n = self._next_launch_no([root])
            self._launch_no = n
        stem = f"{base}-{cand_name}-L{n:03d}" if cand_name else f"{base}-L{n:03d}"
        handles = []
        for seed in seed_list:
            run_id = stem + mslug + (f"-s{seed}" if seed is not None else "")
            run_dir = root / run_id
            cmd = [self.python_exe, "-m", "harness.serve",
                   "--config", str(self.bases[base]),
                   "--run-dir", str(run_dir), "--repo-root", str(self.repo_root),
                   "--run-id", stem]
            if model:
                cmd += ["--model", model]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if stretch:
                cmd += ["--stretch", str(stretch)]
            if mock_llm:
                cmd += ["--mock-llm"]
            if pause_at:
                cmd += ["--pause-at", str(pause_at)]
            if parent is not None:
                cmd += ["--resume-from", str(parent)]
            root.mkdir(parents=True, exist_ok=True)
            log = open(root / f"{run_id}.serve.log", "ab")
            proc = subprocess.Popen(cmd, cwd=str(self.repo_root),
                                    stdout=subprocess.PIPE, stderr=log,
                                    start_new_session=True)
            handle = self._wait_ready(proc, run_id)
            if seed is None:  # the effective seed is the config's (serve wrote it)
                try:
                    seed = json.loads((run_dir / "config.json").read_text()).get("seed")
                except (OSError, ValueError):
                    pass
            rec = {"handle": handle, "proc": proc, "actor": None,
                   "run_dir": run_dir, "base": base, "seed": seed,
                   "stretch": stretch, "mock_llm": mock_llm, "model": model,
                   "candidate": str(cand_dir) if cand_dir else None,
                   "heldout": heldout, "launched_at": _now(),
                   "pause_at": pause_at,
                   "resume_from": str(parent) if parent is not None else None,
                   **(extra or {})}
            self._write_launch(run_id, rec)
            if cand_dir:
                rec["actor"] = self._spawn_actor(run_dir, cand_dir, handle,
                                                 whole=parent is not None)
                handle = {**handle, "actor_pid": rec["actor"].pid}
                rec["handle"] = handle
                self._write_launch(run_id, rec)  # now with the actor's pids
            if heldout:
                # the searcher sees a trimmed handle: no url/token to poke at
                handle = {k: handle[k] for k in ("run_id", "sim_start", "sim_end")}
                handle["heldout"] = True
            self.runs[run_id] = rec
            if heldout:
                threading.Thread(target=self._watch_heldout, args=(run_id,),
                                 daemon=True).start()
            handles.append(handle)
        return handles

    def _write_launch(self, run_id: str, rec: dict) -> None:
        """launch.json: the run's bookkeeping PLUS the pids/process groups
        of its server and actor, so a later launcher (or harness.runs) can
        stop it without a live Popen handle."""
        data = {k: v for k, v in rec.items()
                if k not in ("handle", "proc", "actor", "adopted")}
        data["run_id"] = run_id
        data["server_pid"] = rec["proc"].pid
        data["server_pgid"] = _pgid_of(rec["proc"])
        if rec.get("actor") is not None:
            data["actor_pid"] = rec["actor"].pid
            data["actor_pgid"] = _pgid_of(rec["actor"])
        (rec["run_dir"] / "launch.json").write_text(
            json.dumps(data, indent=1, default=str))

    # -- adoption (runs started by an earlier launcher) --------------------------------

    def adopt_from_disk(self) -> list[str]:
        """Take over every run under the roots that carries a launch.json
        and is not already known: pids come from launch.json, the handle
        from run.json. Returns the adopted ids."""
        adopted = []
        for root in {self.runs_root, self.heldout_root}:
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and d.name not in self.runs \
                        and (d / "launch.json").is_file():
                    if self._adopt(d) is not None:
                        adopted.append(d.name)
        return adopted

    def _adopt(self, run_dir: Path) -> dict | None:
        try:
            info = json.loads((run_dir / "launch.json").read_text())
        except (OSError, ValueError):
            return None
        run_id = info.get("run_id") or run_dir.name
        handle = None
        if (run_dir / "run.json").is_file():
            try:
                handle = json.loads((run_dir / "run.json").read_text())
            except (OSError, ValueError):
                handle = None
        proc = _PidHandle(info.get("server_pid"), info.get("server_pgid"), run_id)
        actor = (_PidHandle(info["actor_pid"], info.get("actor_pgid"), run_id)
                 if info.get("actor_pid") else None)
        rec = {"handle": handle, "proc": proc, "actor": actor,
               "run_dir": run_dir, "base": info.get("base"),
               "seed": info.get("seed"), "stretch": info.get("stretch"),
               "mock_llm": bool(info.get("mock_llm")), "model": info.get("model"),
               "candidate": info.get("candidate"),
               "heldout": bool(info.get("heldout")),
               "launched_at": info.get("launched_at"), "adopted": True}
        self.runs[run_id] = rec
        return rec

    def _check_candidate(self, candidate, forbidden: dict | None = None) -> Path:
        cand = Path(candidate)
        if not cand.is_absolute() and self.session_dir:
            cand = self.session_dir / cand
        cand = cand.resolve()
        if not (cand / "main.py").is_file():
            raise ValueError(f"candidate {cand} has no main.py")
        if not (cand / "runtime" / "actor.py").is_file():
            raise ValueError(f"candidate {cand} has no runtime/actor.py "
                             "(a candidate is a complete program: main.py + "
                             "modules + runtime/)")
        if self.session_dir and not _inside(cand, self.session_dir):
            raise ValueError(f"candidate must lie inside the session dir "
                             f"{self.session_dir}")
        hits = leak_audit(cand, self.forbidden if forbidden is None else forbidden)
        if hits["refuse"]:
            raise LeakError("leak audit: candidate contains forbidden "
                            "ground-truth strings: "
                            + json.dumps(hits["refuse"][:20]))
        self._last_audit = hits
        return cand

    def _check_resume_from(self, resume_from) -> Path:
        """A paused run dir to continue (harness/checkpoint.py). Under a
        session it must lie under this launcher's roots: a searcher can
        never resume a held-out run or a run of another session."""
        from harness.checkpoint import checkpoint_of

        parent = Path(resume_from)
        if not parent.is_absolute() and self.session_dir:
            parent = self.session_dir / parent
        parent = parent.resolve()
        checkpoint_of(parent)  # raises when it is not a paused run
        if self.session_dir and not _inside(parent, self.runs_root):
            raise ValueError(f"resume_from must lie under the session's runs "
                             f"root {self.runs_root}")
        return parent

    def _check_caps(self, base: str, n: int, heldout: bool, mock_llm: bool) -> None:
        if mock_llm:
            return  # a mocked-LLM run spends nothing
        if heldout and self.max_validation_runs is not None:
            done = sum(1 for p in self.heldout_root.glob("*") if p.is_dir())
            if done + n > self.max_validation_runs:
                raise SpendCapError(
                    f"held-out cap: {done} validation runs exist, {n} more "
                    f"would exceed max_validation_runs={self.max_validation_runs}")
        if self.max_spend_usd is not None:
            from harness.config import load_config

            budget = load_config(str(self.bases[base])).budget_usd
            acct = self.budget()
            if acct["committed_usd"] + n * budget > self.max_spend_usd:
                raise SpendCapError(
                    f"spend cap: ${acct['committed_usd']:.2f} committed "
                    f"(${acct['finished_spent_usd']:.2f} spent by "
                    f"{len(acct['finished'])} finished runs + "
                    f"${acct['live_reserved_usd']:.2f} reserved by "
                    f"{len(acct['live'])} live runs) + {n} x ${budget:g} > "
                    f"max_spend_usd ${self.max_spend_usd:g}; remaining "
                    f"${acct['remaining_usd']:.2f} — GET /budget for the rows")

    def budget(self) -> dict:
        """The session's spend accounting: over runs THIS launcher (or an
        earlier launcher of the same session) started — the dirs carrying a
        launch.json (naming this session, when the launcher has a
        session_name) — finished runs count what they spent, stopped runs
        (killed.json) and paused runs (checkpoint.json) what their ledger
        booked, unfinished runs reserve what their wallet can still spend;
        mocked runs count nothing. A resumed run's ledger starts with its
        parent's rows (harness/checkpoint.py): only the rows after its
        `resume` row are its own spend, and its reservation is the wallet
        minus what the chain had spent at the cut. Copies of past runs in
        the workspace carry no launch.json and never count."""
        finished, live = [], []
        for root in {self.runs_root, self.heldout_root}:
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                launch = d / "launch.json"
                if not d.is_dir() or not launch.exists():
                    continue
                try:
                    info = json.loads(launch.read_text())
                    if info.get("mock_llm"):
                        continue
                    if self.session_name is not None \
                            and info.get("session") != self.session_name:
                        continue
                    res, cfg = d / "results.json", d / "config.json"
                    inherited = _resume_spend(d)
                    if res.exists():
                        usd = float(json.loads(res.read_text())["resources"]["spent_usd"])
                        finished.append({"run_id": d.name,
                                         "spent_usd": round(usd - inherited, 4)})
                    elif (d / "killed.json").exists():  # stopped: book the ledger
                        finished.append({"run_id": d.name, "killed": True,
                                         "spent_usd": round(_ledger_spend(d) - inherited, 4)})
                    elif (d / "checkpoint.json").exists():  # paused: likewise
                        finished.append({"run_id": d.name, "paused": True,
                                         "spent_usd": round(_ledger_spend(d) - inherited, 4)})
                    elif cfg.exists():
                        usd = float(json.loads(cfg.read_text()).get("budget_usd", 0.0))
                        live.append({"run_id": d.name,
                                     "reserved_usd": round(max(usd - inherited, 0.0), 4)})
                except (OSError, ValueError, KeyError):
                    continue
        spent = sum(r["spent_usd"] for r in finished)
        reserved = sum(r["reserved_usd"] for r in live)
        out = {"max_spend_usd": self.max_spend_usd,
               "finished_spent_usd": round(spent, 4),
               "live_reserved_usd": round(reserved, 4),
               "committed_usd": round(spent + reserved, 4),
               "remaining_usd": (round(self.max_spend_usd - spent - reserved, 4)
                                 if self.max_spend_usd is not None else None),
               "finished": finished, "live": live,
               "max_validation_runs": self.max_validation_runs,
               "validation_runs_used": (sum(1 for p in self.heldout_root.glob("*")
                                            if p.is_dir())
                                        if self.heldout_root.exists() else 0)}
        return out

    def committed_usd(self) -> float:
        return self.budget()["committed_usd"]

    def _next_launch_no(self, extra_roots) -> int:
        n = self._launch_no
        for root in {self.runs_root, self.heldout_root, *extra_roots}:
            if root and root.exists():
                for p in root.iterdir():
                    m = _L_RE.search(p.name)
                    if m:
                        n = max(n, int(m.group(1)))
        return n + 1

    def _spawn_actor(self, run_dir: Path, cand_dir: Path, handle: dict,
                     whole: bool = False) -> subprocess.Popen:
        """Copy the candidate to <run_dir>/workspace and start the runner
        on it. A fresh launch drops the run-time state dirs a candidate
        may carry; a resume (`whole`) copies the snapshot verbatim — its
        transcripts, memory and jails ARE the actor's state at the cut."""
        ws = run_dir / "workspace"
        ignore = (("__pycache__", "*.pyc") if whole else
                  ("__pycache__", "*.pyc", "memory", "logs", "agents",
                   "instructions", "state.json"))
        shutil.copytree(cand_dir, ws, ignore=shutil.ignore_patterns(*ignore))
        # the runner is the environment's half of the lifecycle contract, not
        # the candidate's: install the current one over the copy (candidates
        # carry it so `runtime/actor.py check` works offline)
        runner = self.repo_root / "scaffolds" / "runtime" / "actor.py"
        if runner.is_file():
            shutil.copy2(runner, ws / "runtime" / "actor.py")
        log = open(run_dir / "actor.log", "ab")
        return subprocess.Popen(
            [self.python_exe, str(ws / "runtime" / "actor.py"), "run",
             "--workspace", str(ws), "--out", str(run_dir)],
            cwd=str(ws), stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "ENV_URL": handle["env_url"],
                 "ENV_TOKEN": handle["token"],
                 "ENV_MODEL": handle.get("model") or ""},
            start_new_session=True)

    def _wait_ready(self, proc: subprocess.Popen, run_id: str) -> dict:
        """The serve process prints its handle as the first stdout line."""
        assert proc.stdout is not None
        line = None

        def read():
            nonlocal line
            line = proc.stdout.readline()

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(READY_TIMEOUT_S)
        if not line:
            proc.kill()
            raise RuntimeError(f"serve for {run_id} did not announce a handle "
                               f"within {READY_TIMEOUT_S:g}s (see its .serve.log)")
        return json.loads(line)

    # -- held-out summaries ----------------------------------------------------------------

    def _watch_heldout(self, run_id: str) -> None:
        r = self.runs[run_id]
        if r["actor"] is not None:
            r["actor"].wait()
        deadline = time.monotonic() + RESULTS_WAIT_S
        while not (r["run_dir"] / "results.json").exists():
            if time.monotonic() > deadline or r["proc"].poll() is not None:
                break
            time.sleep(1.0)
        self._write_validation_summary(r["run_dir"])

    def _write_validation_summary(self, run_dir: Path) -> dict | None:
        res = run_dir / "results.json"
        if not res.exists() or self.session_dir is None:
            return None
        info = {}
        launch = run_dir / "launch.json"
        if launch.exists():
            info = json.loads(launch.read_text())
        summary = trimmed_results(json.loads(res.read_text()))
        summary.update({"base": info.get("base"), "seed": info.get("seed"),
                        "candidate": (Path(info["candidate"]).name
                                      if info.get("candidate") else None),
                        "finished_at": _now(), "heldout": True})
        out = self.session_dir / "validation"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_dir.name}.json").write_text(json.dumps(summary, indent=1))
        return summary

    def _backfill_validation_summaries(self) -> None:
        """Held-out runs that finished while no launcher was up still get
        their summary written (their traces stay where they are)."""
        if self.session_dir is None or not self.heldout_root.exists():
            return
        for d in self.heldout_root.iterdir():
            if d.is_dir() and (d / "results.json").exists() \
                    and not (self.session_dir / "validation" / f"{d.name}.json").exists():
                self._write_validation_summary(d)

    # -- queries ---------------------------------------------------------------------

    def _run(self, run_id: str) -> dict:
        r = self.runs.get(run_id)
        if r is None:  # started by another launcher since we scanned?
            for root in {self.runs_root, self.heldout_root}:
                d = root / run_id
                if (d / "launch.json").is_file():
                    r = self._adopt(d)
                    break
        if r is None:
            raise KeyError(run_id)
        return r

    def status(self, run_id: str) -> dict:
        r = self._run(run_id)
        alive = r["proc"].poll() is None
        out = {"run_id": run_id, "base": r["base"], "seed": r["seed"],
               "heldout": r["heldout"],
               "candidate": Path(r["candidate"]).name if r["candidate"] else None,
               "server_alive": alive,
               "actor_alive": (r["actor"] is not None and r["actor"].poll() is None),
               "results_written": (r["run_dir"] / "results.json").exists(),
               "paused": (r["run_dir"] / "checkpoint.json").exists(),
               "killed": (r["run_dir"] / "killed.json").exists(),
               "adopted": bool(r.get("adopted")),
               "server_pid": r["proc"].pid,
               "actor_pid": r["actor"].pid if r["actor"] is not None else None}
        if not r["heldout"]:
            out["run_dir"] = str(r["run_dir"])
        if alive and r["handle"]:
            try:
                out["sim"] = _get(r["handle"], "/status")
            except Exception as e:  # server bound but not answering yet
                out["sim_error"] = str(e)
        if out["results_written"]:
            try:
                res = json.loads((r["run_dir"] / "results.json").read_text())
                out["results_summary"] = {
                    "primary": res["performance"]["primary"],
                    "spent_usd": res["resources"]["spent_usd"],
                    "flags": res["flags"]}
            except (OSError, ValueError, KeyError):
                pass
        return out

    def results(self, run_id: str) -> dict:
        r = self._run(run_id)
        p = r["run_dir"] / "results.json"
        if not p.exists():
            raise FileNotFoundError("results.json not written yet")
        res = json.loads(p.read_text())
        return trimmed_results(res) if r["heldout"] else res

    def ledger_text(self, run_id: str) -> str:
        r = self._run(run_id)
        if r["heldout"]:
            raise HeldOutError("held-out run: ledger is not readable")
        p = r["run_dir"] / "ledger.jsonl"
        return p.read_text() if p.exists() else ""

    def outcomes(self, run_id: str) -> list[dict]:
        return [json.loads(l) for l in self.ledger_text(run_id).splitlines()
                if l.strip() and json.loads(l).get("type") == "outcome"]

    def kill(self, run_id: str) -> dict:
        """Stop a run: SIGTERM the actor's process group, SIGKILL the server,
        wait for both to be gone, then leave `killed.json` in the run dir —
        a killed server never writes results.json, and without the marker
        budget() would read the run as live and reserve its wallet forever."""
        r = self._run(run_id)
        killed_actor = killed_server = False
        if r["actor"] is not None and r["actor"].poll() is None:
            try:
                os.killpg(_pgid_of(r["actor"]), signal.SIGTERM)
            except OSError:
                r["actor"].kill()
            killed_actor = True
        if r["proc"].poll() is None:
            r["proc"].kill()
            killed_server = True
        deadline = time.monotonic() + KILL_SETTLE_S
        while time.monotonic() < deadline and (
                r["proc"].poll() is None
                or (r["actor"] is not None and r["actor"].poll() is None)):
            time.sleep(0.05)
        spent = _ledger_spend(r["run_dir"])
        if not (r["run_dir"] / "results.json").exists() \
                and not (r["run_dir"] / "checkpoint.json").exists():
            (r["run_dir"] / "killed.json").write_text(json.dumps(
                {"run_id": run_id, "killed_at": _now(),
                 "actor_killed": killed_actor, "server_killed": killed_server,
                 "spent_usd": round(spent, 4)}, indent=1))
        return {"run_id": run_id, "killed": True, "actor_killed": killed_actor,
                "server_killed": killed_server, "spent_usd": round(spent, 4)}

    def public(self, run_id: str) -> dict:
        r = self._run(run_id)
        out = {"run_id": run_id, "base": r["base"], "seed": r["seed"],
               "stretch": r["stretch"], "mock_llm": r["mock_llm"],
               "model": r["model"], "heldout": r["heldout"],
               "candidate": Path(r["candidate"]).name if r["candidate"] else None,
               "launched_at": r["launched_at"],
               "killed": (r["run_dir"] / "killed.json").exists(),
               "results_written": (r["run_dir"] / "results.json").exists(),
               "paused": (r["run_dir"] / "checkpoint.json").exists(),
               "adopted": bool(r.get("adopted")),
               "server_alive": r["proc"].poll() is None,
               "actor_alive": (r["actor"] is not None
                               and r["actor"].poll() is None)}
        if not r["heldout"]:
            out["handle"] = r["handle"]
            out["run_dir"] = str(r["run_dir"])
        return out


def trimmed_results(res: dict) -> dict:
    """What a held-out run discloses: the metrics, the money, the flags —
    never the task block (per-breakpoint / per-alert / per-entity detail)."""
    return {"run_id": res.get("run_id"),
            "performance": res.get("performance"),
            "resources": res.get("resources"),
            "flags": res.get("flags"),
            "constraints": res.get("constraints")}


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
    except ValueError:
        return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


KILL_SETTLE_S = 5.0  # how long kill() waits for the processes to be gone


def _pgid_of(proc) -> int:
    pg = getattr(proc, "pgid", None)
    if pg:
        return int(pg)
    try:
        return os.getpgid(proc.pid)
    except OSError:
        return proc.pid  # start_new_session: pid == pgid


def pid_runs(pid: int | None, run_id: str) -> bool:
    """Is `pid` alive AND still one of this run's processes (its command
    line names the run id)? Guards against pid reuse after a restart."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:
        cmd = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True  # cannot inspect: trust the pid
    return run_id in cmd


class _PidHandle:
    """A Popen look-alike for a process this launcher did not spawn:
    liveness from the pid table (verified against the run id), kill by
    signal, `wait()` a poll loop. Enough for status(), kill() and the
    held-out watcher."""

    def __init__(self, pid, pgid, run_id: str):
        self.pid = int(pid) if pid else None
        self.pgid = int(pgid) if pgid else self.pid
        self.run_id = run_id

    def poll(self):
        return None if pid_runs(self.pid, self.run_id) else 0

    def kill(self) -> None:
        if self.poll() is None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass

    def wait(self, timeout: float | None = None) -> int:
        deadline = time.monotonic() + (timeout if timeout is not None else 1e9)
        while self.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        return 0


def _resume_spend(run_dir: Path) -> float:
    """What the chain had spent when this run resumed (the `resume`
    ledger row's spent_usd; harness/checkpoint.py) — rows before it are
    the parent's and are booked under the parent. 0 for a fresh run."""
    if not (run_dir / "resume.json").exists():
        return 0.0
    inherited = 0.0
    try:
        with open(run_dir / "ledger.jsonl") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "resume":
                    inherited = float(e.get("spent_usd") or 0.0)
    except OSError:
        pass
    return inherited


def _ledger_spend(run_dir: Path) -> float:
    """What a run's ledger booked so far — the spend of a run that will
    never write results.json."""
    total = 0.0
    try:
        with open(run_dir / "ledger.jsonl") as f:
            for line in f:
                try:
                    total += float(json.loads(line).get("cost") or 0.0)
                except (ValueError, AttributeError):
                    continue
    except OSError:
        pass
    return total


def _get(handle: dict, path: str) -> dict:
    req = urllib.request.Request(
        handle["env_url"] + path,
        headers={"Authorization": "Bearer " + handle["token"]})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# -- HTTP ------------------------------------------------------------------------------------


def make_app(launcher: Launcher) -> FastAPI:
    app = FastAPI(title="experiment-launcher")

    @app.get("/bases")
    async def bases() -> dict:
        return {"bases": sorted(launcher.bases),
                "heldout": sorted(launcher.heldout)}

    @app.get("/budget")
    async def budget() -> dict:
        return launcher.budget()

    async def _launch(request: Request, validate: bool) -> dict:
        try:
            body = await request.json()
            assert isinstance(body, dict) and body.get("base")
        except Exception:
            raise HTTPException(400, "body must be {base, seeds?, candidate?, "
                                     "stretch?, mock_llm?, model?, pause_at?, "
                                     "resume_from?}")
        try:
            handles = launcher.launch(body["base"], body.get("seeds"),
                                      body.get("stretch"),
                                      bool(body.get("mock_llm", False)),
                                      body.get("model"),
                                      candidate=body.get("candidate"),
                                      run_root=body.get("run_root"),
                                      validate=validate,
                                      pause_at=body.get("pause_at"),
                                      resume_from=body.get("resume_from"))
        except KeyError as e:
            raise HTTPException(404, str(e))
        except HeldOutError as e:
            raise HTTPException(403, str(e))
        except SpendCapError as e:
            raise HTTPException(402, str(e))
        except (ValueError, LeakError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"launch failed: {e}")
        out = {"runs": handles}
        audit = getattr(launcher, "_last_audit", None)
        if audit and audit.get("warn"):
            out["warnings"] = audit["warn"]
        return out

    @app.post("/runs")
    async def create(request: Request) -> dict:
        return await _launch(request, validate=False)

    @app.post("/validate")
    async def validate(request: Request) -> dict:
        return await _launch(request, validate=True)

    @app.get("/runs")
    async def list_runs() -> dict:
        return {"runs": [launcher.public(r) for r in launcher.runs]}

    def _wrap(fn, run_id):
        try:
            return fn(run_id)
        except KeyError:
            raise HTTPException(404, f"unknown run {run_id!r}")
        except FileNotFoundError as e:
            raise HTTPException(409, str(e))
        except HeldOutError as e:
            raise HTTPException(403, str(e))

    @app.get("/runs/{run_id}/status")
    async def status(run_id: str) -> dict:
        return _wrap(launcher.status, run_id)

    @app.get("/runs/{run_id}/results")
    async def results(run_id: str) -> dict:
        return _wrap(launcher.results, run_id)

    @app.get("/runs/{run_id}/ledger")
    async def ledger(run_id: str) -> dict:
        return {"ledger_jsonl": _wrap(launcher.ledger_text, run_id)}

    @app.get("/runs/{run_id}/outcomes")
    async def outcomes(run_id: str) -> dict:
        return {"outcomes": _wrap(launcher.outcomes, run_id)}

    @app.delete("/runs/{run_id}")
    async def kill(run_id: str) -> dict:
        return _wrap(launcher.kill, run_id)

    return app


# -- registry ---------------------------------------------------------------------------------


def load_registry(session: str | None, bases_args: list[str]) -> dict:
    """Bases + options from `<session>.launcher.json` (v2 or the flat v1
    form) and/or `--bases id=path` arguments."""
    reg: dict = {"bases": {}, "heldout": set()}
    if session:
        meta = Path(session).resolve()
        meta = meta.parent / f"{meta.name}.launcher.json"
        if not meta.is_file():
            raise FileNotFoundError(f"no {meta} (create the session with "
                                    "a <session>.launcher.json beside the session dir)")
        data = json.loads(meta.read_text())
        for k, v in data["bases"].items():
            if isinstance(v, dict):
                reg["bases"][k] = Path(v["yaml"])
                if v.get("heldout"):
                    reg["heldout"].add(k)
            else:
                reg["bases"][k] = Path(v)
        reg["session_dir"] = Path(data.get("session_dir") or meta.parent / data["session"])
        for k in ("runs_root", "heldout_root", "max_spend_usd",
                  "max_validation_runs", "model", "launcher_url"):
            if data.get(k) is not None:
                reg[k] = data[k]
        if data.get("forbidden_tokens"):
            fp = Path(data["forbidden_tokens"])
            reg["forbidden"] = json.loads(fp.read_text()) if fp.is_file() else {}
    for item in bases_args:
        if "=" in item:
            k, v = item.split("=", 1)
        else:
            k, v = Path(item).stem, item
        reg["bases"][k] = Path(v)
    return reg


def load_bases(session: str | None, bases_args: list[str]) -> dict[str, Path]:
    return load_registry(session, bases_args)["bases"]


DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8765


def resolve_bind(reg: dict, host: str | None, port: int | None) -> tuple[str, int]:
    """Where to listen: explicit --host/--port win; otherwise the registry's
    `launcher_url` (the URL baked into the session's HOWTO at init), so a
    session and its launcher cannot disagree; else the defaults."""
    from urllib.parse import urlsplit

    url = urlsplit(reg.get("launcher_url") or "")
    return (host or url.hostname or DEFAULT_HOST,
            port if port is not None else (url.port or DEFAULT_PORT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sim launcher service.")
    parser.add_argument("--session", default=None,
                        help="session dir; bases + options from "
                             "<session>.launcher.json")
    parser.add_argument("--bases", nargs="*", default=[],
                        help="id=path.yaml (or path.yaml, id = stem)")
    parser.add_argument("--runs-root", default=None,
                        help="where run dirs go (default: the registry's "
                             "runs_root, else runs/launcher)")
    parser.add_argument("--repo-root", "-r", default=".")
    parser.add_argument("--host", default=None,
                        help="default: the registry's launcher_url host, "
                             f"else {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=None,
                        help="default: the registry's launcher_url port, "
                             f"else {DEFAULT_PORT} — one launcher per "
                             "session, one port per launcher")
    args = parser.parse_args(argv)

    import dotenv
    import uvicorn

    repo_root = Path(args.repo_root).resolve()
    dotenv.load_dotenv(repo_root / ".env")  # provider keys for the serve children
    reg = load_registry(args.session, args.bases)
    if not reg["bases"]:
        parser.error("no bases registered (--session or --bases)")
    runs_root = (Path(args.runs_root).resolve() if args.runs_root
                 else Path(reg["runs_root"]).resolve() if reg.get("runs_root")
                 else repo_root / "runs" / "launcher")
    launcher = Launcher(reg["bases"], runs_root, repo_root,
                        heldout=reg["heldout"],
                        session_dir=reg.get("session_dir"),
                        heldout_root=reg.get("heldout_root"),
                        max_spend_usd=reg.get("max_spend_usd"),
                        max_validation_runs=reg.get("max_validation_runs"),
                        forbidden=reg.get("forbidden"),
                        default_model=reg.get("model"))
    host, port = resolve_bind(reg, args.host, args.port)
    print(f"launcher on http://{host}:{port} — bases: "
          f"{sorted(reg['bases'])} (held out: {sorted(reg['heldout'])}); "
          f"model {reg.get('model') or '(from the yaml / request)'}; "
          f"runs under {runs_root}; held-out runs under {launcher.heldout_root}",
          flush=True)
    uvicorn.run(make_app(launcher), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
