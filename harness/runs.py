"""Run inventory and kill switch that needs no launcher.

    python -m harness.runs ps   [--root DIR ...] [--all]
    python -m harness.runs kill <run_id | Lnnn | substring | /regex/> ... [--dry-run]
    python -m harness.runs stop <run_id | ...>                          [--dry-run]

`ps` joins the machine's process table (every `harness.serve` and every
`runtime/actor.py run`, keyed by the run dir on their command lines) with
the run dirs on disk (`launch.json`, `run.json`, `results.json`,
`killed.json`, the ledger) and prints one row per run with a state:

    running    server alive, actor alive
    idle       server alive, no actor (a launch without a candidate, or the
               actor died) — the sim waits forever and reserves its wallet
    orphan     a process alive whose run dir no longer exists
    finished   results.json present
    paused     checkpoint.json present (a stage of an online-update chain,
               harness/checkpoint.py; resumable)
    killed     killed.json present
    dead       nothing alive, no results, no checkpoint, no killed.json
               (crashed or lost)

`kill` SIGTERMs the actor's process group, SIGKILLs the server, and writes
`killed.json` (ledger spend) when the run dir exists and has no
results.json — exactly what the launcher's DELETE does, from the shell,
for runs no launcher remembers. `stop` touches `<run_dir>/stop` instead:
the server notices within seconds, writes killed.json itself and exits;
the actor fails on its next call. Patterns match the run id: exact id,
a launch number (`L004`), a substring, or `/regex/`.

Default roots for `ps`: every `runs/task_*/<model>/` under the repo, plus
the run dirs the process table names. With an explicit `--root`, `kill`
and `stop` act only on runs under it. A pid is acted on only while `ps`
still shows that run id on it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KILL_SETTLE_S = 5.0
_RUN_DIR_RE = re.compile(r"--run-dir\s+(\S+)")
_OUT_RE = re.compile(r"--out\s+(\S+)")
_WORKSPACE_RE = re.compile(r"--workspace\s+(\S+)")
_RUN_ID_RE = re.compile(r"--run-id\s+(\S+)")
_MODEL_RE = re.compile(r"--model\s+(\S+)")
_L_RE = re.compile(r"-L(\d{3})(?:-|$)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- process table ------------------------------------------------------------------------


def ps_lines() -> list[str]:
    out = subprocess.run(["ps", "-ww", "-axo", "pid=,pgid=,command="],
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def parse_ps(lines: list[str]) -> list[dict]:
    """Rows {pid, pgid, role, run_dir, run_id, cmd} for every serve / actor
    process; other processes are ignored."""
    rows = []
    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pgid, cmd = int(parts[0]), int(parts[1]), parts[2]
        if "harness.serve" in cmd and "-m harness.serve" in cmd:
            m = _RUN_DIR_RE.search(cmd)
            run_dir = Path(m.group(1)) if m else None
            rows.append({"pid": pid, "pgid": pgid, "role": "server",
                         "run_dir": run_dir,
                         "run_id": run_dir.name if run_dir else
                         (_RUN_ID_RE.search(cmd).group(1) if _RUN_ID_RE.search(cmd) else None),
                         "model": (_MODEL_RE.search(cmd).group(1)
                                   if _MODEL_RE.search(cmd) else None),
                         "cmd": cmd})
        elif "actor.py" in cmd and " run" in cmd and "harness.runs" not in cmd:
            m = _OUT_RE.search(cmd)
            if m:
                run_dir = Path(m.group(1))
            else:
                w = _WORKSPACE_RE.search(cmd)
                run_dir = Path(w.group(1)).parent if w else None
            rows.append({"pid": pid, "pgid": pgid, "role": "actor",
                         "run_dir": run_dir,
                         "run_id": run_dir.name if run_dir else None,
                         "model": None, "cmd": cmd})
    return rows


def pid_runs(pid: int | None, run_id: str) -> bool:
    """Alive and still this run's process (command line names the id)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    cmd = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                         capture_output=True, text=True).stdout
    return run_id in cmd


# -- inventory ------------------------------------------------------------------------


def default_roots() -> list[Path]:
    runs = REPO_ROOT / "runs"
    if not runs.exists():
        return []
    roots = []
    for task in sorted(runs.glob("task_*")):
        roots += [d for d in sorted(task.iterdir()) if d.is_dir()]
    roots += [d for d in sorted(runs.glob("improver/**/heldout")) if d.is_dir()]
    return roots


def _ledger_tail(run_dir: Path) -> tuple[str | None, float]:
    """(sim_time of the last ledger row, total booked cost)."""
    p = run_dir / "ledger.jsonl"
    if not p.is_file():
        return None, 0.0
    total, last = 0.0, None
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                total += float(e.get("cost") or 0.0)
                last = e.get("sim_time") or last
    except OSError:
        pass
    return last, total


def inventory(roots: list[Path] | None = None, procs: list[dict] | None = None,
              include_finished: bool = False) -> list[dict]:
    """One row per run: disk state joined with live processes."""
    procs = parse_ps(ps_lines()) if procs is None else procs
    roots = default_roots() if roots is None else [Path(r).resolve() for r in roots]
    by_dir: dict[Path, dict] = {}

    def row_for(run_dir: Path) -> dict:
        run_dir = Path(run_dir)
        if run_dir not in by_dir:
            by_dir[run_dir] = {"run_id": run_dir.name, "run_dir": run_dir,
                               "dir_exists": run_dir.is_dir(),
                               "server_pid": None, "server_pgid": None,
                               "actor_pid": None, "actor_pgid": None,
                               "model": None, "launch": None}
        return by_dir[run_dir]

    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and ((d / "launch.json").is_file() or (d / "run.json").is_file()):
                r = row_for(d.resolve())
                try:
                    r["launch"] = json.loads((d / "launch.json").read_text())
                except (OSError, ValueError):
                    pass
    for p in procs:
        if p["run_dir"] is None:
            continue
        r = row_for(p["run_dir"].resolve() if p["run_dir"].exists() else p["run_dir"])
        r[f"{p['role']}_pid"] = p["pid"]
        r[f"{p['role']}_pgid"] = p["pgid"]
        r["model"] = r["model"] or p.get("model")

    rows = []
    for r in by_dir.values():
        d = r["run_dir"]
        r["results"] = (d / "results.json").is_file()
        r["paused"] = (d / "checkpoint.json").is_file()
        r["killed"] = (d / "killed.json").is_file()
        r["sim_time"], r["spent_usd"] = _ledger_tail(d) if r["dir_exists"] else (None, 0.0)
        if r["launch"]:
            r["model"] = r["model"] or r["launch"].get("model")
            r["candidate"] = (Path(r["launch"]["candidate"]).name
                              if r["launch"].get("candidate") else None)
        else:
            r["candidate"] = None
        alive = r["server_pid"] is not None or r["actor_pid"] is not None
        if alive and not r["dir_exists"]:
            r["state"] = "orphan"
        elif r["server_pid"] is not None and r["actor_pid"] is not None:
            r["state"] = "running"
        elif r["server_pid"] is not None:
            r["state"] = "idle"
        elif r["actor_pid"] is not None:
            r["state"] = "actor-only"
        elif r["results"]:
            r["state"] = "finished"
        elif r["paused"]:
            r["state"] = "paused"
        elif r["killed"]:
            r["state"] = "killed"
        else:
            r["state"] = "dead"
        if include_finished or r["state"] not in ("finished", "paused", "killed"):
            rows.append(r)
    rows.sort(key=lambda r: (r["state"], r["run_id"]))
    return rows


def format_table(rows: list[dict]) -> str:
    head = ["state", "run_id", "server", "actor", "sim_time", "spent", "dir"]
    body = [[r["state"], r["run_id"], str(r["server_pid"] or "-"),
             str(r["actor_pid"] or "-"), (r["sim_time"] or "-")[:16],
             f"{r['spent_usd']:.2f}", "ok" if r["dir_exists"] else "GONE"]
            for r in rows]
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return "\n".join(fmt.format(*x) for x in [head] + body) if body else "(no runs)"


# -- selection ----------------------------------------------------------------------------


def matches(run_id: str, pattern: str) -> bool:
    if pattern == run_id:
        return True
    if len(pattern) > 2 and pattern[0] == pattern[-1] == "/":
        return re.search(pattern[1:-1], run_id) is not None
    if re.fullmatch(r"L\d{3}", pattern):
        return f"-{pattern}-" in run_id or run_id.endswith(f"-{pattern}")
    return pattern in run_id


def select(rows: list[dict], patterns: list[str]) -> list[dict]:
    return [r for r in rows if any(matches(r["run_id"], p) for p in patterns)]


# -- actions ------------------------------------------------------------------------------


def write_killed(run_dir: Path, run_id: str, *, actor_killed: bool,
                 server_killed: bool, reason: str = "harness.runs kill") -> float | None:
    if not run_dir.is_dir() or (run_dir / "results.json").exists() \
            or (run_dir / "checkpoint.json").exists() \
            or (run_dir / "killed.json").exists():
        return None
    _, spent = _ledger_tail(run_dir)
    (run_dir / "killed.json").write_text(json.dumps(
        {"run_id": run_id, "killed_at": _now(), "reason": reason,
         "actor_killed": actor_killed, "server_killed": server_killed,
         "spent_usd": round(spent, 4)}, indent=1))
    return spent


def kill_run(row: dict, dry_run: bool = False) -> dict:
    """SIGTERM the actor group, SIGKILL the server, mark the dir."""
    rid = row["run_id"]
    out = {"run_id": rid, "actor_killed": False, "server_killed": False,
           "killed_json": False}
    if row.get("actor_pid") and pid_runs(row["actor_pid"], rid):
        out["actor_killed"] = True
        if not dry_run:
            try:
                os.killpg(row.get("actor_pgid") or row["actor_pid"], signal.SIGTERM)
            except OSError:
                os.kill(row["actor_pid"], signal.SIGTERM)
    if row.get("server_pid") and pid_runs(row["server_pid"], rid):
        out["server_killed"] = True
        if not dry_run:
            deadline = time.monotonic() + KILL_SETTLE_S
            while out["actor_killed"] and pid_runs(row["actor_pid"], rid) \
                    and time.monotonic() < deadline:
                time.sleep(0.1)
            try:
                os.kill(row["server_pid"], signal.SIGKILL)
            except OSError:
                pass
    if not dry_run:
        deadline = time.monotonic() + KILL_SETTLE_S
        while time.monotonic() < deadline and (
                pid_runs(row.get("server_pid"), rid) or pid_runs(row.get("actor_pid"), rid)):
            time.sleep(0.1)
        spent = write_killed(Path(row["run_dir"]), rid,
                             actor_killed=out["actor_killed"],
                             server_killed=out["server_killed"])
        out["killed_json"] = spent is not None
        out["spent_usd"] = round(spent, 4) if spent is not None else None
    return out


def stop_run(row: dict, dry_run: bool = False) -> dict:
    d = Path(row["run_dir"])
    if not d.is_dir():
        return {"run_id": row["run_id"], "stopped": False,
                "error": "run dir gone — use kill"}
    if not dry_run:
        (d / "stop").touch()
    return {"run_id": row["run_id"], "stopped": True, "marker": str(d / "stop")}


def _under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root)
        return True
    except ValueError:
        return False


# -- CLI ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run inventory / kill switch (no launcher needed).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ps = sub.add_parser("ps", help="list runs: processes joined with run dirs")
    p_ps.add_argument("--root", action="append", default=None,
                      help="run roots to scan (default: every runs/task_*/<model>/)")
    p_ps.add_argument("--all", action="store_true", help="include finished / killed runs")
    p_ps.add_argument("--json", action="store_true")
    for name, help_ in (("kill", "SIGTERM actor group, SIGKILL server, write killed.json"),
                        ("stop", "touch <run_dir>/stop: the server stops itself")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("patterns", nargs="+",
                        help="run id, Lnnn, substring, or /regex/")
        sp.add_argument("--root", action="append", default=None)
        sp.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "ps":
        rows = inventory(args.root, include_finished=args.all)
        if args.json:
            print(json.dumps([{k: (str(v) if isinstance(v, Path) else v)
                               for k, v in r.items() if k != "launch"} for r in rows],
                             indent=1))
        else:
            print(format_table(rows))
        return 0

    rows = select(inventory(args.root, include_finished=True), args.patterns)
    if args.root:  # an explicit root also FENCES kill/stop to runs under it
        roots = [Path(r).resolve() for r in args.root]
        rows = [r for r in rows if any(_under(Path(r["run_dir"]), root) for root in roots)]
    if not rows:
        print("no run matches", args.patterns, file=sys.stderr)
        return 1
    live = [r for r in rows if r["state"] in ("running", "idle", "orphan", "actor-only")]
    if not live:
        print("matched", len(rows), "run(s), none alive:",
              ", ".join(f"{r['run_id']} ({r['state']})" for r in rows), file=sys.stderr)
        return 1
    for r in live:
        res = kill_run(r, args.dry_run) if args.cmd == "kill" else stop_run(r, args.dry_run)
        print(("DRY " if args.dry_run else "") + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
