#!/usr/bin/env python
"""Batch runner for the web-task episode grid: models x {tmA, tmB, tmD} x seeds, headless
browsers, a fixed number of episodes in flight.

One job = one episode, run start to end by one worker thread:

    prepare   python -m harness.mcp prepare --task .. --tm .. --seed ..
                  --app opencode --model <api_provider:model>      (headless)
    act       tm A/B: sh run_opencode.sh          (one fenced `opencode run`)
              tm C/D: python -m harness.mcp drive (one per firing)
    settle    python -m harness.mcp settle        -> server/results.json

Order: for each model, tmA -> tmB -> tmD, each with all seeds. Configs are
untouched (the smoke cells' window, budget and cadence); only --model and
--seed vary. A finished episode (server/results.json) is skipped unless
--force; an unfinished directory is moved aside as legacy-<ts>-<name>.

    python scripts/run_web_episodes.py --dry-run                  # list only
    nohup python scripts/run_web_episodes.py > runs/web_episodes.log 2>&1 &
    python scripts/run_web_episodes.py --models openai:gpt-5.6-terra \\
        --tms D --seeds 0 --workers 1
    python scripts/run_web_episodes.py --mock-llm --stretch 2d \\
        --models openai:gpt-5.6-luna --seeds 0     # free plumbing check

Concurrency: a job holds a sim server (+2 web hosts), a headless Chrome,
OpenCode and the MCP bridge — about 1 GB and mostly waiting on the LLM.
The default of 6 suits a 16 GB machine; a gpt-5.6-terra episode took
~45 min alone, so the default grid (8 models x 3 x 3 = 72 episodes) is
roughly 9-12 h.

Stopping: SIGINT/SIGTERM to this process kills every live actor and sim
server; those episodes have no results.json and are re-run next time. An
episode exceeding --timeout-hours (status `timeout`) or whose actor exits
non-zero (`rc=<n>`; OpenCode exits 0 even on LLM errors, so this is a
broken launch) is killed the same way, never settled: a hung or broken
actor must not become a result.

Each episode gets its own browser-server port from --port-base up (outside
the OS's ephemeral range): under tm=D that port is free between firings,
and a concurrent episode could otherwise be handed it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The paper's model roster (in table order) -> launch name `api_provider:model`; the bare
# model is the pricing key of configs/model_costs.yaml.
MODELS = {
    "GPT-5.6 luna":          "openai:gpt-5.6-luna",
    "GPT-5.6 terra":         "openai:gpt-5.6-terra",
    "Gemini-3.5-flash-lite": "openrouter:google/gemini-3.5-flash-lite",
    "Gemini-3.5-flash":      "openrouter:google/gemini-3.5-flash",
    "Claude-haiku-4.5":      "openrouter:anthropic/claude-haiku-4.5",
    "Qwen3.7-plus":          "openrouter:qwen/qwen3.7-plus",
    "MiniMax-M3":            "openrouter:minimax/minimax-m3",
    "DeepSeek-V4-flash":     "openrouter:deepseek/deepseek-v4-flash",
}
TASKS = {"edgar_portfolio": "edgar", "broker_ops": "bops"}  # -> run-id stem

_live: dict[str, dict] = {}   # job name -> {"proc": Popen | None, "serve_pid": int | None}
_live_lock = threading.Lock()
_stop = threading.Event()


def slug(model: str) -> str:  # same flattening as harness.model_costs.model_slug
    return model.split(":", 1)[1].replace("/", "-")


def plan(tasks, models, tms, seeds, out: Path, port_base: int) -> list[dict]:
    jobs = []
    for task in tasks:
        for model in models:
            for tm in tms:
                for seed in seeds:
                    run_id = f"{TASKS[task]}-tm{tm}-{slug(model)}"
                    jobs.append(dict(
                        task=task, model=model, tm=tm, seed=seed, run_id=run_id,
                        name=f"{run_id}-s{seed}",
                        out=out / task / slug(model),
                        dir=out / task / slug(model) / f"{run_id}-s{seed}",
                        port=port_base + len(jobs)))
    return jobs


def preflight(jobs: list[dict], mock: bool) -> None:
    import dotenv

    from harness.model_costs import PROVIDER_KEYS, load_model_costs, parse_model_spec

    dotenv.load_dotenv(ROOT / ".env")
    priced = load_model_costs()
    problems = []
    for model in sorted({j["model"] for j in jobs}):
        provider, bare = parse_model_spec(model)
        if bare not in priced:
            problems.append(f"{model}: {bare!r} has no entry in configs/model_costs.yaml")
        if not mock and not os.environ.get(PROVIDER_KEYS[provider]):
            problems.append(f"{model}: {PROVIDER_KEYS[provider]} is not set (env or .env)")
    for task, tm in sorted({(j["task"], j["tm"]) for j in jobs}):
        cells = ROOT / "tasks" / task / "configs" / "cells"
        if not list(cells.rglob(f"*tm{tm}-tlrnnone-signone-algnone.yaml")):
            problems.append(f"{task}: no tm{tm} algnone cell under {cells}")
    fence = ("nc", "sandbox-exec") if sys.platform == "darwin" else ()
    for exe in ("opencode", "node") + fence:
        if shutil.which(exe) is None:
            problems.append(f"`{exe}` is not on PATH")
    for j in jobs:
        with socket.socket() as s:
            # as node binds it: a port in TIME_WAIT (a batch stopped seconds
            # ago) is free, a listening one is not
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", j["port"]))
            except OSError:
                problems.append(f"port {j['port']} ({j['name']}) is taken — "
                                "another batch running? pass --port-base")
    if problems:
        raise SystemExit("ABORT:\n  " + "\n  ".join(problems))


def _kill_group(proc: subprocess.Popen | None) -> None:
    """TERM then KILL the actor's process group; the grace period is for
    the launch script's trap / the driver's `finally` (they stop the
    browser server and the firing)."""
    if proc is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):  # gone / only zombies left
            pass
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _abort(name: str) -> None:
    """Kill a job's actor and its sim server: no results.json, so the
    episode is re-run by the next launch."""
    with _live_lock:
        live = dict(_live.get(name) or {})
    _kill_group(live.get("proc"))
    if live.get("serve_pid"):
        try:
            os.kill(live["serve_pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass


def _act(job: dict, cmd: list[str], log, timeout_s: float) -> str:
    """Run the actor in a process group of its own; 'ok' | 'timeout' |
    'stopped' | 'rc=<n>'."""
    proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    with _live_lock:
        _live[job["name"]]["proc"] = proc
    t0 = time.monotonic()
    while proc.poll() is None:
        if _stop.is_set():
            return "stopped"
        if time.monotonic() - t0 > timeout_s:
            return "timeout"
        time.sleep(1.0)
    return "ok" if proc.returncode == 0 else f"rc={proc.returncode}"


def run_job(job: dict, logs: Path, ts: str, args) -> dict:
    out = dict(name=job["name"], status="stopped", minutes=0.0)
    if _stop.is_set():
        return out
    ep_dir: Path = job["dir"]
    if ep_dir.exists():
        aside = ep_dir.parent / f"legacy-{ts}-{ep_dir.name}"
        shutil.move(str(ep_dir), str(aside))
        print(f"moved aside {ep_dir.name} -> {aside.name}", flush=True)
    job["out"].mkdir(parents=True, exist_ok=True)
    episode = ep_dir / "episode.json"
    mcp = [sys.executable, "-m", "harness.mcp"]
    prepare = mcp + ["prepare", "--task", job["task"], "--tm", job["tm"],
                     "--seed", str(job["seed"]), "--app", "opencode",
                     "--model", job["model"], "--out", str(job["out"]),
                     "--run-id", job["run_id"],
                     "--playwright-port", str(job["port"])]  # headless: the default
    if args.mock_llm:
        prepare.append("--mock-llm")
    if args.stretch:
        prepare += ["--stretch", args.stretch]
    print(f"{datetime.now():%H:%M:%S} launch {job['name']}", flush=True)
    t0 = time.time()
    with _live_lock:
        _live[job["name"]] = {"proc": None, "serve_pid": None}
    try:
        with open(logs / f"{job['name']}.log", "w") as log:
            rc = subprocess.run(prepare, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not episode.exists():
                out["status"] = f"prepare rc={rc}"
                return out
            with _live_lock:
                _live[job["name"]]["serve_pid"] = json.loads(
                    episode.read_text())["serve_pid"]
            act = (["sh", str(ep_dir / "run_opencode.sh")] if job["tm"] in "AB"
                   else mcp + ["drive", "--episode", str(episode)])
            out["status"] = _act(job, act, log, args.timeout_hours * 3600)
            if out["status"] != "ok":
                # timeout, stop, or an actor / driver that died (OpenCode
                # exits 0 even on LLM errors, so non-zero is infrastructure):
                # never settled — burning the rest of the window would turn
                # a broken launch into a result
                _abort(job["name"])
                return out
            settled = subprocess.run(mcp + ["settle", "--episode", str(episode)],
                                     cwd=ROOT, capture_output=True, text=True)
            log.write(settled.stdout + settled.stderr)
            if settled.returncode != 0:
                out["status"] += " settle-failed"
                _abort(job["name"])
                return out
            res = json.loads(settled.stdout)
            out.update(primary=(res.get("primary") or {}).get("value"),
                       spent_usd=res.get("spent_usd"), flags=res.get("flags"),
                       triggers_burned=res.get("triggers_burned"))
            return out
    finally:
        out["minutes"] = round((time.time() - t0) / 60, 1)
        with _live_lock:
            _live.pop(job["name"], None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["edgar_portfolio"], choices=list(TASKS))
    ap.add_argument("--models", nargs="+", default=list(MODELS.values()),
                    help="api_provider:model launch names (default: the paper's 8)")
    ap.add_argument("--tms", nargs="+", default=list("ABD"), choices=list("ABCD"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6, help="episodes in flight")
    ap.add_argument("--timeout-hours", type=float, default=6.0,
                    help="real-time ceiling per episode; beyond it the episode "
                         "is killed and left without results")
    ap.add_argument("--out", default="runs/episodes_grid")
    ap.add_argument("--port-base", type=int, default=42000,
                    help="browser-server ports are port-base + job index")
    ap.add_argument("--mock-llm", action="store_true",
                    help="zero-cost plumbing check (results go under <out>/_mock)")
    ap.add_argument("--stretch", default=None, help="shorten the sim, e.g. 2d")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run episodes that already have results.json")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    if args.mock_llm or args.stretch:
        out = out / "_mock"  # never mistaken for (or skipped as) a real episode
    jobs = plan(args.tasks, args.models, args.tms, args.seeds, out, args.port_base)
    todo = [j for j in jobs
            if args.force or not (j["dir"] / "server" / "results.json").exists()]
    print(f"{len(jobs)} episodes planned ({len(args.models)} models x tm{''.join(args.tms)} "
          f"x {len(args.seeds)} seeds x {len(args.tasks)} task(s)); "
          f"{len(jobs) - len(todo)} already finished; {len(todo)} to launch; "
          f"workers={args.workers}; out={out}")
    for j in jobs:
        state = ("done " if j not in todo else "aside" if j["dir"].exists() else "new  ")
        print(f"  {state} {j['task']:16s} {j['model']:42s} tm{j['tm']} s{j['seed']}  "
              f":{j['port']}  {j['name']}")
    if args.dry_run or not todo:
        return
    preflight(todo, args.mock_llm)

    def on_signal(signum, _frame):
        print(f"\nsignal {signum}: stopping — killing live actors and sim servers",
              flush=True)
        _stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs = out / "_logs" / ts
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, logs, ts, args) for j in todo]
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            with open(logs / "summary.jsonl", "a") as sf:
                sf.write(json.dumps(row) + "\n")
            if row["status"] == "stopped" and not row["minutes"]:
                continue  # never started
            spent = row.get("spent_usd")
            print(f"{datetime.now():%H:%M:%S} finished {row['name']} "
                  f"status={row['status']} primary={row.get('primary')} "
                  f"spent={'$%.2f' % spent if spent is not None else '-'} "
                  f"flags={row.get('flags')} ({row['minutes']} min)", flush=True)
    bad = [(r["name"], r["status"]) for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} episodes, {len(bad)} not ok: {bad}\nlogs: {logs}")


if __name__ == "__main__":
    main()
