"""Serve one sim — the DETACHED server mode.

    python -m harness.serve --config c.yaml --run-dir d [--seed N]
                            [--model api_provider:model_name]
                            [--mock-llm] [--stretch 2d] [--host H --port P]

creates the run dir, starts the env server for that config and writes
`d/run.json = {run_id, env_url, token, model, watchdog_seconds}` — the
handle an actor runner needs (`cd <workspace> && ENV_URL=… ENV_TOKEN=…
ENV_MODEL=… python runtime/actor.py run`). It owns NO workspace: the sim,
the clock, the ledger, the results. The clock does not start until the
first `GET /trigger/next`; the process serves until the run is done
(results.json written) or it is killed. One process = one run = one Sim =
one clock, exactly the run unit of harness.run.

`--mock-llm` selects a canned zero-cost LLM upstream (preflights);
`--stretch 2d` shortens sim_end to sim_start + 2 days.

Pause / resume (harness/checkpoint.py):
`--pause-at <iso>` stops the clock there — the actor sees experiment_over,
the server writes checkpoint.json + results_partial.json instead of
results.json and exits 0. `--resume-from <run dir>` continues a paused
run: same yaml, same model, any seed; the new run dir starts with the
parent's ledger and schedule store, the sim is rebuilt at the pause
instant, and `results.json` (or the next checkpoint) covers the whole
chain. Both may be combined for a middle stage.

Self-termination: the
process polls its run dir every WATCH_INTERVAL_S. If the directory is
gone it exits (code 3) — a server nobody can find must not keep serving
and billing. If `<run_dir>/stop` appears it writes `killed.json` (the
ledger's spend, reason "stop marker") and exits 0 — the launcher-free
way to stop a run: `touch <run_dir>/stop`, or `python -m harness.runs
stop <run_id>`. `run.json` carries the server's pid/pgid for the same
reason.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness import checkpoint
from harness.api import tools_manifest
from harness.config import RunConfig, load_config
from harness.contract import is_external
from harness.model_costs import apply_launch_model
from harness.runtime import Sim
from harness.supervisor import Scheduler
from harness.task import load_task_class
from harness.timeutil import iso, parse_iso
from harness.web import start_hosts, stop_hosts

_STRETCH_RE = re.compile(r"^(\d+)([dh])$")
STOP_MARKER = "stop"      # touch <run_dir>/stop -> the server stops itself
WATCH_INTERVAL_S = 2.0    # how often the run dir / stop marker are checked
EXIT_RUN_DIR_GONE = 3


class RunDirGone(RuntimeError):
    """The run directory disappeared under a live server."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_stretch(s: str) -> timedelta:
    m = _STRETCH_RE.match(s.strip())
    if not m:
        raise ValueError(f"--stretch wants e.g. 2d or 36h, got {s!r}")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(days=n) if unit == "d" else timedelta(hours=n)


def install_mock_llm(sim_end: str | None = None) -> None:
    """A canned zero-token upstream for preflights. Every chat reply is one
    tool call, `sleep` until `sim_end` (an empty string when unknown): an
    agent holding a sleep tool parks once and the run ends in a handful of
    calls; one without it (a cron-fired agent, an authored-wait agent) sees
    an unknown-tool error and stops at its own call guard. Process-private
    (the serve process)."""
    import harness.llm_proxy as llm_proxy

    canned = json.dumps({"tool": "sleep", "args": {"until": sim_end or ""},
                         "thought": "mock LLM: park until the end"})

    async def _mock_upstream(path: str, body: dict):
        model = body.get("model", "mock")
        if path == "chat/completions":
            resp = {"id": "mock", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": canned}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                              "total_tokens": 0}}
        else:
            resp = {"id": "mock", "object": "response", "model": model,
                    "output": [], "usage": {"input_tokens": 0,
                                            "output_tokens": 0}}
        return resp, 0.0

    async def _mock_upstream_stream(path: str, body: dict):
        model = body.get("model", "mock")
        chunks = [
            {"id": "mock", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "finish_reason": None,
                          "delta": {"role": "assistant", "content": canned}}]},
            {"id": "mock", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]},
            {"id": "mock", "object": "chat.completion.chunk", "model": model,
             "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                      "total_tokens": 0}},
        ]

        async def gen():
            for c in chunks:
                yield c
        return gen()

    llm_proxy._upstream = _mock_upstream
    llm_proxy._upstream_stream = _mock_upstream_stream


async def serve(cfg: RunConfig, *, repo_root: Path, run_dir: Path,
                host: str = "127.0.0.1", port: int = 0,
                ready=None, pause_at: datetime | None = None,
                resume_from: Path | None = None) -> dict | None:
    """Serve `cfg` from `run_dir` until the run is done; returns the
    results dict, or None when stopped through the stop marker (then
    `killed.json` is in the run dir instead of results.json) or paused
    at `pause_at` (then `checkpoint.json` + `results_partial.json`).
    `resume_from` continues a paused run (harness/checkpoint.py). Raises
    RunDirGone when the run dir disappears. `ready(handle)` is called
    once the server is bound."""
    from harness.run import start_server

    if cfg.cell.tm == "B" and not is_external(cfg):
        raise ValueError(
            "a constructor-built tm=B cell runs its authored programs server-side (ProgramApp "
            "needs the workspace in this process): use harness.run; only "
            "ext: programs are served detached")
    ck = None
    if resume_from is not None:
        ck = checkpoint.prepare_resume(run_dir, resume_from, cfg)
        if pause_at is not None and pause_at <= parse_iso(ck["paused_at"]):
            raise ValueError(f"pause_at {iso(pause_at)} is not after the "
                             f"parent's pause {ck['paused_at']}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    task = load_task_class(cfg.task_name).from_run_config(cfg, repo_root)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=1, default=str))
    # no workspace here — the actor's is on its side; the path only names
    # where one WOULD be for code that formats it
    sim = Sim(cfg, run_dir, run_dir / "workspace", task, pause_at=pause_at)
    sim.scheduler = Scheduler(sim)
    (run_dir / "workspace_manifest.json").write_text(json.dumps(
        {"files": None, "tools": tools_manifest(sim)}, indent=1))
    if ck is not None:
        checkpoint.restore(sim, sim.scheduler, resume_from, ck)
    else:
        sim.schedule.run_at("__bootstrap__", cfg.sim_start)

    server, server_task = await start_server(sim, host, port)
    web = await start_hosts(sim, task.web_apps(sim), host)
    handle = {"run_id": cfg.run_id, "env_url": sim.url, "token": sim.token,
              "agent_token": sim.agent_token, "hosts": dict(sim.hosts),
              "model": cfg.agent.model or "",
              "watchdog_seconds": cfg.watchdog_seconds,
              "sim_start": cfg.sim_start.isoformat(),
              "sim_end": cfg.sim_end.isoformat(),
              "pause_at": pause_at.isoformat() if pause_at else None,
              "resumed_from": str(resume_from) if resume_from else None,
              "pid": os.getpid(), "pgid": os.getpgid(0)}
    (run_dir / "run.json").write_text(json.dumps(handle, indent=1))
    if ready is not None:
        ready(handle)
    stopped = False
    try:
        last_watch = 0.0
        while not sim.scheduler.done:
            if server_task.done():
                server_task.result()
                raise RuntimeError("HTTP server exited")
            loop_t = asyncio.get_running_loop().time()
            if loop_t - last_watch >= WATCH_INTERVAL_S:
                last_watch = loop_t
                if not run_dir.is_dir():
                    raise RunDirGone(f"run directory gone: {run_dir}")
                if (run_dir / STOP_MARKER).exists():
                    stopped = True
                    break
            await asyncio.sleep(0.2)
    finally:
        server.should_exit = True
        await server_task
        await stop_hosts(web)
        sim.ledger.close()
    if stopped:
        (run_dir / "killed.json").write_text(json.dumps(
            {"run_id": cfg.run_id, "killed_at": _now(), "reason": "stop marker",
             "server_killed": True, "actor_killed": False,
             "spent_usd": round(sim.ledger.total_cost(), 4)}, indent=1))
        return None
    if sim.scheduler.paused:
        checkpoint.write_checkpoint(sim, sim.scheduler, resume_from)
        return None
    results = sim.results()
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one sim (detached mode).")
    parser.add_argument("--config", required=True, help="path to a run YAML")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", "-r", default=".",
                        help="base for relative data paths")
    parser.add_argument("--seed", type=int, default=None,
                        help="repetition label: overrides cfg.seed and "
                             "suffixes the run_id with -s<seed>")
    parser.add_argument("--run-id", default=None,
                        help="override cfg.run_id (the launcher keeps ids "
                             "unique across launches)")
    parser.add_argument("--model", default=None,
                        help="api_provider:model_name (openai:gpt-5.6-luna, "
                             "openrouter:deepseek/deepseek-v4-pro): overrides "
                             "cfg.agent.model; the model must be priced in "
                             "configs/model_costs.yaml and its name suffixes "
                             "the run_id")
    parser.add_argument("--mock-llm", action="store_true",
                        help="canned zero-cost LLM upstream (preflights)")
    parser.add_argument("--stretch", default=None,
                        help="shorten the sim to sim_start + this (2d, 36h)")
    parser.add_argument("--pause-at", default=None,
                        help="iso instant inside the window: stop the clock "
                             "there and write a checkpoint instead of results")
    parser.add_argument("--resume-from", default=None,
                        help="a paused run dir to continue from (same yaml "
                             "and model; the seed may differ)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    import dotenv

    repo_root = Path(args.repo_root).resolve()
    dotenv.load_dotenv(repo_root / ".env")

    cfg = load_config(args.config)
    if args.run_id:
        cfg.run_id = args.run_id
    apply_launch_model(cfg, args.model)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.run_id = f"{cfg.run_id}-s{args.seed}"
    if args.stretch:
        cfg.sim_end = min(cfg.sim_end, cfg.sim_start + parse_stretch(args.stretch))
    if args.mock_llm:
        install_mock_llm(cfg.sim_end.isoformat().replace("+00:00", "Z"))
        cfg.log_llm_traffic = False  # zero-token canned traffic is noise
    pause_at = checkpoint.parse_pause_at(cfg, args.pause_at)
    resume_from = Path(args.resume_from).resolve() if args.resume_from else None

    def announce(handle: dict) -> None:
        print(json.dumps(handle), flush=True)

    run_dir = Path(args.run_dir).resolve()
    try:
        results = asyncio.run(serve(cfg, repo_root=repo_root, run_dir=run_dir,
                                    host=args.host, port=args.port,
                                    ready=announce, pause_at=pause_at,
                                    resume_from=resume_from))
    except RunDirGone as e:
        print(f"run {cfg.run_id}: {e} — exiting", file=sys.stderr, flush=True)
        sys.exit(EXIT_RUN_DIR_GONE)
    if results is None:
        if (run_dir / checkpoint.CHECKPOINT).exists():
            print(f"run {cfg.run_id} paused at {pause_at.isoformat()} "
                  "(checkpoint.json written)", file=sys.stderr, flush=True)
        else:
            print(f"run {cfg.run_id} stopped by its stop marker (killed.json "
                  "written)", file=sys.stderr, flush=True)
        return
    primary = results["performance"]["primary"]
    print(f"run {results['run_id']} finished: {primary['name']} = "
          f"{primary['value']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
