"""Run one experiment — the COMBINED mode: workspace init, HTTP server,
scheduler and actor runner in one process, results.

CLI:  python -m harness.run --config tasks/weather_fixture/configs/baseline.yaml
      [--scaffold-src <program dir>]   (an ext:/fixture program copied verbatim)
Programmatic (tests): asyncio.run(run_experiment(cfg, ...)).

Since the server/actor detach
the lifecycle is two halves — harness/supervisor.Scheduler on the server
and scaffolds/runtime/actor.py as the runner — and this entry point simply
runs both: the runner loop executes in a worker thread against the local
server over HTTP, exactly as a detached runner would (harness.serve +
`python runtime/actor.py run`), so the two modes produce the same ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn

from harness import checkpoint
from harness.api import make_app, tools_manifest
from harness.config import RunConfig, load_config
from harness.contract import is_adhoc, is_external, render_instruction
from harness.model_costs import apply_launch_model, model_slug
from harness.runtime import Sim
from harness.supervisor import Scheduler
from harness.task import Task, load_task_class
from harness.web import start_hosts, stop_hosts

RUNTIME_SRC = Path(__file__).resolve().parents[1] / "scaffolds" / "runtime"


def default_run_dir(repo_root: Path, cfg: RunConfig) -> Path:
    """Where a run lands unless the caller passes run_dir explicitly:
    runs are grouped per task, then per model —
    runs/task_<task_name>/<model_slug>/<run_id> (no model set, e.g. the
    mock-LLM e2e: runs/task_<task_name>/<run_id>)."""
    base = repo_root / "runs" / f"task_{cfg.task_name}"
    if cfg.agent.model:
        base = base / model_slug(cfg.agent.model)
    return base / cfg.run_id


def _init_workspace(cfg: RunConfig, run_dir: Path,
                    scaffold_src: Path | None) -> tuple[Path, list[str]]:
    """Materialize the workspace program: either the
    constructor builds it from the cell config, or — for fixture/hand-
    written programs — `scaffold_src` is copied verbatim. Returns the
    workspace and its file list."""
    workspace = run_dir / "workspace"
    if scaffold_src is not None:
        workspace.mkdir(parents=True)
        shutil.copytree(scaffold_src, workspace, dirs_exist_ok=True)
        if is_adhoc(cfg):
            # an external/adhoc program is a complete actor program; if it
            # was authored without the library, mount runtime/ as compose
            # would, and give it this cell's cell_config.py
            from scaffolds.compose import render_cell_config

            if not (workspace / "runtime" / "env_client.py").exists():
                shutil.copytree(RUNTIME_SRC, workspace / "runtime",
                                dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__"))
            cell_config = render_cell_config(cfg)
            if cell_config is not None:
                (workspace / "cell_config.py").write_text(cell_config,
                                                          encoding="utf-8")
        (workspace / "logs").mkdir(exist_ok=True)
        (workspace / "memory").mkdir(exist_ok=True)
        files = sorted(str(p.relative_to(workspace))
                       for p in workspace.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
    elif is_external(cfg):
        raise ValueError(f"scaffold {cfg.agent.scaffold!r} is an external "
                         "program: pass its directory as scaffold_src "
                         "(--scaffold-src)")
    else:
        from scaffolds.compose import build_workspace

        workspace.mkdir(parents=True)
        files = build_workspace(cfg, workspace)
    if not (workspace / "main.py").exists():
        raise FileNotFoundError("workspace provides no main.py (the program "
                                "entry point)")
    return workspace, files


def _init_workspace_git(workspace: Path) -> None:
    """Alg-C/D sandbox: the editable program dir is a git
    repo; reflection edits are committed by the agent and the supervisor can
    roll back to the last good commit after post-edit crash streaks."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(workspace), *args],
                       capture_output=True, check=True)

    # runtime artifacts are data, not code: keep them out of the edit
    # history so reflection commits and the scope audit see only real edits
    (workspace / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nlogs/\nmemory/\nstate.json\nstate.tmp\n")
    git("init", "-q")
    git("config", "user.email", "harness@experiment")
    git("config", "user.name", "harness")
    git("add", "-A")
    git("commit", "-q", "-m", "workspace init", "--allow-empty")


def _render_instruction(cfg: RunConfig, task: Task, workspace: Path,
                        hosts: dict[str, str] | None = None) -> None:
    """Write the rendered INSTRUCTION.md (harness/contract.py) into the
    workspace, when the task ships a template."""
    text = render_instruction(cfg, task, hosts)
    if text is not None:
        (workspace / "INSTRUCTION.md").write_text(text)


async def start_server(sim: Sim, host: str = "127.0.0.1", port: int = 0):
    """Bind the env server for `sim` and set sim.url. Returns (server,
    task): set server.should_exit and await task to stop."""
    server = uvicorn.Server(uvicorn.Config(
        make_app(sim), host=host, port=port, log_level="warning",
        access_log=False))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        if server_task.done():
            server_task.result()  # surface startup errors
            raise RuntimeError("HTTP server exited before starting")
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    sim.url = f"http://{host}:{port}"
    return server, server_task


async def run_experiment(
    cfg: RunConfig,
    *,
    repo_root: Path | None = None,
    run_dir: Path | None = None,
    scaffold_src: Path | None = None,
    python_exe: str = sys.executable,
    pause_at=None,
    resume_from: Path | None = None,
) -> dict | None:
    """Run one experiment to sim_end and return the results dict.

    `scaffold_src`: explicit program directory copied verbatim (fixture and
    hand-written programs); default is the constructor-built workspace
    (scaffolds/compose.py, keyed on cfg.cell + cfg.agent.scaffold).
    `pause_at` / `resume_from` (harness/checkpoint.py): stop at an instant
    inside the window and write a checkpoint (returns None), or continue a
    paused run — `scaffold_src` is then the snapshot workspace to copy.
    """
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    run_dir = Path(run_dir) if run_dir else default_run_dir(repo_root, cfg)
    pause_at = checkpoint.parse_pause_at(cfg, pause_at)
    ck = None
    if resume_from is not None:
        ck = checkpoint.prepare_resume(run_dir, resume_from, cfg)
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    task = load_task_class(cfg.task_name).from_run_config(cfg, repo_root)

    workspace, files = _init_workspace(
        cfg, run_dir, Path(scaffold_src) if scaffold_src else None)
    _render_instruction(cfg, task, workspace)
    if cfg.cell.alg in ("config", "full"):
        _init_workspace_git(workspace)

    (run_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=1, default=str))

    sim = Sim(cfg, run_dir, workspace, task, pause_at=pause_at)
    sim.scheduler = Scheduler(sim)
    # what this run's agent could see and call, recorded per run (P3's fix)
    (run_dir / "workspace_manifest.json").write_text(json.dumps(
        {"files": files, "tools": tools_manifest(sim)}, indent=1))
    if ck is not None:
        checkpoint.restore(sim, sim.scheduler, resume_from, ck)
    else:
        sim.schedule.run_at("__bootstrap__", cfg.sim_start)

    server, server_task = await start_server(sim)
    web = await start_hosts(sim, task.web_apps(sim))
    if sim.hosts:  # ${<name>_url} placeholders exist only once bound
        _render_instruction(cfg, task, workspace, sim.hosts)

    # the actor runner (scaffolds/runtime/actor.py) in its own thread,
    # talking to the server above over HTTP — the same program a detached
    # actor runs; a dedicated executor so a parked program pool can never
    # starve it
    from scaffolds.runtime.actor import run_loop

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actor-runner")
    try:
        await asyncio.get_running_loop().run_in_executor(
            pool, run_loop, sim.url, sim.token, cfg.agent.model or "",
            workspace, run_dir, python_exe, True)
    finally:
        pool.shutdown(wait=False)
        server.should_exit = True
        await server_task
        await stop_hosts(web)
        sim.ledger.close()

    if sim.scheduler.paused:
        checkpoint.write_checkpoint(sim, sim.scheduler, resume_from)
        return None
    results = sim.results()
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one agent experiment.")
    parser.add_argument("--config", required=True, help="path to a run YAML")
    parser.add_argument("--repo-root", "-r", default=".", help="base for runs/ and relative data paths")
    parser.add_argument("--seed", type=int, default=None,
                        help="repetition label: overrides cfg.seed and "
                             "suffixes the run_id with -s<seed>")
    parser.add_argument("--model", default=None,
                        help="api_provider:model_name (openai:gpt-5.6-luna, "
                             "openrouter:deepseek/deepseek-v4-pro): overrides "
                             "cfg.agent.model; the model must be priced in "
                             "configs/model_costs.yaml and its name suffixes "
                             "the run_id")
    parser.add_argument("--scaffold-src", default=None,
                        help="program directory copied verbatim as the "
                             "workspace (ext:/fixture programs)")
    parser.add_argument("--pause-at", default=None,
                        help="iso instant inside the window: stop there and "
                             "write a checkpoint instead of results")
    parser.add_argument("--resume-from", default=None,
                        help="a paused run dir to continue from (then "
                             "--scaffold-src is the snapshot workspace)")
    args = parser.parse_args(argv)

    # real provider keys live only in the helper process
    import dotenv

    dotenv.load_dotenv(Path(args.repo_root).resolve() / ".env")

    cfg = load_config(args.config)
    apply_launch_model(cfg, args.model)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.run_id = f"{cfg.run_id}-s{args.seed}"
    results = asyncio.run(run_experiment(
        cfg, repo_root=Path(args.repo_root).resolve(),
        scaffold_src=Path(args.scaffold_src) if args.scaffold_src else None,
        pause_at=args.pause_at,
        resume_from=Path(args.resume_from).resolve() if args.resume_from else None))
    if results is None:
        print(f"run {cfg.run_id} paused at {args.pause_at} (checkpoint.json "
              f"written under {default_run_dir(Path(args.repo_root).resolve(), cfg)})")
        return

    primary = results["performance"]["primary"]
    res = results["resources"]
    value = primary["value"]
    print(f"run {results['run_id']} finished: {primary['name']} = "
          f"{value if value is not None else 'n/a'} "
          f"(spent ${res['spent_usd']:.2f} of ${res['budget_usd']:g}"
          f"{'' if results['constraints']['within_budget'] else ' — EXHAUSTED'})")
    if results["flags"]:
        print(f"flags: {', '.join(results['flags'])}")
    if res["llm_usd"] or res["llm_provider_usd"]:
        print(f"llm: booked ${res['llm_usd']:.4f} (config rates) vs "
              f"provider-issued ${res['llm_provider_usd']:.4f}")
    print(f"details: {default_run_dir(Path(args.repo_root).resolve(), cfg) / 'results.json'}")


if __name__ == "__main__":
    main()
