"""Run ONE experiment config as several repetitions and print a summary.

Each repetition launches `python -m harness.run --config <yaml> --seed N`
as its own OS process — the exact runtime of a science run — so runs land
side by side under runs/task_<task_name>/ as `<run_id>-s<seed>` and share
nothing but the machine. (They used to share one event loop, whose
default thread pool two in-process tm=B permarket sims can exhaust.) NB:
the harness does not feed `seed` to
the LLM provider — repetitions differ through provider sampling
nondeterminism; the seed is the repetition's label, recorded in
config.json.

Usage:
  python scripts/run_seeds.py \
      tasks/reddit_ai_popularity/configs/cells/tmB-tlrndaily-sigoracle-algmemory.yaml \
      --seeds 1 2 3 [--model api_provider:model_name] [--parallel N] [--force]

Repetitions run in parallel by default; cap concurrency with --parallel N
to bound the number of paid LLM runs billing at once.
"""

import argparse
import asyncio
import json
import shutil
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.config import load_config  # noqa: E402
from harness.model_costs import apply_launch_model  # noqa: E402
from harness.run import default_run_dir  # noqa: E402


async def run_one(cfg_path: Path, seed: int, model: str | None, force: bool,
                  sem: asyncio.Semaphore):
    # the same run_id derivation harness.run applies in the subprocess,
    # so the exists-check below looks at the dir the run will use
    cfg = load_config(cfg_path)
    apply_launch_model(cfg, model)
    cfg.seed = seed
    cfg.run_id = f"{cfg.run_id}-s{seed}"
    run_dir = default_run_dir(REPO_ROOT, cfg)
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        else:
            print(f"[{cfg.run_id}] SKIPPED: {run_dir} exists (--force to redo)")
            return cfg.run_id, None, None
    async with sem:
        print(f"[{cfg.run_id}] starting")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "harness.run",
            "--config", str(cfg_path), "--seed", str(seed),
            *(["--model", model] if model else []),
            "--repo-root", str(REPO_ROOT),
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        output, _ = await proc.communicate()
        results_path = run_dir / "results.json"
        if proc.returncode != 0 or not results_path.exists():
            tail = output.decode(errors="replace")[-2000:]
            err = RuntimeError(f"harness.run exited {proc.returncode}: {tail}")
            print(f"[{cfg.run_id}] FAILED (exit {proc.returncode}); "
                  f"output tail:\n{tail}")
            return cfg.run_id, None, err
        results = json.loads(results_path.read_text())
        p = results["performance"]["primary"]
        print(f"[{cfg.run_id}] done: {p['name']} = {p['value']}")
        return cfg.run_id, results, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Repeat one run config across seeds.")
    ap.add_argument("config", type=Path, help="path to the run YAML")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default=None,
                    help="api_provider:model_name, passed through to every "
                         "repetition (overrides cfg.agent.model)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="concurrent runs (default: all seeds at once)")
    ap.add_argument("--force", action="store_true",
                    help="delete and redo existing repetition run dirs")
    args = ap.parse_args()

    import dotenv

    dotenv.load_dotenv(REPO_ROOT / ".env")  # provider-key launch check

    async def sweep():
        sem = asyncio.Semaphore(args.parallel or len(args.seeds))
        return await asyncio.gather(*(
            run_one(args.config, s, args.model, args.force, sem)
            for s in args.seeds))

    rows = asyncio.run(sweep())

    print("\n== summary ==")
    values = []
    for run_id, results, err in rows:
        if results is None:
            print(f"{run_id}: {'FAILED: ' + str(err) if err else 'skipped'}")
            continue
        p = results["performance"]["primary"]
        res = results["resources"]
        flags = ",".join(results["flags"]) or "-"
        print(f"{run_id}: {p['name']} = {p['value']}"
              f" (spent ${res['spent_usd']:.2f}, flags: {flags})")
        if p["value"] is not None:
            values.append(p["value"])
    if len(values) >= 2:
        print(f"n={len(values)} mean = {statistics.mean(values):.4f} "
              f"sd = {statistics.stdev(values):.4f} "
              f"range = [{min(values)}, {max(values)}]")


if __name__ == "__main__":
    main()
