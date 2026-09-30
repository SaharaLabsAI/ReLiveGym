#!/usr/bin/env python
"""Settle an episode whose sim server died before `settle` (no
server/results.json).

`harness.mcp settle` needs the live server. A server killed with its
ledger intact (the batch runner kills on a non-zero actor exit; a crash;
a reboot) is rebuilt from that ledger through the resume path of
harness/checkpoint.py, then settled as usual:

    1. server/ -> server_dead/, with a checkpoint.json at the ledger's
       last instant (what a `--pause-at` there would have written)
    2. harness.serve --resume-from server_dead --run-dir server
       (ledger copied, scoring state replayed, wallet from the cost
       column, clock at the cut — no LLM call, nothing is billed)
    3. episode.json re-pointed at the new server; harness.mcp settle

The resulting server/ledger.jsonl is the whole chain plus one `resume`
row, and server/results.json covers the episode from sim_start: the actor
did nothing after the cut, exactly as in an episode settled by hand.

    python scripts/settle_dead_episode.py runs/episodes_grid/.../<episode dir> [...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def settle_dead(ep_dir: Path) -> dict:
    episode = ep_dir / "episode.json"
    ep = json.loads(episode.read_text())
    server, dead = Path(ep["server_dir"]), ep_dir / "server_dead"
    if (server / "results.json").exists():
        raise SystemExit(f"{ep_dir.name}: already settled")
    if _alive(ep.get("serve_pid")):
        raise SystemExit(f"{ep_dir.name}: the server (pid {ep['serve_pid']}) is "
                         "alive — use `python -m harness.mcp settle`")
    if dead.exists():
        raise SystemExit(f"{ep_dir.name}: {dead.name}/ exists — a recovery was "
                         "already attempted; inspect it first")
    cfg = json.loads((server / "config.json").read_text())
    rows = [json.loads(l) for l in
            (server / "ledger.jsonl").read_text().splitlines()]  # a torn last row fails here
    shutil.move(str(server), str(dead))
    (dead / "checkpoint.json").write_text(json.dumps({
        "run_id": cfg["run_id"], "seed": cfg["seed"],
        "paused_at": rows[-1]["sim_time"],
        "sim_start": cfg["sim_start"], "sim_end": cfg["sim_end"],
        "code_sha": None, "ledger_rows": len(rows),
        "spent_usd": round(sum(float(r.get("cost") or 0.0) for r in rows), 8),
        "flags": [], "parent": None,
        "written_at": "recovered by scripts/settle_dead_episode.py"}, indent=1))

    agent = cfg["agent"]
    cmd = [sys.executable, "-m", "harness.serve",
           "--config", str(ep_dir / "config_gen.yaml"), "--run-dir", str(server),
           "--repo-root", str(ROOT), "--seed", str(cfg["seed"]),
           "--resume-from", str(dead)]
    if agent.get("provider") and agent.get("model"):
        cmd += ["--model", f"{agent['provider']}:{agent['model']}"]
    log = open(ep_dir / "serve_resumed.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log,
                            start_new_session=True)
    t0 = time.monotonic()
    while not (server / "run.json").exists():
        if proc.poll() is not None:
            raise SystemExit(f"{ep_dir.name}: harness.serve exited "
                             f"{proc.returncode} (see serve_resumed.log); the "
                             f"original run dir is {dead}")
        if time.monotonic() - t0 > 300:
            proc.kill()
            raise SystemExit(f"{ep_dir.name}: timed out waiting for the server")
        time.sleep(0.5)
    handle = json.loads((server / "run.json").read_text())
    ep.update(env_url=handle["env_url"], token=handle["token"],
              agent_token=handle.get("agent_token") or handle["token"],
              serve_pid=proc.pid, hosts=handle.get("hosts") or ep.get("hosts"),
              recovered_from=str(dead))
    episode.write_text(json.dumps(ep, indent=1))
    subprocess.run([sys.executable, "-m", "harness.mcp", "settle",
                    "--episode", str(episode)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    proc.wait(timeout=60)
    res = json.loads((server / "results.json").read_text())
    return {"episode": ep_dir.name, "cut": rows[-1]["sim_time"],
            "primary": res["performance"]["primary"]["value"],
            "spent_usd": res["resources"]["spent_usd"], "flags": res.get("flags")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("episodes", nargs="+", help="episode directories")
    for d in ap.parse_args().episodes:
        print(json.dumps(settle_dead(Path(d).resolve())), flush=True)
