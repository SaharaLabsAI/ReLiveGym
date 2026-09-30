"""Write the published actor contract of a config to disk — what a running
sim serves as GET /contract,
for an author who has no running sim:

    python scripts/gen_contract.py --config tasks/x/configs/y.yaml --out contract/y

    contract/y/
      tools.json        the manifest [{name, doc, price, tags}] + llm terms
      tools.md          the same, readable
      envkit.py         typed client stub of the program-callable tools
      INSTRUCTION.md    the rendered task instruction for this config
      cell_config.py    this cell's parameters (task:/ext: programs)
      base.json         id, task, cell, window, budget, model — the summary

Provisions the apps for the config WITHOUT starting a run (a Sim on a
temp dir; the task's data is loaded, so this can take a while for big
tasks). Nothing here restates a tool: the @tool declarations, the task's
INSTRUCTION.md template and scaffolds/compose.render_cell_config are the
single sources (harness/contract.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.config import RunConfig, load_config  # noqa: E402
from harness.contract import contract_payload, render_tools_md  # noqa: E402
from harness.runtime import Sim  # noqa: E402
from harness.task import load_task_class  # noqa: E402


def build_payload(cfg: RunConfig, repo_root: Path = REPO_ROOT) -> dict:
    task = load_task_class(cfg.task_name).from_run_config(cfg, repo_root)
    with tempfile.TemporaryDirectory() as tmp:
        sim = Sim(cfg, Path(tmp), Path(tmp) / "workspace", task)
        payload = contract_payload(sim)
        sim.ledger.close()
    return payload


def base_summary(cfg: RunConfig, base_id: str) -> dict:
    return {"id": base_id, "task": cfg.task_name, "cell": cfg.cell.label,
            "scaffold": cfg.agent.scaffold, "model": cfg.agent.model,
            "sim_start": cfg.sim_start.isoformat(),
            "sim_end": cfg.sim_end.isoformat(),
            "budget_usd": cfg.budget_usd,
            "domain_budgets": dict(cfg.domain_budgets),
            "watchdog_seconds": cfg.watchdog_seconds,
            "context_tokens": cfg.agent.context_tokens}


def write_contract(cfg: RunConfig, out: Path, base_id: str,
                   repo_root: Path = REPO_ROOT) -> dict:
    payload = build_payload(cfg, repo_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tools.json").write_text(json.dumps(
        {"tools": payload["tools"], "llm": payload["llm"]}, indent=1))
    (out / "tools.md").write_text(render_tools_md(payload))
    (out / "envkit.py").write_text(payload["envkit_py"])
    if payload["instruction_md"] is not None:
        (out / "INSTRUCTION.md").write_text(payload["instruction_md"])
    if payload["cell_config_py"] is not None:
        (out / "cell_config.py").write_text(payload["cell_config_py"])
    (out / "base.json").write_text(json.dumps(base_summary(cfg, base_id),
                                              indent=1))
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", default=None, help="base id (default: yaml stem)")
    parser.add_argument("--repo-root", "-r", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    base_id = args.id or Path(args.config).stem
    payload = write_contract(cfg, Path(args.out), base_id,
                             Path(args.repo_root).resolve())
    print(f"wrote {args.out}: {len(payload['tools'])} tools, cell "
          f"{payload['cell']}")


if __name__ == "__main__":
    main()
