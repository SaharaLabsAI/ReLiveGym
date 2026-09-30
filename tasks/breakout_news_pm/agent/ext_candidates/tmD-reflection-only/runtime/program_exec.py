"""Subprocess entry for episode-mode `run_program`. Executes ONE authored program with
`runtime/program.py` semantics against the env named by ENV_URL/ENV_TOKEN
and writes the outcome dict as JSON to --out (programs own stdout, so the
result never travels through it).

Run with `python -I` (the agent's cwd and site stay off sys.path); the
scaffolds/ directory is added here so the `runtime` package imports the
same way it does from a workspace mount. The network fence around this
process is the caller's job (harness/mcp.py::_fence_cmd)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="authored program path")
    ap.add_argument("--root", required=True, help="program scratch root")
    ap.add_argument("--out", required=True, help="outcome JSON file")
    ap.add_argument("--until", default=None)
    ap.add_argument("--waiter-id", default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime.program import run_program

    try:
        outcome = run_program(args.file, until=args.until,
                              waiter_id=args.waiter_id, root=args.root,
                              validate=args.validate)
    except Exception as e:  # bad `until`, env unreachable, ...
        outcome = {"woke_for": "error", "error": f"{type(e).__name__}: {e}"}
    Path(args.out).write_text(json.dumps(outcome, default=str),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
