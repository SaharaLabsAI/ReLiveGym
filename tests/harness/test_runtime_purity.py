"""Task-purity invariant: scaffolds/runtime holds only task-agnostic
capabilities. Task semantics enter a workspace only through task files
mounted from tasks/<name>/agent/ (records.py hooks, prompts,
compose_spec.py), never through the shared library. Enforced as a test
so the boundary survives the next task port."""

from __future__ import annotations

import re
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[2] / "scaffolds" / "runtime"

# vocabulary of the tasks that have run on the v3 runtime so far; extend
# when a new task's nouns must not leak in
FORBIDDEN = re.compile(r"market|news|breakout|gold_group|seen_by_you",
                       re.IGNORECASE)


def test_runtime_library_is_task_free():
    assert RUNTIME.is_dir()
    hits = []
    for p in sorted(RUNTIME.glob("*.py")):
        for i, line in enumerate(
                p.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, \
        "task vocabulary leaked into scaffolds/runtime:\n" + "\n".join(hits)
