"""Tiny persistent state helper. Program library.

Each launch of your program is a fresh process: anything you want to remember
between runs must live in a file. This wraps state.json in the workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path("state.json")


def load_state() -> dict:
    if _PATH.exists():
        return json.loads(_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    tmp = _PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(_PATH)
