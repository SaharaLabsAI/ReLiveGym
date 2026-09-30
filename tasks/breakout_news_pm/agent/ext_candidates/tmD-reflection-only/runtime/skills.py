"""Curated skill block: store + token-capped render + tools. Program library.

Files live under memory/ in the workspace:

  memory/skills.md        curated block; overwritten by reflection
  memory/block_meta.json  version counter + last update (traces reference it)

One store and one injection path for every learning cell, so representation
(raw memory vs curated skills) is never confounded with context length.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import trace
from .env_client import iso
from .tokens import BUDGET_TOKENS, truncate

BLOCK_TOKENS = BUDGET_TOKENS  # set by the program from its cell config

MEMORY_DIR = Path("memory")
SKILLS = MEMORY_DIR / "skills.md"
META = MEMORY_DIR / "block_meta.json"


def update_skills(text: str) -> dict:
    """Overwrite the curated skill block (reflection output)."""
    MEMORY_DIR.mkdir(exist_ok=True)
    SKILLS.write_text(text, encoding="utf-8")
    meta = {"version": block_version() + 1, "kind": "skills"}
    META.write_text(json.dumps(meta))
    return meta


def block_version() -> int:
    if META.exists():
        return json.loads(META.read_text()).get("version", 0)
    return 0


def render_block(budget_tokens: int | None = None) -> str:
    """The curated learned block: skills.md verbatim, hard-truncated."""
    if not SKILLS.exists():
        return ""
    return render_text(SKILLS.read_text(encoding="utf-8"), budget_tokens)


def render_text(text: str, budget_tokens: int | None = None) -> str:
    """render_block for an arbitrary block text (a replay renders a
    candidate block exactly as the live block would render)."""
    text = (text or "").strip()
    if not text:
        return ""
    budget = BLOCK_TOKENS if budget_tokens is None else budget_tokens
    return ("## What you have learned so far\n" + truncate(text, budget))


REMINDER_HEAD = ("Reminder (from your review of settled outcomes, "
                 "version {version}):")


def render_reminder(text: str | None = None, version: int | None = None,
                    budget_tokens: int | None = None) -> str:
    """The reminder message: skills.md —
    or a candidate text — under a provenance header, hard-truncated to
    the block budget. "" when there is nothing to remind of."""
    if text is None:
        if not SKILLS.exists():
            return ""
        text = SKILLS.read_text(encoding="utf-8")
    text = (text or "").strip()
    if not text:
        return ""
    budget = BLOCK_TOKENS if budget_tokens is None else budget_tokens
    v = block_version() if version is None else version
    return REMINDER_HEAD.format(version=v) + "\n" + truncate(text, budget)


def skills_tools(env) -> dict[str, dict]:
    def fn(args):
        meta = update_skills(str(args["text"]))
        trace.log(iso(env.now()), "reflect",
                  block_version_after=meta["version"],
                  skills=str(args["text"]))
        return meta

    return {
        "update_skills": {
            "doc": "update_skills(text: str) -> replace your skill block "
                   "(shown in your system prompt; <= 5000 tokens). Reflect "
                   "on outcomes and update it once per day.",
            "fn": fn,
        },
    }
