"""Client-side TM-B affordances for an externally-run (ext:) per-market
program: the `run_program` wait tool and the jail file tools
(`ls`, `read_file`, `write_file`, `edit_file`) that the constructor's tm=B gets from
the environment's WorkspaceApp + ProgramApp.

Under `ext:` the environment provisions only `sleep` — agent code never
executes inside the env process — so the TM-B discipline is the program's
own: each market agent's tool
registry gets these five entries, with the SAME names and docstrings as
the environment's, implemented on a local jail `agents/<waiter_id>/` and
on `runtime.program.run_program`, whose only clock is the env's `sleep`
endpoint. Same tools, same wake semantics; error texts differ in shape
(a local exception instead of an HTTP 4xx).

Jails are seeded like the environment's: the sim's published envkit.py
(GET /contract), plus every `jail_seed/*.py` file shipped with the
candidate (sleep.py, example_gatekeeper.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from runtime import program

JAILS = Path("agents")
SEEDS = Path("jail_seed")
READ_LIMIT_KB = 16

# Docstrings verbatim from the environment's authored-program tools (a
# repo-side test pins them to the server's declarations).
DOCS = {
    "ls": ("ls(path?: str) -> free: list your private scratch workspace "
           "(yours alone; ships envkit.py — the API an authored program "
           "imports — sleep.py, the blind-wait program, and "
           "example_gatekeeper.py, a runnable watcher template)"),
    "read_file": ("read_file(path: str, offset?: int, limit?: int) -> free: read a "
                  "workspace file (1-based line offset, default limit 2000 lines; "
                  f"a page is clipped at {READ_LIMIT_KB} KB)"),
    "write_file": ("write_file(path: str, content: str) -> free: create or "
                   "overwrite a workspace file (parent dirs created)"),
    "edit_file": ("edit_file(path: str, old: str, new: str) -> free: in-place "
                  "edit — `old` must occur exactly once in the file"),
    "run_program": ("run_program(path: str, until?: iso datetime, validate?: bool) "
                    "-> run a Python program from your workspace forward through "
                    "sim time, until it calls envkit.handover(payload) — returned "
                    "here verbatim — a scheduled trigger fires, or the `until` "
                    "deadline (a hard timeout backstop, default: experiment end; "
                    "the program picks its actual wake times via envkit.wait). "
                    "Inside it the world is observable ONLY through envkit's fetch "
                    "functions (each billed like the same-named tool); sim time "
                    "advances ONLY inside envkit.wait (free); local compute and "
                    "workspace file IO are free. validate=true dry-runs it at zero "
                    "cost and zero sim time, surfacing errors. Read envkit.py in "
                    "your workspace for the exact API."),
}
TAGS = {"run_program": {"wait"}}


def jail(name: str, env) -> Path:
    """This waiter's private scratch root, seeded on first use."""
    root = (JAILS / name).resolve()
    if not (root / "envkit.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        program.ensure_envkit(env, root)  # the sim's published stub
        if SEEDS.is_dir():
            for f in sorted(SEEDS.glob("*.py")):
                if not (root / f.name).exists():
                    shutil.copyfile(f, root / f.name)
    return root


def _resolve(root: Path, path, field: str = "path") -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field!r} must be a non-empty string")
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path outside your workspace: {path!r}")
    return p


def ls(name: str, env, args: dict) -> dict:
    root = jail(name, env)
    base = _resolve(root, args.get("path") or ".")
    if not base.exists():
        raise ValueError(f"no such path: {args.get('path')!r}")
    files = sorted(p for p in base.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    return {"root": str(JAILS / name),
            "files": [{"path": str(p.relative_to(root)),
                       "bytes": p.stat().st_size} for p in files]}


def read_file(name: str, env, args: dict) -> dict:
    root = jail(name, env)
    p = _resolve(root, args.get("path"))
    if not p.is_file():
        raise ValueError(f"no such file: {args.get('path')!r}")
    offset = args.get("offset", 1)
    limit = args.get("limit", 2000)
    if not isinstance(offset, int) or offset < 1:
        raise ValueError("offset must be an int >= 1")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an int >= 1")
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    page = lines[offset - 1:offset - 1 + limit]
    content = "".join(page)
    out = {"path": args["path"], "offset": offset,
           "total_lines": len(lines), "lines": len(page),
           "truncated": offset - 1 + limit < len(lines)}
    blob = content.encode("utf-8")
    if len(blob) > READ_LIMIT_KB * 1024:
        content = blob[:READ_LIMIT_KB * 1024].decode("utf-8", errors="ignore")
        out["truncated"] = True
        out["clipped"] = (f"page over the {READ_LIMIT_KB} KB read "
                          f"limit ({len(blob)} bytes) — showing the "
                          f"head; have a program summarize the file "
                          f"instead of reading it whole")
    out["content"] = content
    return out


def write_file(name: str, env, args: dict) -> dict:
    root = jail(name, env)
    p = _resolve(root, args.get("path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("'content' must be a string")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"path": args["path"], "bytes": len(content.encode())}


def edit_file(name: str, env, args: dict) -> dict:
    root = jail(name, env)
    p = _resolve(root, args.get("path"))
    if not p.is_file():
        raise ValueError(f"no such file: {args.get('path')!r}")
    old, new = args.get("old"), args.get("new")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise ValueError("edit_file needs a non-empty string 'old' and "
                         "a string 'new'")
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        raise ValueError(f"`old` occurs {n} times (must be exactly 1)")
    text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    return {"path": args["path"], "bytes": len(text.encode())}


def run_program(name: str, env, args: dict) -> dict:
    """The wait tool: run an authored file from this waiter's jail forward
    through sim time (runtime.program; waits go to the env's sleep with
    this waiter_id, so the party barrier is honoured)."""
    root = jail(name, env)
    file = _resolve(root, args.get("path"))
    if not file.is_file():
        raise ValueError(f"no such program: {args.get('path')!r}")
    return program.run_program(file, until=args.get("until"), waiter_id=name,
                               env=env, root=root,
                               validate=bool(args.get("validate")))


_FNS = {"ls": ls, "read_file": read_file, "write_file": write_file,
        "edit_file": edit_file, "run_program": run_program}


def jail_tools(name: str, env) -> dict[str, dict]:
    """Registry entries (agent.env_tools shape) for one waiter, bound to
    its jail. A `waiter_id` in the args is accepted and ignored — the
    binding IS the identity."""
    def bind(fn):
        return lambda args, _fn=fn: _fn(
            name, env, {k: v for k, v in (args or {}).items() if k != "waiter_id"})

    return {n: {"doc": DOCS[n], "tags": set(TAGS.get(n, ())), "fn": bind(f)}
            for n, f in _FNS.items()}
