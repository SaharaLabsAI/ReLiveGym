"""The published actor contract as data: everything config-specific a
workspace holds besides its own program
and `runtime/` — the tool manifest, the LLM proxy terms, the rendered
INSTRUCTION.md and cell_config.py — plus a typed client stub of the
program-callable tools (envkit).

Served live by every sim as `GET /contract` (the runner writes the two
files into the workspace before the first trigger) and written to disk by
scripts/gen_contract.py so an author can read it without a running sim.
One source: the @tool declarations, the task's INSTRUCTION.md template and
scaffolds/compose.render_cell_config — nothing here restates them.
"""

from __future__ import annotations

import string
from pathlib import Path

from harness.config import RunConfig
from harness.task import Task, task_dir

# tools an authored program may not call from client-side envkit: waiting
# and scheduling move sim time or the roster (the program passes time only
# through envkit.wait), the file tools are the server-side TM-B jail
NOT_PROGRAM_TOOLS = frozenset({
    "sleep", "run_program", "set_party", "get_crontab", "set_crontab",
    "run_at", "ls", "read_file", "write_file", "edit_file"})


def is_external(cfg: RunConfig) -> bool:
    """`ext:<name>` — an externally-authored program: adhoc
    provisioning, client-side TM-B, no harness skill appendix."""
    return cfg.agent.scaffold.startswith("ext:")


def is_adhoc(cfg: RunConfig) -> bool:
    """task: or ext: — checked-in / external main.py + generated
    cell_config.py, wait-party roster tool provisioned."""
    return cfg.agent.scaffold.startswith(("task:", "ext:"))


def _run_model_rates(cfg: RunConfig) -> dict[str, float] | None:
    """The pinned $/token rates of THIS run's model (per-run override,
    else the global table) — the only rates the agent ever sees."""
    from harness.model_costs import resolve_rates

    if not cfg.agent.model:
        return None
    return resolve_rates(cfg, cfg.agent.model)


def llm_price_table(cfg: RunConfig) -> str:
    """Human-readable LLM token prices for INSTRUCTION.md: the run
    model's pinned rates, cached-input rate included when discounted."""
    rates = _run_model_rates(cfg)
    if rates is None:
        return ("the provider's list token prices (billed per input/output "
                "token)")
    cached = rates.get("cached_in")
    cached_txt = (f" (${cached * 1e6:g}/Mtok cached)"
                  if cached is not None else "")
    return (f"{cfg.agent.model}: ${rates.get('in', 0.0) * 1e6:g}/Mtok "
            f"in{cached_txt}, ${rates.get('out', 0.0) * 1e6:g}/Mtok out")


def render_instruction(cfg: RunConfig, task: Task,
                       hosts: dict[str, str] | None = None) -> str | None:
    """The task's INSTRUCTION.md (the agent-facing task spec u): ${...}
    placeholders are filled from the task's context plus the harness-level
    budget, LLM prices and the web hosts' URLs (`${<name>_url}`), so the
    spec always matches the run's actual config. None when
    the task ships no template."""
    template = task_dir(cfg.task_name) / "INSTRUCTION.md"
    if not template.exists():
        return None
    caps = [f"${v:g} on {'LLM calls' if k == 'llm' else k.replace('_', ' ')}"
            for k, v in sorted(cfg.domain_budgets.items())]
    domain_caps = ("" if not caps else
                   " Within it, hard per-category caps apply: "
                   + "; ".join(caps) + " — a spent cap refuses that "
                   "category's calls while everything else keeps working.")
    # how sim time passes for this arm (web tasks' "Time, budget, cost"
    # paragraph): a cron-fired agent has no wait tool
    cron = cfg.cell.tm in ("C", "D")
    context = {"budget_usd": f"${cfg.budget_usd:g}",
               "domain_caps": domain_caps,
               "llm_price_table": llm_price_table(cfg),
               "time_passes": ("between your wakings" if cron
                               else "inside your wait tool"),
               "stale_since": ("in an earlier waking" if cron
                               else "before a wait"),
               **{f"{n}_url": u for n, u in (hosts or {}).items()},
               **task.instruction_context()}
    text = string.Template(template.read_text()).safe_substitute(context)
    if cfg.cell.tm == "B" and not is_external(cfg):
        # the TM-B skill appendix: the rendered instruction tracks
        # the cell's tool surface (no sleep tool; waiting = run_program).
        # ext: arms wait by client-side program — their skill
        # text is their own, not this appendix
        skill = Path(__file__).resolve().parent / "authored_skill.md"
        text = text.rstrip() + "\n\n" + skill.read_text()
    return text


def llm_terms(cfg: RunConfig) -> dict:
    """The /llm proxy contract as data."""
    from harness.llm_proxy import SUPPORTED_PATHS

    rates = _run_model_rates(cfg)
    return {"endpoint": "/llm/<path>",
            "paths": sorted(SUPPORTED_PATHS),
            "model": cfg.agent.model,
            "token_rates_usd": ({cfg.agent.model: dict(rates)}
                                if rates is not None else {}),
            "price_table": llm_price_table(cfg),
            "budget_usd": cfg.budget_usd,
            "domain_budgets": dict(cfg.domain_budgets)}


def contract_payload(sim) -> dict:
    """The GET /contract body."""
    from harness.api import tools_manifest
    from harness.web import readonly_hosts
    from scaffolds.compose import render_cell_config

    tools = tools_manifest(sim)
    return {"run_id": sim.cfg.run_id,
            "task": sim.cfg.task_name,
            "cell": sim.cfg.cell.label,
            "sim_start": sim.cfg.sim_start.isoformat(),
            "sim_end": sim.cfg.sim_end.isoformat(),
            "tools": tools,
            "llm": llm_terms(sim.cfg),
            "hosts": dict(sim.hosts),
            "hosts_readonly": readonly_hosts(sim),
            "instruction_md": render_instruction(sim.cfg, sim.task, sim.hosts),
            "cell_config_py": render_cell_config(sim.cfg),
            "envkit_py": render_envkit(tools)}


# -- the client-side envkit stub ------------------------------------------------------

_ENVKIT_HEADER = '''\
"""envkit — an authored program's ONLY window on the world.

Import this from a program run via `runtime.program.run_program`; its
functions fail cleanly anywhere else. The contract:

- Observation = the fetch functions below; each call is billed exactly
  like the same-named agent tool. Fetching never moves the clock.
- wait(until) is the ONLY way sim time advances, and it is free. A loop
  that never waits makes no sim-time progress (and the run's real-time
  watchdog kills a program that stops calling the environment).
- handover(payload) ends the run: payload (JSON-able, <= <LIMIT> KB)
  becomes the run_program result your loop sees next. Write bulk to
  files under workspace() and hand over a pointer + summary.
- A due scheduled trigger or the run_program deadline also ends the run
  (woke_for "trigger" / "timeout").
- Local compute and file IO under workspace() — your private scratch —
  are free: they happen at a frozen instant.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _bridge():
    # runtime.program.run_program injects _current_bridge when it loads
    # this module; the import is the fallback for other loaders
    cb = globals().get("_current_bridge")
    if cb is None:
        from runtime.program import current_bridge as cb
    return cb()


def workspace() -> Path:
    """Your private scratch root — the only writable place."""
    return _bridge().root


def now() -> datetime:
    """Current sim time (free; constant between wait() calls)."""
    return _bridge().now()


def deadline() -> datetime:
    """This run's hard deadline (run_program's `until`, or experiment
    end): reaching it exits the program with woke_for "timeout"."""
    return _bridge().deadline


def wait(until: "datetime | str") -> dict:
    """Advance sim time to `until` (free); returns {"now": ...}. A due
    trigger, the run deadline, or experiment end exits the program
    instead, returning control to your loop."""
    return _bridge().wait(until)


def handover(payload) -> None:
    """End the run: `payload` becomes the run_program result."""
    _bridge().handover(payload)
'''

HANDOVER_LIMIT_KB = 16


def program_tools(manifest: list[dict]) -> list[dict]:
    """The manifest entries an authored program may call."""
    return [t for t in manifest
            if t["name"] not in NOT_PROGRAM_TOOLS
            and "wait" not in (t.get("tags") or ())]


DEFAULT_RUNNER_REF = "a program run via `runtime.program.run_program`"


def render_envkit(manifest: list[dict], runner: str = DEFAULT_RUNNER_REF) -> str:
    """envkit.py for a client-side authored program: the static contract +
    one wrapper per callable tool, doc verbatim from the manifest — the
    API reference and the call surface are one artifact. `runner` names
    how the program gets run in the header sentence (default: the ext:
    actor contract, byte-identical to before)."""
    header = _ENVKIT_HEADER.replace("<LIMIT>", str(HANDOVER_LIMIT_KB))
    header = header.replace(
        "Import this from a program run via `runtime.program.run_program`; its",
        f"Import this from {runner}; its")
    parts = [header]
    for t in sorted(program_tools(manifest), key=lambda t: t["name"]):
        doc = t["doc"].replace("\\", "\\\\").replace('"""', "'''")
        parts.append(f"def {t['name']}(**args) -> dict:\n"
                     f'    """{doc} [{t["price"]}]"""\n'
                     f'    return _bridge().call("{t["name"]}", args)\n')
    return "\n\n".join(parts)


def render_tools_md(payload: dict) -> str:
    """tools.md — the manifest as a readable table + the /llm terms."""
    lines = [f"# Tool contract — {payload['task']} / {payload['cell']}", "",
             f"Sim window: {payload['sim_start']} → {payload['sim_end']}.",
             "Every tool is `POST /call/<name>` with a JSON object of args "
             "(bearer token in `Authorization`); the manifest itself is "
             "`GET /tools`. Docs below are verbatim what the actor sees.", "",
             "| tool | price | tags | doc |", "|---|---|---|---|"]
    for t in payload["tools"]:
        doc = t["doc"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{t['name']}` | {t['price']} | "
                     f"{', '.join(t.get('tags') or [])} | {doc} |")
    llm = payload["llm"]
    lines += ["", "## LLM proxy", "",
              f"`POST {llm['endpoint']}` for path in {llm['paths']}; "
              f"model for this cell: `{llm['model']}`; prices: "
              f"{llm['price_table']}. Budget ${llm['budget_usd']:g}"
              + (f", domain caps {llm['domain_budgets']}"
                 if llm["domain_budgets"] else "") + ".", ""]
    return "\n".join(lines)
