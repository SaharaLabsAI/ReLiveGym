"""The constructor:
compiles a run config into a concrete workspace program.

Harness-side only — this file is never copied into a workspace. It maps the
cell spec to (a) the exact file set the workspace gets and (b) a generated
main.py: short, straight-line, branch-free code containing only what the
cell uses. The generated file IS the program; an alg-full agent edits it as
its own source. All experiment discipline lives here: only registered
program shapes are emitted, identical concerns copy byte-identical files
across cells, and the composition is recorded per run in
workspace_manifest.json (together with the env's tool manifest snapshot).

Registered shapes (cfg.agent.scaffold):
  react              persistent actor loop        (cell.tm in {A, B})
  tmc                per-firing cron program      (cell.tm == C)
  <baseline name>    task-owned baseline program  (tasks/<task>/agent/baselines/)
  task:<name>        task-authored EXPLORATORY program declared in the
                     task's compose_spec.py TASK_SCAFFOLDS: a checked-in main.py
                     (like a baseline) mounted with the SAME cell-axis-
                     gated learning files as react, plus a generated
                     cell_config.py so cell parameters reach the static
                     program without codegen. Within-task evidence only —
                     never pooled across tasks like the constructor's cells.

The learning trigger is the tlrn axis (harness/config.TLRN_LEVELS): the
renderers compile the resolved {source, params} — never the tag — into a
cadence-neutral `learn(state)` step plus its trigger wiring (crontab entry
+ dispatch branch under tmc; crontab entry + wait-tool wrapper under
react). The learn() body is byte-identical across TM arms for identical
(sig, alg) — pinned by the cross-cell diff test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from harness.config import RunConfig
from harness.task import task_dir

CODE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = CODE_ROOT / "scaffolds" / "runtime"

# alg-config's editable surface (successor of EDIT_SCOPE_CONFIG; the scope
# is baked into the generated program text)
EDIT_SCOPE_CONFIG = ("prompts/", "scan.py")

# per-task constructor declarations: scan-dep
# modules, the candidate-search tool to wrap, the action-wrapper splice,
# per-baseline deps. Declared in tasks/<name>/agent/compose_spec.py —
# compose reads them and never enumerates tasks itself.
_SPEC_DEFAULTS = {"SCAN_DEPS": [], "CANDIDATE_SEARCH": None,
                  "ACTION_WRAPPER": [], "BASELINE_DEPS": {},
                  "TASK_SCAFFOLDS": {}, "VERIFIED_FILES": [],
                  "REPLAY_PARAMS": {}, "REPLAY_MAX_CALLS": 30,
                  "ROLLOUT_FILES": [], "ROLLOUT_MAX_CALLS": 300}

SKILL_ALGS = ("skills", "vskills", "vskills2", "vskills3", "config", "full")  # render skills.md
VERIFIED_ALGS = ("vskills", "vskills2", "vskills3")  # replay-verified adoption
REMINDER_ALGS = ("vskills2", "vskills3")  # learned text = per-wake reminder,
#   the generated react program of _render_react_v2


def _compose_spec(task_name: str) -> dict:
    spec = dict(_SPEC_DEFAULTS)
    path = _agent_dir(task_name) / "compose_spec.py"
    if path.is_file():
        ns: dict = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
             ns)
        spec.update({k: ns[k] for k in spec if k in ns})
    return spec

# sig_self module NOT provisioned by any registered cell: it is the
# compiled body of a future `tlrn: agent` level
_SIG_SELF_EXCLUDE = {"feedback_tools.py"}


def _agent_dir(task_name: str) -> Path:
    return task_dir(task_name) / "agent"


# runtime/ modules that never enter a workspace: program_exec.py is the
# episode-mode subprocess entry (harness/mcp.py runs it from the REPO copy)
# — mounting it would change every workspace's file set for nothing.
_RUNTIME_EXCLUDE = frozenset({"program_exec.py"})


def _runtime_files() -> dict[str, Path]:
    return {f"runtime/{p.name}": p for p in sorted(RUNTIME_SRC.glob("*.py"))
            if p.name not in _RUNTIME_EXCLUDE}


def _sig_self_files(agent: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted((agent / "sig_self").glob("*.py"))
            if p.name not in _SIG_SELF_EXCLUDE}


def _learning_files(cell, agent: Path, files: dict[str, Path]) -> None:
    """The cell-axis-gated learning mounts of a persistent-actor cell —
    shared by react and task-authored scaffolds so the learning stack
    never forks."""
    if cell.sig != "none":  # learn() links outcomes via records.py
        files["records.py"] = agent / "records.py"
    if cell.sig == "self":
        files["pull_feedback.py"] = agent / "pull_feedback.py"
        files.update(_sig_self_files(agent))
    if cell.alg in SKILL_ALGS:
        files["prompts/reflect.md"] = agent / "prompts" / "reflect.md"
    if cell.alg in VERIFIED_ALGS:
        _verified_files(cell, agent, files)


def _verified_files(cell, agent: Path, files: dict[str, Path]) -> None:
    """alg=vskills: the replay evaluator next to the learning stack
    . Declared per task (VERIFIED_FILES); a
    task that declares none cannot run the cell."""
    spec = _compose_spec(agent.parent.name)
    if not spec["VERIFIED_FILES"]:
        raise ValueError(f"alg={cell.alg}: task {agent.parent.name!r} declares "
                         "no VERIFIED_FILES in compose_spec.py")
    for rel in spec["VERIFIED_FILES"]:
        files[rel] = agent / rel
    if cell.alg == "vskills3":  # the day-rollout evaluator on top
        if not spec["ROLLOUT_FILES"]:
            raise ValueError(f"alg=vskills3: task {agent.parent.name!r} "
                             "declares no ROLLOUT_FILES in compose_spec.py")
        for rel in spec["ROLLOUT_FILES"]:
            files[rel] = agent / rel


RENDER_CAP_FIELDS = (  # (cell_config name, AgentSpec field, default)
    ("REFLECT_OWN_HISTORY_CAP", "reflect_own_history_cap", 30),
    ("REFLECT_DIGEST_CAP", "reflect_digest_cap", 40),
    ("REFLECT_EXAMPLES_PER_STRATUM", "reflect_examples_per_stratum", 4),
    ("REPLAY_MAX_WORKERS", "replay_max_workers", 1),
    ("REFLECT_MAX_EDITS", "reflect_max_edits", 3),
    ("REFLECT_MAX_EDIT_WORDS", "reflect_max_edit_words", 100),
    ("REFLECT_DIFF_ROWS", "reflect_diff_rows", 60),
    ("REFLECT_TRANSCRIPT_TOKENS", "reflect_transcript_tokens", 60_000),
    ("REFLECT_TRAJECTORY_WAKES", "reflect_trajectory_wakes", 6),
    ("REFLECT_TRAJECTORY_TOKENS", "reflect_trajectory_tokens", 20_000),
    ("REFLECT_TRAJECTORY_RESULT_CHARS", "reflect_trajectory_result_chars", 300),
    ("BLOCK_TOKENS", "block_tokens", 5000),
)
TRAJECTORY_CAPS = ("REFLECT_TRAJECTORY_WAKES", "REFLECT_TRAJECTORY_TOKENS",
                   "REFLECT_TRAJECTORY_RESULT_CHARS")


def render_caps(cfg: RunConfig) -> list[tuple[str, object]]:
    """The reflection-render / learned-block caps of a cell (run-config
    fields on agent:), emitted into cell_config.py."""
    return [(name, getattr(cfg.agent, field))
            for name, field, _ in RENDER_CAP_FIELDS]


def _cap_overrides(cfg: RunConfig, cell) -> list[str]:
    """Generated-main lines applying NON-default caps to the runtime
    modules (defaults leave pre-knob programs byte-identical)."""
    evaluator = "rollout" if cell.alg == "vskills3" else "replay"
    targets = {"REFLECT_OWN_HISTORY_CAP": "reflect.OWN_HISTORY_CAP",
               "REFLECT_DIGEST_CAP": "reflect.DIGEST_CAP",
               "REFLECT_EXAMPLES_PER_STRATUM": "reflect.K_PER_STRATUM",
               "REPLAY_MAX_WORKERS": f"{evaluator}.MAX_WORKERS",
               "REFLECT_MAX_EDITS": "reflect.MAX_EDITS",
               "REFLECT_MAX_EDIT_WORDS": "reflect.MAX_EDIT_WORDS",
               "REFLECT_DIFF_ROWS": f"{evaluator}.DIFF_ROWS",
               "REFLECT_TRANSCRIPT_TOKENS": "reflect.TRANSCRIPT_TOKENS",
               "REFLECT_TRAJECTORY_WAKES": "trajectory.VERBATIM_WAKES",
               "REFLECT_TRAJECTORY_TOKENS": "trajectory.BUDGET_TOKENS",
               "REFLECT_TRAJECTORY_RESULT_CHARS": "trajectory.RESULT_CHARS"}
    out = []
    for name, field, default in RENDER_CAP_FIELDS:
        v = getattr(cfg.agent, field)
        if v == default:
            continue
        if name == "BLOCK_TOKENS":
            mods = {"memory": ["memory"], "skills": ["skills"],
                    "vskills": ["skills"], "vskills2": ["skills"],
                    "vskills3": ["skills"],
                    "config": ["skills"], "full": ["skills"]}.get(cell.alg, [])
            out += [f"{m}.BLOCK_TOKENS = {v!r}" for m in mods]
        elif name in TRAJECTORY_CAPS:
            if cell.alg in REMINDER_ALGS:
                out.append(f"{targets[name]} = {v!r}")
        elif name == "REFLECT_TRANSCRIPT_TOKENS":
            if cell.alg == "vskills":  # vskills2/3 never compact
                out.append(f"{targets[name]} = {v!r}")
        elif name == "REFLECT_DIFF_ROWS":
            if cell.alg in VERIFIED_ALGS:
                out.append(f"{targets[name]} = {v!r}")
        elif cell.alg in SKILL_ALGS:
            out.append(f"{targets[name]} = {v!r}")
    return out


def replay_constants(cfg: RunConfig) -> list[tuple[str, object]]:
    """The REPLAY_* constants of an alg=vskills cell: task parameters the
    replay must mirror (from the run's task section, else the declared
    default) plus the curation loop caps."""
    spec = _compose_spec(cfg.task_name)
    out = [(f"REPLAY_{name}", cfg.task.get(key, default))
           for name, (key, default) in spec["REPLAY_PARAMS"].items()]
    out.append(("REPLAY_MAX_ITER", cfg.agent.replay_max_iter))
    if cfg.cell.alg == "vskills3":  # decision points = settled days
        out.append(("REPLAY_MAX_ITEMS", cfg.agent.rollout_days))
        out.append(("REPLAY_MAX_CALLS", spec["ROLLOUT_MAX_CALLS"]))
    else:
        out.append(("REPLAY_MAX_ITEMS", cfg.agent.replay_max_items))
        out.append(("REPLAY_MAX_CALLS", spec["REPLAY_MAX_CALLS"]))
    if cfg.cell.alg in REMINDER_ALGS:
        # the settlement constants as one dict (replay_scorer.settle's
        # `params`), plus the run start the acceptance rules need
        from harness.timeutil import iso as _iso
        params = {name.lower(): cfg.task.get(key, default)
                  for name, (key, default) in spec["REPLAY_PARAMS"].items()}
        params["sim_start"] = _iso(cfg.sim_start)
        out.append(("REPLAY_PARAMS", params))
    return out


def workspace_files(cfg: RunConfig) -> dict[str, Path]:
    """workspace-relative path -> source file. Exactly what this cell gets;
    main.py is generated separately (render_main)."""
    files = _scaffold_files(cfg)
    if cfg.agent.seed_skills is not None:
        # EXPLORATORY seeded-skills variant (wait grammar v2 plan): mount
        # an expert-curated file as the initial memory/skills.md. Only a
        # skills-rendering cell reads that file, so anything else is a
        # silent no-op — reject it.
        if cfg.cell.alg not in SKILL_ALGS:
            raise ValueError(
                "agent.seed_skills requires a skills-rendering cell "
                f"(alg in skills/config/full), got {cfg.cell.label}")
        src = _agent_dir(cfg.task_name) / cfg.agent.seed_skills
        if not src.is_file():
            raise ValueError(f"agent.seed_skills: no such file {src}")
        files["memory/skills.md"] = src
    if cfg.agent.seed_monitor_plans is not None:
        src = _agent_dir(cfg.task_name) / cfg.agent.seed_monitor_plans
        if not src.is_file():
            raise ValueError(
                f"agent.seed_monitor_plans: no such file {src}")
        files["memory/seed_monitor_plans.json"] = src
    return files


def _scaffold_files(cfg: RunConfig) -> dict[str, Path]:
    cell, scaffold = cfg.cell, cfg.agent.scaffold
    agent = _agent_dir(cfg.task_name)
    files = _runtime_files()

    if scaffold == "react":
        if cell.tm not in ("A", "B"):
            raise ValueError(f"react requires tm in A/B, got {cell.label}")
        if cell.alg == "vskills":
            raise ValueError("alg=vskills is not wired for the generated "
                             "react program yet (per_market_cron / cron_react / tmc)")
        _learning_files(cell, agent, files)
        return files

    if scaffold.startswith("ext:"):
        raise ValueError(
            f"{scaffold!r} is an external program: the workspace IS the program "
            "directory — run it with --scaffold-src / a detached runner; "
            "the constructor renders nothing for it")

    if scaffold.startswith("task:"):
        name = scaffold[len("task:"):]
        ts = _compose_spec(cfg.task_name)["TASK_SCAFFOLDS"].get(name)
        if ts is None:
            raise ValueError(
                f"task scaffold {name!r} is not declared in "
                f"{cfg.task_name}'s compose_spec.py TASK_SCAFFOLDS")
        tms = tuple(ts.get("tm", ("A", "B")))
        if cell.tm not in tms:
            raise ValueError(f"task scaffold {name!r} supports tm in "
                             f"{tms}, got {cell.label}")
        algs = tuple(ts.get("alg", ("none", "memory", "skills",
                                    "config", "full")))
        if cell.alg not in algs:
            raise ValueError(f"task scaffold {name!r} supports alg in "
                             f"{algs}, got {cell.label}")
        _learning_files(cell, agent, files)
        for dep in ts.get("deps", []):
            files[dep] = agent / dep
        files["main.py"] = agent / ts["main"]
        return files

    if scaffold == "tmc":
        if cell.tm != "C":
            raise ValueError(f"tmc requires tm=C, got {cell.label}")
        if cell.alg in REMINDER_ALGS:
            raise ValueError(f"alg={cell.alg} is wired for the generated react "
                             "program (tmA/tmB) only (vskills2 also "
                             "task:cron_react)")
        files["scan.py"] = agent / "scan.py"
        if cell.sig != "none":  # outcome-record semantics: only with a signal
            files["records.py"] = agent / "records.py"
        files["prompts/decide.md"] = agent / "prompts" / "decide.md"
        if cell.alg in SKILL_ALGS:
            files["prompts/reflect.md"] = agent / "prompts" / "reflect.md"
        if cell.alg == "vskills":
            _verified_files(cell, agent, files)
        if cell.sig == "self":
            files["pull_feedback.py"] = agent / "pull_feedback.py"
            files.update(_sig_self_files(agent))
        for dep in _compose_spec(cfg.task_name)["SCAN_DEPS"]:
            files[dep] = agent / dep
        return files

    baseline = agent / "baselines" / scaffold
    if baseline.is_dir():
        if cell.label != "tmC-tlrnnone-signone-algnone":
            raise ValueError(
                f"baseline {scaffold!r} runs in the default cell only, "
                f"got {cell.label}")
        for p in sorted(baseline.glob("*.py")):
            if p.name != "main.py":
                files[p.name] = p
        files["main.py"] = baseline / "main.py"
        deps = _compose_spec(cfg.task_name)["BASELINE_DEPS"]
        for dep in deps.get(scaffold, []):
            files[dep] = agent / dep
        return files

    raise ValueError(f"unregistered scaffold {scaffold!r} "
                     f"(registered: react, tmc, or a baseline under "
                     f"{baseline.parent})")


# -- generated programs ----------------------------------------------------------------


def render_main(cfg: RunConfig) -> str | None:
    """The generated main.py for react/tmc cells (None for baselines, whose
    program is a checked-in file)."""
    scaffold = cfg.agent.scaffold
    if scaffold == "react":
        return _render_react(cfg)
    if scaffold == "tmc":
        return _render_tmc(cfg)
    return None


def render_cell_config(cfg: RunConfig) -> str | None:
    """cell_config.py for task-authored (task:) and external (ext:)
    programs (None otherwise): their main.py is a checked-in / foreign
    file, so the cell's parameters reach it as a generated sibling module
    instead of compiled-in program text."""
    if not cfg.agent.scaffold.startswith(("task:", "ext:")):
        return None
    cell = cfg.cell
    spec = cell.tlrn_spec or {}
    external = cfg.agent.scaffold.startswith("ext:")
    if cell.tm != "B" or external:
        # (ext: tm=B has no server-side ProgramApp: the env's only wait is
        # sleep, and the program's authored waits reach it through
        # runtime.program.run_program — TM tells the program the
        # discipline, WAIT_TOOL the endpoint)
        wait_tool, wait_args = "sleep", {}
    else:
        # TM-B plumbing waits (coordinator park) run a scaffold-written
        # one-liner in the caller's own jail — never the agent-facing
        # sleep.py; agent-issued waits pass their own path (the wait
        # wrapper only adds waiter_id)
        wait_tool, wait_args = "run_program", {"path": "park.py"}
    return "\n".join([
        '"""Generated by scaffolds/compose.py: this cell\'s parameters."""',
        "",
        f'TM = "{cell.tm}"',
        f'SIG = "{cell.sig}"',
        f'ALG = "{cell.alg}"',
        f'WAIT_TOOL = "{wait_tool}"',
        f"WAIT_ARGS = {wait_args!r}  # plumbing waits (agent waits pass "
        "their own args)",
        f"LEARN_CRON = {spec.get('cron')!r}",
        f"CONTEXT_TOKENS = {cfg.agent.context_tokens}",
        # the run's task section (minus name), so a task-authored main can
        # derive program constants from task params instead of duplicating
        # them (e.g. daily_reddit_digest's act cron from digest_hour_utc)
        f"TASK_PARAMS = {cfg.task_params!r}",
        *[f"{k} = {v!r}" for k, v in render_caps(cfg)],
        *([f"{k} = {v!r}" for k, v in replay_constants(cfg)]
          if cell.alg in VERIFIED_ALGS else []),
        "",
    ])


def _header(cfg: RunConfig) -> list[str]:
    return [
        f'"""Generated by scaffolds/compose.py for task {cfg.task_name}.',
        "",
        "This file IS the program: plain code, no hidden framework. Every",
        "line is live for this cell — edit freely where your privileges",
        'allow (see INSTRUCTION.md)."""',
    ]


def _reflect_call(cell) -> str | None:
    """The reflection line of the learn step (None for alg=memory). The
    whole records module is passed: reflect reads stratum_of always and
    the optional is_own / render_context hooks where the task defines
    them."""
    return {
        "memory": None,
        "skills": "reflect.reflect(env, state, records)",
        "vskills": ("reflect.reflect_verified(\n"
                    "        env, state, records, replay, None,\n"
                    "        n_iter=REPLAY_MAX_ITER, max_items=REPLAY_MAX_ITEMS,\n"
                    "        claim_window_hours=REPLAY_CLAIM_WINDOW_HOURS,\n"
                    "        price_delay_minutes=REPLAY_PRICE_DELAY_MINUTES,\n"
                    "        context_tokens=0, max_calls=REPLAY_MAX_CALLS)"),
        "vskills2": ("reflect.reflect_verified2(\n"
                     "        env, state, records, replay, make_episode,\n"
                     "        n_iter=REPLAY_MAX_ITER, max_items=REPLAY_MAX_ITEMS,\n"
                     "        params=REPLAY_PARAMS, context_tokens=CONTEXT_TOKENS,\n"
                     "        max_calls=REPLAY_MAX_CALLS)"),
        "vskills3": ("reflect.reflect_verified2(\n"
                     "        env, state, records, rollout, make_episode,\n"
                     "        n_iter=REPLAY_MAX_ITER, max_items=REPLAY_MAX_ITEMS,\n"
                     "        params=REPLAY_PARAMS, context_tokens=CONTEXT_TOKENS,\n"
                     "        max_calls=REPLAY_MAX_CALLS,\n"
                     "        system_prompt=reflect.VERIFIED3_SYSTEM,\n"
                     "        template=reflect.VERIFIED3_TEMPLATE)"),
        "config": ("reflect.reflect(env, state, records,\n"
                   f"                    edit_scope={EDIT_SCOPE_CONFIG!r})"),
        "full": ("reflect.reflect(env, state, records,\n"
                 "                    edit_scope=reflect.EDIT_ALL)"),
    }[cell.alg]


def _render_learn(cell) -> list[str]:
    """The cadence-neutral learn step: sig pull -> record
    append -> reflection -> state save. Byte-identical across TM arms for
    identical (sig, alg)."""
    lines = ["", "", "def learn(state):"]
    if cell.sig == "oracle":
        lines.append("    memory.pull_oracle(env, state, records.action_for)")
    else:  # self
        lines += [
            "    t = iso(env.now())",
            "    for out in pull_feedback.compile(env, state):",
            "        action = records.action_for(out, state)",
            '        memory.append_record(t=t, src="self", outcome=out,',
            "                             action=action)",
            '        trace.log(t, "feedback", src="self", outcome=out)',
        ]
    call = _reflect_call(cell)
    if call:
        lines.append(f"    {call}")
    lines.append("    save_state(state)")
    return lines


def _learn_crontab(cell) -> str:
    spec = cell.tlrn_spec
    return ('env.call("set_crontab", entries=['
            f'{{"id": "learn", "cron_expr": "{spec["cron"]}"}}])')


def _render_react(cfg: RunConfig) -> str:
    cell = cfg.cell
    # TM-A waits with the blind sleep tool; TM-B holds no sleep tool at
    # all — waiting IS run_program (the jail-seeded sleep.py is the
    # blind-wait floor), and the rendered INSTRUCTION.md carries the
    # authored-program skill appendix (harness/authored_skill.md)
    wait_tool = "sleep" if cell.tm == "A" else "run_program"
    if cell.alg in REMINDER_ALGS:
        return _render_react_v2(cfg, wait_tool)
    learning = cell.alg != "none"  # tlrn/sig validators guarantee coupling
    lines = _header(cfg) + [
        "",
        "import json",
        "import os",
        "",
    ]
    runtime_mods = ["agent"]
    if learning:
        runtime_mods.append("memory")
    if cell.alg == "skills":
        runtime_mods += ["reflect", "skills"]
    if cell.sig == "self":
        runtime_mods.append("trace")
    lines.append(f"from runtime import {', '.join(sorted(runtime_mods))}")
    lines.append("from runtime.env_client import Env"
                 + (", iso" if cell.sig == "self" else ""))
    if learning:
        lines.append("from runtime.state import load_state, save_state")
        lines.append("")
        lines.append("import records")
        if cell.sig == "self":
            lines.append("import pull_feedback")
    lines += [
        "",
        "env = Env()",
    ]
    overrides = _cap_overrides(cfg, cell)
    if overrides:
        lines += [""] + overrides
    if learning:
        lines.append("state = load_state()")
        if cell.alg == "memory":
            lines.append("memory.bind(records, env, state)  # formatted "
                         "learned block: task semantics from records.py")
        lines += _render_learn(cell)
        lines += [
            "",
            _learn_crontab(cell),
        ]
    lines += [
        "",
        "tools = agent.env_tools(env)  # everything the manifest provisions",
        'for _name in ("get_crontab", "set_crontab", "run_at"):',
        "    tools.pop(_name, None)  # acting stays agent-owned: the actor's",
        "    # wake is its wait tool, never a schedule it maintains",
    ]
    spec = _compose_spec(cfg.task_name)
    if learning:
        lines += [
            "",
            f'_wait_inner = tools["{wait_tool}"]["fn"]',
            "",
            "",
            "def _wait(args):  # the actor's wait tool: learning runs behind",
            '    state["last_wait"] = args  # standing posture, read by learn()',
            "    while True:   # it, invisibly, at the armed wait's expense",
            "        wake = _wait_inner(args)",
            "        trig = wake.get(\"trigger\") or {}",
            '        if wake.get("woke_for") == "trigger" '
            'and trig.get("id") == "learn":',
            "            learn(state)  # same sim instant; zero sim time",
            "            continue      # re-issue the remaining wait",
            "        return wake",
            "",
            "",
            f'tools["{wait_tool}"] = {{**tools["{wait_tool}"], "fn": _wait}}',
        ]
    if learning and spec["CANDIDATE_SEARCH"]:
        search = spec["CANDIDATE_SEARCH"]
        lines += [
            "",
            f'_search_inner = tools["{search}"]["fn"]',
            "",
            "",
            "def _search(args):  # register everything observed: id -> title",
            "    result = _search_inner(args)  # for the reflection render;",
            "    records.register_candidates(state, result)  # sig-self reads",
            "    return result  # the same registry as its candidate pool",
            "",
            "",
            f'tools["{search}"] = {{**tools["{search}"], "fn": _search}}',
        ]
    if learning and spec["ACTION_WRAPPER"]:
        # task-declared splice: registers each action in state["actions"]
        # so the task's records.action_for links it to the settled outcome
        lines += spec["ACTION_WRAPPER"]
    block_fn = {"none": None, "memory": "memory.render_block",
                "skills": "skills.render_block"}[cell.alg]
    lines += [
        "",
        'actor = agent.Agent(env, "actor", tools,',
        '                    transcript="logs/react_transcript.jsonl",',
    ]
    if block_fn:
        lines.append(f"                    block_fn={block_fn},")
    if cfg.agent.context_tokens != 60_000:  # default stays byte-identical
        lines.append(f'                    wait_tool="{wait_tool}",')
        lines.append(f"                    context_tokens="
                     f"{cfg.agent.context_tokens})")
    else:
        lines.append(f'                    wait_tool="{wait_tool}")')
    if learning:
        lines += [
            'trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))',
            'if trigger.get("id") == "learn":  # crash recovery: re-invoked',
            "    learn(state)                  # at the learn firing",
            "actor.wake(trigger)",
        ]
    else:
        lines.append(
            'actor.wake(json.loads(os.environ.get("ENV_TRIGGER", "{}")))')
    lines += [
        "",
        "while actor.turn():  # sim time passes inside the wait tool",
        "    pass",
        "",
    ]
    return "\n".join(lines)


def _render_react_v2(cfg: RunConfig, wait_tool: str) -> str:
    """The generated react program of an alg=vskills2 / vskills3 cell:
    the same actor as every other react cell — same tools, same wait
    wrapper, same transcript — with the learned text delivered as a
    per-wake REMINDER message (no system-prompt block), ONE `build_agent`
    shared by the live actor and the replay episodes, and learn() = the
    trajectory-aware verified curation (v3: verified on day rollouts,
    the `rollout` evaluator). Other cells' programs are rendered by
    _render_react and stay byte-identical."""
    cell = cfg.cell
    spec = _compose_spec(cfg.task_name)
    if spec["CANDIDATE_SEARCH"] or spec["ACTION_WRAPPER"]:
        raise ValueError(f"alg={cell.alg}: the generated react program does "
                         "not splice CANDIDATE_SEARCH / ACTION_WRAPPER "
                         f"(task {cfg.task_name!r} declares them)")
    if cell.sig != "oracle":
        raise ValueError(f"alg={cell.alg} requires sig=oracle")
    lines = _header(cfg) + [
        "",
        "import json",
        "import os",
        "",
        "from runtime import agent, memory, reflect, skills, trajectory",
        "from runtime.env_client import Env",
        "from runtime.state import load_state, save_state",
        "",
        "import records",
        "import replay",
        *(["import rollout"] if cell.alg == "vskills3" else []),
        "",
    ]
    for k, v in replay_constants(cfg):
        lines.append(f"{k} = {v!r}")
    lines.append(f"CONTEXT_TOKENS = {cfg.agent.context_tokens!r}")
    lines += ["", "env = Env()"]
    overrides = _cap_overrides(cfg, cell)
    if overrides:
        lines += [""] + overrides
    lines += [
        "state = load_state()",
        "memory.bind(records, env, state)  # formatted learned block: the "
        "curator's evidence (task semantics from records.py)",
    ]
    lines += _render_learn(cell)
    lines += ["", _learn_crontab(cell)]
    lines += [
        "",
        "",
        "def _tools(env_):",
        "    tools = agent.env_tools(env_)  # everything the manifest provisions",
        '    for _name in ("get_crontab", "set_crontab", "run_at"):',
        "        tools.pop(_name, None)  # acting stays agent-owned: the actor's",
        "        # wake is its wait tool, never a schedule it maintains",
        "    return tools",
        "",
        "",
        "def _wait(inner, args):  # the actor's wait tool: learning runs behind",
        '    state["last_wait"] = args  # standing posture, read by learn()',
        "    while True:   # it, invisibly, at the armed wait's expense",
        "        wake = inner(args)",
        "        trig = wake.get(\"trigger\") or {}",
        '        if wake.get("woke_for") == "trigger" '
        'and trig.get("id") == "learn":',
        "            learn(state)  # same sim instant; zero sim time",
        "            continue      # re-issue the remaining wait",
        "        return wake",
        "",
        "",
        "def build_agent(env_=env, reminder_fn=None,",
        '                transcript="logs/react_transcript.jsonl", label=None):',
        '    """The ONE constructor of the actor, live and replayed alike:',
        "    a replay passes its frozen env (its wait tool ends the episode",
        '    there) and the candidate reminder."""',
        "    tools = _tools(env_)",
        "    if env_ is env:  # live: the learn trigger fires behind the wait",
        f'        inner = tools["{wait_tool}"]["fn"]',
        f'        tools["{wait_tool}"] = {{**tools["{wait_tool}"],',
        '                               "fn": lambda args: _wait(inner, args)}',
        '    return agent.Agent(env_, label or "actor", tools,',
        "                       transcript=transcript,",
        "                       reminder_fn=reminder_fn,",
        f'                       wait_tool="{wait_tool}",',
        "                       context_tokens=CONTEXT_TOKENS)",
        "",
        "",
        "def make_episode(replay_env, agent_name, reminder_fn, transcript,",
        "                 scratch_state, label):",
        '    """Episode factory for replay.replay_wake."""',
        "    return build_agent(replay_env, reminder_fn, transcript, label)",
        "",
        "",
        "actor = build_agent(reminder_fn=skills.render_reminder)",
        'trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))',
        'if trigger.get("id") == "learn":  # crash recovery: re-invoked',
        "    learn(state)                  # at the learn firing",
        "actor.wake(trigger)",
        "",
        "while actor.turn():  # sim time passes inside the wait tool",
        "    pass",
        "",
    ]
    return "\n".join(lines)


def _render_tmc(cfg: RunConfig) -> str:
    cell = cfg.cell
    learning = cell.alg != "none"
    lines = _header(cfg) + [
        "",
        "import json",
        "import os",
        "",
    ]
    runtime_mods = []
    if learning:
        runtime_mods.append("memory")
    if cell.alg in SKILL_ALGS:
        runtime_mods += ["reflect", "skills"]
    if cell.sig == "self":
        runtime_mods.append("trace")
    if runtime_mods:
        lines.append(f"from runtime import {', '.join(sorted(runtime_mods))}")
    lines.append("from runtime.env_client import Env"
                 + (", iso" if cell.sig == "self" else ""))
    lines.append("from runtime.state import load_state, save_state")
    lines.append("")
    lines.append("import scan")
    if cell.sig != "none":
        lines.append("import records")
    if cell.alg == "vskills":
        lines.append("import replay")
        lines.append("")
        for k, v in replay_constants(cfg):
            lines.append(f"{k} = {v!r}")
    if cell.sig == "self":
        lines.append("import pull_feedback")
    lines += [
        "",
        "env = Env()",
        'trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))',
    ]
    overrides = _cap_overrides(cfg, cell)
    if overrides:
        lines += [""] + overrides
    if learning:
        lines += _render_learn(cell)
    block = {"none": '""', "memory": "memory.render_block()",
             "skills": "skills.render_block()",
             "vskills": "skills.render_block()",
             "config": "skills.render_block()",
             "full": "skills.render_block()"}[cell.alg]
    entries = ['{"id": "scan", "cron_expr": scan.SCAN_CRON}']
    if learning:
        spec = cell.tlrn_spec
        entries.append(f'{{"id": "learn", "cron_expr": "{spec["cron"]}"}}')
    lines += [
        "",
        'env.call("set_crontab", entries=[',
        *[f"    {e}," for e in entries],
        "])",
        "",
        "state = load_state()",
    ]
    if cell.alg == "memory":
        lines.append("memory.bind(records, env, state)  # formatted "
                     "learned block: task semantics from records.py")
    if not learning:
        lines += [
            f"scan.scan(env, state, {block})",
        ]
    else:
        lines += [
            'if trigger.get("id") == "learn":',
            "    learn(state)",
            "else:  # scan / bootstrap / fallback / crash_recovery",
            f"    scan.scan(env, state, {block})",
        ]
    lines += [
        "save_state(state)",
        "",
    ]
    return "\n".join(lines)


# -- build -----------------------------------------------------------------------------


def build_workspace(cfg: RunConfig, workspace: Path) -> list[str]:
    """Materialize the workspace; returns the sorted file list for
    workspace_manifest.json."""
    files = workspace_files(cfg)
    main = render_main(cfg)
    if main is None and "main.py" not in files:
        raise ValueError(f"no program for scaffold {cfg.agent.scaffold!r}")
    workspace.mkdir(parents=True, exist_ok=True)
    for rel, src in files.items():
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    if main is not None:
        (workspace / "main.py").write_text(main, encoding="utf-8")
    cell_config = render_cell_config(cfg)
    if cell_config is not None:
        (workspace / "cell_config.py").write_text(cell_config,
                                                  encoding="utf-8")
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    out = sorted(files) + (["main.py"] if main is not None else []) \
        + (["cell_config.py"] if cell_config is not None else [])
    return sorted(set(out))
