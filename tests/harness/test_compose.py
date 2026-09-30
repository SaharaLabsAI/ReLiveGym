"""The constructor: registered
configs only, exact file sets per cell (every provisioned file is live),
parseable branch-free generated programs, and cross-cell/cross-task
discipline — deltas between generated programs are exactly the declared
factors, and the learn step is byte-identical across TM arms."""

from __future__ import annotations

import ast
from datetime import datetime, timezone

import pytest

from harness.config import RunConfig
from scaffolds.compose import (
    build_workspace,
    render_cell_config,
    render_main,
    workspace_files,
)

UTC = timezone.utc

RUNTIME = {"runtime/__init__.py", "runtime/actor.py", "runtime/agent.py",
           "runtime/env_client.py", "runtime/llm_client.py",
           "runtime/memory.py", "runtime/program.py", "runtime/reflect.py",
           "runtime/replay_env.py", "runtime/skills.py", "runtime/state.py",
           "runtime/tokens.py", "runtime/trace.py", "runtime/trajectory.py",
           "runtime/view.py"}


def cfg_for(task: str, cell: dict, scaffold: str) -> RunConfig:
    # the task section is opaque to compose (only the name is read), so a
    # bare name suffices for constructor tests
    return RunConfig(run_id="x", task={"name": task},
                     sim_start=datetime(2021, 6, 1, tzinfo=UTC),
                     sim_end=datetime(2021, 6, 8, tzinfo=UTC),
                     cell=cell, agent={"scaffold": scaffold})


REGISTERED = [
    ({"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"}, "react"),
    ({"tm": "B", "tlrn": "none", "sig": "none", "alg": "none"}, "react"),
    ({"tm": "B", "tlrn": "daily", "sig": "oracle", "alg": "memory"}, "react"),
    ({"tm": "B", "tlrn": "daily", "sig": "oracle", "alg": "skills"}, "react"),
    ({"tm": "B", "tlrn": "daily", "sig": "self", "alg": "skills"}, "react"),
    ({"tm": "A", "tlrn": "daily", "sig": "self", "alg": "memory"}, "react"),
    ({"tm": "C", "tlrn": "none", "sig": "none", "alg": "none"}, "tmc"),
    ({"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "memory"}, "tmc"),
    ({"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "skills"}, "tmc"),
    ({"tm": "C", "tlrn": "daily", "sig": "self", "alg": "skills"}, "tmc"),
    ({"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "config"}, "tmc"),
    ({"tm": "C", "tlrn": "daily", "sig": "self", "alg": "full"}, "tmc"),
]

# breakout_news_pm is the full-stack constructor fixture (records, scan,
# sig_self, prompts); reddit/crypto exercise the no-agent-stack renders
FULL_TASK = "breakout_news_pm"
TASKS = ["reddit_ai_popularity", "crypto_price_consistency", FULL_TASK]


@pytest.mark.parametrize("cell,scaffold", REGISTERED,
                         ids=[f"{c['tm']}-{c['tlrn']}-{c['sig']}-{c['alg']}-{s}"
                              for c, s in REGISTERED])
def test_registered_cells_compose_and_parse(cell, scaffold):
    cfg = cfg_for(FULL_TASK, cell, scaffold)
    files = workspace_files(cfg)
    assert RUNTIME <= set(files)
    main = render_main(cfg)
    assert main is not None
    ast.parse(main)  # branch-free by construction; at least it must parse
    # provisioning invariants: a file exists iff its concern is provisioned
    has_sig_self = "feedback_fn.py" in files
    assert has_sig_self == (cell["sig"] == "self")
    assert "feedback_tools.py" not in files  # tlrn:agent body, unprovisioned
    assert ("records.py" in files) == (cell["sig"] != "none")
    assert ("prompts/reflect.md" in files) == (
        cell["alg"] in ("skills", "config", "full"))
    # tlrn compilation: learn code exists iff the cell has a learn trigger
    assert ("def learn(state):" in main) == (cell["tlrn"] == "daily")
    assert ('"id": "learn"' in main) == (cell["tlrn"] == "daily")
    if cell["tlrn"] == "daily":
        assert "30 0 * * *" in main  # the registered cron, compiled in
    # nothing about other cells leaks into the program text
    if cell["alg"] == "none":
        assert "memory" not in main and "skills" not in main
    if cell["sig"] != "oracle":
        assert "pull_oracle" not in main
    if scaffold == "react":
        # the only wait tool is sleep (no condition grammar); wait_until
        # must never appear
        assert "wait_until" not in main
        # the actor owns no learning tools
        assert "update_skills" not in main
        assert "skills_tools" not in main and "memory_tools" not in main
        assert "wrap_feedback" not in main and "get_feedback" not in main
        # registration wrappers: every
        # learning cell of a registered task records what it observed and
        # what it claimed; baselines get none of it
        learning = cell["sig"] != "none"
        assert ("register_candidates" in main) == learning
        assert ("_notify" in main) == learning
        assert ('state["last_wait"]' in main) == learning


def test_exact_file_set_bridging_cell():
    cfg = cfg_for(FULL_TASK,
                  {"tm": "C", "tlrn": "daily", "sig": "self",
                   "alg": "skills"}, "tmc")
    assert set(workspace_files(cfg)) == RUNTIME | {
        "scan.py", "records.py", "pull_feedback.py", "feedback_fn.py",
        "breakout_stats.py", "question_keywords.py",
        "prompts/decide.md", "prompts/reflect.md"}


def test_generated_program_is_task_agnostic():
    """Same cell, different task -> byte-identical program below the
    header, modulo the task-keyed candidate-search wrapper (a declared
    per-task provision, like scan deps). Task specifics live in
    provisioned files, never in the generated trunk."""
    for cell, scaffold in REGISTERED:
        mains = {t: render_main(cfg_for(t, cell, scaffold)) for t in TASKS}
        strip = lambda s: s.split('"""', 2)[2]  # drop the generated header
        a = strip(mains["reddit_ai_popularity"])
        b = strip(mains["crypto_price_consistency"])
        assert a == b, f"{cell} {scaffold}"
        c = strip(mains[FULL_TASK])
        if not (scaffold == "react" and cell["sig"] != "none"):
            assert a == c, f"{cell} {scaffold}"
        else:  # the only deltas are the task-keyed registration wrappers
            # (candidate-search, action registration)
            a_lines = a.splitlines()
            extra = [l for l in c.splitlines() if l not in a_lines]
            assert extra, f"{cell} {scaffold}"
            allowed = ("_search", "register_candidates", "result",
                       "_action", "_notify", "reg", "nid",
                       '"actions"', "save_state", '"did"', '"title"',
                       '"published"', '"market_id"', '"direction"',
                       '"at"')
            assert all(any(tok in l for tok in allowed)
                       for l in extra), extra


def test_learn_step_byte_identical_across_tm_arms():
    """Same (sig, alg): the generated learn() is byte-identical whether it
    runs behind a react wait tool or a tmc dispatch branch."""
    def learn_block(main: str) -> str:
        lines = main.splitlines()
        i = lines.index("def learn(state):")
        j = i + 1
        while j < len(lines) and (not lines[j] or lines[j].startswith(" ")):
            j += 1
        return "\n".join(lines[i:j]).rstrip()

    for sig, alg in [("oracle", "memory"), ("oracle", "skills"),
                     ("self", "skills")]:
        react = render_main(cfg_for(
            FULL_TASK,
            {"tm": "B", "tlrn": "daily", "sig": sig, "alg": alg}, "react"))
        tmc = render_main(cfg_for(
            FULL_TASK,
            {"tm": "C", "tlrn": "daily", "sig": sig, "alg": alg}, "tmc"))
        assert learn_block(react) == learn_block(tmc), (sig, alg)


def test_react_ab_programs_differ_only_in_wait_tool():
    """TM-A vs TM-B (same tlrn/sig/alg): the generated programs are
    byte-identical after normalizing the wait-tool name — TM-B holds no
    sleep tool (waiting is run_program), and that name is the ONLY
    program-text difference; everything else about the contrast is env
    provisioning + the INSTRUCTION.md skill appendix."""
    for cell in [{"tlrn": "none", "sig": "none", "alg": "none"},
                 {"tlrn": "daily", "sig": "oracle", "alg": "memory"}]:
        a = render_main(cfg_for(FULL_TASK, {"tm": "A", **cell}, "react"))
        b = render_main(cfg_for(FULL_TASK, {"tm": "B", **cell}, "react"))
        assert a != b
        assert b.replace('"run_program"', '"sleep"') == a


def test_memory_cells_bind_the_formatted_render():
    """alg=memory cells wire memory.bind(records, env, state) after the
    state loads in both generated paths ; the
    old memory.LEGEND splice is gone; non-memory cells bind nothing."""
    for cell, scaffold in [
            ({"tm": "B", "tlrn": "daily", "sig": "oracle",
              "alg": "memory"}, "react"),
            ({"tm": "C", "tlrn": "daily", "sig": "oracle",
              "alg": "memory"}, "tmc")]:
        main = render_main(cfg_for(FULL_TASK, cell, scaffold))
        assert "memory.bind(records, env, state)" in main
        assert "memory.LEGEND" not in main
        assert main.index("state = load_state()") \
            < main.index("memory.bind(records, env, state)")
    for cell, scaffold in [
            ({"tm": "B", "tlrn": "daily", "sig": "oracle",
              "alg": "skills"}, "react"),
            ({"tm": "C", "tlrn": "daily", "sig": "oracle",
              "alg": "skills"}, "tmc")]:
        assert "memory.bind" not in render_main(
            cfg_for(FULL_TASK, cell, scaffold))


def test_tmc_learning_dispatch_and_no_per_firing_pull():
    """TM-C learning cells: the learn branch exists, and the scan branch
    does not pull feedback."""
    main = render_main(cfg_for(
        FULL_TASK,
        {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "memory"},
        "tmc"))
    assert 'if trigger.get("id") == "learn":' in main
    assert "pull_signal" not in main
    assert main.count("pull_oracle") == 1  # exactly once, inside learn()


def test_unregistered_shapes_rejected():
    with pytest.raises(ValueError, match="react requires"):
        workspace_files(cfg_for(FULL_TASK,
                                {"tm": "C", "tlrn": "none", "sig": "none",
                                 "alg": "none"}, "react"))
    with pytest.raises(ValueError, match="tmc requires"):
        workspace_files(cfg_for(FULL_TASK,
                                {"tm": "A", "tlrn": "none", "sig": "none",
                                 "alg": "none"}, "tmc"))
    with pytest.raises(ValueError, match="unregistered scaffold"):
        workspace_files(cfg_for(FULL_TASK,
                                {"tm": "C", "tlrn": "none", "sig": "none",
                                 "alg": "none"}, "nonesuch"))
    with pytest.raises(ValueError, match="default cell only"):
        workspace_files(cfg_for("reddit_ai_popularity",
                                {"tm": "C", "tlrn": "daily", "sig": "oracle",
                                 "alg": "memory"}, "cascade_forecaster"))


# -- seeded expert skills (EXPLORATORY; wait grammar v2 plan) --------------------------


def test_seed_skills_mounts_file_as_initial_skills_md(tmp_path):
    for scaffold in ("react", "task:per_market"):
        cfg = cfg_for(FULL_TASK,
                      {"tm": "B", "tlrn": "daily", "sig": "oracle",
                       "alg": "skills"}, scaffold)
        cfg.agent.seed_skills = "expert_skills_v1.md"
        files = workspace_files(cfg)
        assert files["memory/skills.md"].name == "expert_skills_v1.md"
        ws = tmp_path / scaffold.replace(":", "_")
        manifest = build_workspace(cfg, ws)
        assert "memory/skills.md" in manifest
        assert (ws / "memory" / "skills.md").read_bytes() == \
            files["memory/skills.md"].read_bytes()


def test_seed_skills_requires_skills_rendering_cell():
    for cell in ({"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"},
                 {"tm": "B", "tlrn": "daily", "sig": "oracle",
                  "alg": "memory"}):  # memory renders records, not skills.md
        cfg = cfg_for(FULL_TASK, cell, "react")
        cfg.agent.seed_skills = "expert_skills_v1.md"
        with pytest.raises(ValueError, match="skills-rendering"):
            workspace_files(cfg)


def test_seed_skills_missing_file_rejected():
    cfg = cfg_for(FULL_TASK,
                  {"tm": "B", "tlrn": "daily", "sig": "oracle",
                   "alg": "skills"}, "react")
    cfg.agent.seed_skills = "nonesuch.md"
    with pytest.raises(ValueError, match="no such file"):
        workspace_files(cfg)


def test_expert_skills_v1_fits_block_budget_unmodified():
    # the checked-in expert block must render verbatim: if edits push it
    # past the skill budget, truncation would silently cut the playbook
    from harness.task import task_dir
    from scaffolds.runtime.tokens import BUDGET_TOKENS, truncate

    text = (task_dir(FULL_TASK) / "agent" / "expert_skills_v1.md"
            ).read_text(encoding="utf-8").strip()
    assert truncate(text, BUDGET_TOKENS) == text


# -- context_tokens knob (wgv3 ctx20k probe) -------------------------------------------


def test_context_tokens_default_leaves_react_program_unchanged():
    cfg = cfg_for(FULL_TASK,
                  {"tm": "B", "tlrn": "daily", "sig": "oracle",
                   "alg": "skills"}, "react")
    assert "context_tokens" not in render_main(cfg)


def test_context_tokens_override_reaches_generated_programs():
    cfg = cfg_for(FULL_TASK,
                  {"tm": "B", "tlrn": "daily", "sig": "oracle",
                   "alg": "skills"}, "react")
    cfg.agent.context_tokens = 20_000
    assert "context_tokens=20000" in render_main(cfg)

    pm = cfg_for(FULL_TASK,
                 {"tm": "B", "tlrn": "daily", "sig": "oracle",
                  "alg": "skills"}, "task:per_market")
    assert "CONTEXT_TOKENS = 60000" in render_cell_config(pm)
    pm.agent.context_tokens = 20_000
    assert "CONTEXT_TOKENS = 20000" in render_cell_config(pm)
