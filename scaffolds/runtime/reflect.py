"""Deterministic reflection handler. Program library.

`reflect` renders the stratified interval view of new memory records, makes
one LLM call against prompts/reflect.md, and overwrites the skill block.
With an `edit_scope`, a second step may edit workspace files (alg-config /
alg-full): the scope is baked in by the constructor — a program reading its
own source sees exactly the privileges it has.

The records module passed in provides the task semantics (this library
knows no task vocabulary — everything task-specific enters through
these hooks): `stratum_of` (required), and
optionally `is_own(record)` (records that are the agent's own actions
even when no action linked), `render_context(env, state)` (an opaque
context dict; only its "extra_slots" entry — slot name -> rendered text
for task-defined template slots — is read here, the rest feeds the
task's own hooks), `enrich(record, ctx)` (resolve bare ids into meaning
on a render copy, in place), `entity_of(record)` (ledger key for the
per-entity outcome table), `digest_line(record)` (one line of the
cumulative outcome digest), `describe_wait(wait)` (billing note for the
standing-wait line of the cost report). All template slots are optional
— a template ignores inputs it has no slot for. Slots: ${run_header}
${instruction} ${skills} ${cum_counts} ${entity_ledger} ${own_history}
${outcome_digest} ${distribution} ${own_actions} ${examples} ${costs},
plus whatever extra_slots the task defines.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import llm_client, memory, skills, trace
from .env_client import EnvError, iso
from .view import enrich, entity_ledger, status_counts  # noqa: F401 —
# hoisted to view.py (shared with the formatted memory render); re-exported
# here so reflection callers and templates keep their import surface

K_PER_STRATUM = 4  # exemplars per stratum in the reflection render

EDIT_ALL = "all"  # edit_scope value for alg-full: anything in the workspace


def _jsonl(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, separators=(",", ":")) for r in records)


def _is_own(r: dict, is_own=None) -> bool:
    from . import view
    return view.is_own(r, is_own)


def reflection_view(records: list[dict], stratum_of,
                    is_own=None) -> tuple[str, str, str]:
    """The stratified interval view: per-status counts,
    all own-action records, and K_PER_STRATUM seeded exemplars per stratum
    of the rest. Deterministic given the records. Own = linked action or
    the task's is_own predicate — own records all render,
    never sampled away."""
    import random

    own = [r for r in records if _is_own(r, is_own)]
    strata: dict[str, list[dict]] = {}
    for r in records:
        if not _is_own(r, is_own):
            strata.setdefault(stratum_of(r), []).append(r)
    distribution = status_counts(records, stratum_of, is_own)
    rng = random.Random(records[-1]["t"] if records else "")
    examples: list[dict] = []
    for name in sorted(strata):
        rs = strata[name]
        examples += rs if len(rs) <= K_PER_STRATUM \
            else rng.sample(rs, K_PER_STRATUM)
    return distribution, _jsonl(own), _jsonl(examples)


def cost_report(env, state: dict, describe_wait=None) -> str:
    """The ${costs} section: spend since the last
    reflection by category, cumulative spend against the run budget, and
    the current standing posture — what running the program costs, fed to
    the same step that decides what deserves watching. Outcomes are
    metric feedback, not dollars, and never appear here."""
    try:
        costs = env.call("get_costs")
    except EnvError:
        return "(spend data unavailable)"
    cur = costs.get("spend_by_type") or {}
    last = state.get("last_costs") or {}
    delta = {k: round(v - last.get(k, 0.0), 8) for k, v in sorted(cur.items())}
    delta = {k: v for k, v in delta.items() if v > 0}
    state["last_costs"] = cur
    spend = costs.get("spend_total", 0.0)
    lines = ["spend since last reflection: "
             + (", ".join(f"{k} ${v:.2f}" for k, v in delta.items())
                if delta else "none"),
             f"cumulative spend: ${spend:.2f} (of the run budget — see "
             f"INSTRUCTION.md)"]
    wait = state.get("last_wait")
    if wait:
        billing = describe_wait(wait) if describe_wait else ""
        lines.append("current standing wait: "
                     + json.dumps(wait, separators=(",", ":")) + billing)
    try:
        crontab = env.call("get_crontab")
        if crontab:
            lines.append("recurring schedule: "
                         + json.dumps(crontab, separators=(",", ":")))
    except EnvError:
        pass
    return "\n".join(lines)


OWN_HISTORY_CAP = 30  # newest own-action records rendered per reflection
DIGEST_CAP = 40  # newest non-own outcomes in the cumulative digest


def prepare_reflection(env, state: dict, records_mod) -> dict:
    """Load and summarize one reflection firing without calling the LLM.

    The returned snapshot is immutable by convention and can be shared by
    multiple task-specific reviewers.  In particular, ``get_costs`` is called
    exactly once and its checkpoint advances exactly where the plain
    ``reflect`` implementation advances it.
    """
    t = iso(env.now())
    since = state.get("last_reflection")
    stratum_of = records_mod.stratum_of
    is_own = getattr(records_mod, "is_own", None)
    ctx_fn = getattr(records_mod, "render_context", None)
    ctx = ctx_fn(env, state) if ctx_fn else {}
    all_records = enrich(memory.read_records(), ctx,
                         getattr(records_mod, "enrich", None))
    records = [r for r in all_records if since is None or r["t"] > since]
    n_firing = state.get("reflection_count", 0) + 1
    state["reflection_count"] = n_firing
    distribution, own_actions, examples = reflection_view(
        records, stratum_of, is_own)
    own_all = [r for r in all_records if _is_own(r, is_own)]
    own_history = _jsonl(own_all[-OWN_HISTORY_CAP:])
    if len(own_all) > OWN_HISTORY_CAP:
        own_history = (f"(oldest {len(own_all) - OWN_HISTORY_CAP} of "
                       f"{len(own_all)} omitted; counts above cover them)\n"
                       + own_history)
    digest_fn = getattr(records_mod, "digest_line", None)
    others_all = [r for r in all_records if not _is_own(r, is_own)]
    if digest_fn:
        digest = "\n".join(digest_fn(r) for r in others_all[-DIGEST_CAP:]) \
            or "(none yet)"
        if len(others_all) > DIGEST_CAP:
            digest = (f"(oldest {len(others_all) - DIGEST_CAP} of "
                      f"{len(others_all)} omitted; counts above cover "
                      f"them)\n" + digest)
    else:
        digest = "(not provided)"
    header = (f"Reflection #{n_firing} — sim time {t} — {len(records)} "
              f"record(s) settled since the previous reflection, "
              f"{len(all_records)} settled in total this run.")
    costs = cost_report(env, state,
                        describe_wait=getattr(records_mod, "describe_wait",
                                              None))
    entity_of = getattr(records_mod, "entity_of", None)
    ledger = (entity_ledger(all_records, stratum_of, entity_of)
              if entity_of else "(not provided)")
    # readable, id-free evidence lines (verified curation): every own
    # action and every other outcome, newest last, records settled this
    # cycle marked NEW
    def _lines(rs, fn, cap):
        if not fn:
            return "(not provided)"
        out = [("NEW " if since is None or r["t"] > since else "    ")
               + fn(r) for r in rs[-cap:]]
        if len(rs) > cap:
            out.insert(0, f"(oldest {len(rs) - cap} of {len(rs)} omitted; "
                          "the counts above cover them)")
        return "\n".join(out) or "(none yet)"
    own_lines = _lines(own_all, getattr(records_mod, "action_line", None),
                       OWN_HISTORY_CAP)
    outcome_lines = _lines(others_all,
                           getattr(records_mod, "outcome_line", None),
                           DIGEST_CAP)
    instruction = (Path("INSTRUCTION.md").read_text(encoding="utf-8")
                   if Path("INSTRUCTION.md").exists() else "")
    current = (skills.SKILLS.read_text(encoding="utf-8")
               if skills.SKILLS.exists() else "(empty)")
    return {
        "t": t,
        "since": since,
        "n_firing": n_firing,
        "records_mod": records_mod,
        "context": ctx,
        "all_records": all_records,
        "records": records,
        "run_header": header,
        "instruction": instruction,
        "skills": current,
        "cum_counts": status_counts(all_records, stratum_of, is_own),
        "entity_ledger": ledger,
        "own_history": own_history or "(none yet)",
        "outcome_digest": digest,
        "own_lines": own_lines,
        "outcome_lines": outcome_lines,
        "distribution": distribution,
        "own_actions": own_actions or "(none)",
        "examples": examples or "(none)",
        "costs": costs,
    }


def render_reflection_prompt(snapshot: dict,
                             template_path: str = "prompts/reflect.md",
                             extra_slots: dict[str, str] | None = None) -> str:
    """Render a prepared snapshot through a reflection prompt template."""
    template = Path(template_path).read_text(encoding="utf-8")
    prompt = (template
              .replace("${outcome_digest}", snapshot["outcome_digest"])
              .replace("${run_header}", snapshot["run_header"])
              .replace("${instruction}", snapshot["instruction"])
              .replace("${skills}", snapshot["skills"])
              .replace("${cum_counts}", snapshot["cum_counts"])
              .replace("${entity_ledger}", snapshot["entity_ledger"])
              .replace("${own_history}", snapshot["own_history"])
              .replace("${own_lines}", snapshot["own_lines"])
              .replace("${outcome_lines}", snapshot["outcome_lines"])
              .replace("${distribution}", snapshot["distribution"])
              .replace("${own_actions}", snapshot["own_actions"])
              .replace("${examples}", snapshot["examples"])
              .replace("${costs}", snapshot["costs"]))
    slots = dict((snapshot.get("context") or {}).get("extra_slots") or {})
    slots.update(extra_slots or {})
    for name, text in slots.items():
        prompt = prompt.replace("${" + name + "}",
                                text or "(not provided)")
    return prompt


def curate_skills(env, state: dict, snapshot: dict,
                  template_path: str = "prompts/reflect.md",
                  extra_slots: dict[str, str] | None = None) -> str:
    """Run the skill-writer for a prepared reflection snapshot."""
    prompt = render_reflection_prompt(snapshot, template_path, extra_slots)
    text = llm_client.chat(env, [{"role": "user", "content": prompt}]).strip()
    v_before = skills.block_version()
    if text:
        skills.update_skills(text)
    trace.log(snapshot["t"], "reflect", block_version_before=v_before,
              block_version_after=skills.block_version(),
              n_new_records=len(snapshot["records"]), skills=text)
    return text


def finish_reflection(state: dict, snapshot: dict) -> None:
    """Commit the interval boundary after all snapshot consumers finish."""
    state["last_reflection"] = snapshot["t"]


def reflect(env, state: dict, records_mod, edit_scope=None) -> None:
    """Backward-compatible one-call reflection wrapper.

    Existing scaffolds retain the same prompt, state transitions, trace, and
    optional arbitrary-file edit step.  Task-specific multi-consumer
    reflection flows use ``prepare_reflection`` / ``curate_skills`` /
    ``finish_reflection`` directly.
    """
    snapshot = prepare_reflection(env, state, records_mod)
    curate_skills(env, state, snapshot)
    finish_reflection(state, snapshot)

    if edit_scope is not None:
        reflect_edits(env, snapshot["distribution"],
                      snapshot["own_actions"], snapshot["examples"],
                      edit_scope, snapshot["costs"])


def reflect_edits(env, distribution: str, own_actions: str, examples: str,
                  edit_scope, costs: str = "") -> None:
    """The edit step: one LLM call may rewrite workspace files and the
    crontab. edit_scope is EDIT_ALL (anything in the workspace) or a tuple
    of allowed paths/prefixes. Every applied edit is traced and
    git-committed (the run harness initializes the workspace repo)."""
    t = iso(env.now())
    if edit_scope == EDIT_ALL:
        scope_note = ("You may edit any file in this workspace, including "
                      "main.py itself, and replace the crontab.")
    else:
        allowed = " or ".join(edit_scope)
        scope_note = (f"You may ONLY edit files under {allowed}, and you "
                      f"may replace the crontab via new_crontab.")
    prompt = (
        f"You maintain the program in this directory (its entry point is "
        f"main.py; schedule = {json.dumps(env.call('get_crontab'))}).\n"
        f"{scope_note}\n\n"
        f"Settled-outcome distribution since the last reflection:\n"
        f"{distribution}\n\n"
        f"Your own actions and how they settled:\n{own_actions or '(none)'}\n\n"
        f"A stratified sample of other settled outcomes:\n"
        f"{examples or '(none)'}\n\n"
        + (f"What running this program currently costs:\n{costs}\n\n"
           if costs else "") +
        "If a change to the program/config would clearly improve the "
        "score, reply with JSON: {\"edits\": [{\"path\": ..., \"content\": "
        "full new file content}], \"new_crontab\": [{\"id\": ..., "
        "\"cron_expr\": ...}] or null, \"reason\": ...}. "
        "Reply {\"edits\": [], \"new_crontab\": null} if nothing is "
        "clearly better. Whole-file contents only."
    )
    reply = llm_client.chat_json(env, [{"role": "user", "content": prompt}])
    if not isinstance(reply, dict):
        trace.log(t, "note", what="edit_reply_unparseable")
        return
    for edit in reply.get("edits") or []:
        path, content = edit.get("path", ""), edit.get("content")
        if not isinstance(content, str) or not _in_scope(path, edit_scope):
            trace.log(t, "edit_rejected", path=path)
            continue
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        trace.log(t, "edit", path=path, bytes=len(content),
                  reason=reply.get("reason"))
    new_crontab = reply.get("new_crontab")
    if isinstance(new_crontab, list) and new_crontab:
        try:
            env.call("set_crontab", entries=new_crontab)
            trace.log(t, "edit", path="<crontab>", entries=new_crontab)
        except EnvError as e:
            trace.log(t, "edit_rejected", path="<crontab>", error=str(e))
    _git_commit(f"reflection @ {t}")


def _in_scope(path: str, edit_scope) -> bool:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        return False
    if edit_scope == EDIT_ALL:
        return True
    return path in edit_scope or any(
        path.startswith(prefix) for prefix in edit_scope
        if prefix.endswith("/"))


def _git_commit(message: str) -> None:
    if not Path(".git").is_dir():
        return
    subprocess.run(["git", "add", "-A"], capture_output=True)
    subprocess.run(["git", "commit", "-m", message, "--allow-empty"],
                   capture_output=True)


# -- verified curation (alg=vskills) ----
#
# The curation is a persistent ReACT-style transcript:
# a static system message (role, glossary, spec), then per firing a user
# turn with the evidence, the model's reply = the candidate block, a user
# turn with the replay verdict (score vs the incumbent, differing decision
# points, adopted or not), the next candidate, ... — continuing across
# firings. The transcript lives in memory/curation_transcript.jsonl and is
# compacted like an agent transcript: whole firing-segments dropped oldest
# first once over TRANSCRIPT_TOKENS, never the current one.

VERIFIED_SYSTEM = "prompts/reflect_verified_system.md"
VERIFIED_TEMPLATE = "prompts/reflect_verified.md"
TRANSCRIPT = Path("memory") / "curation_transcript.jsonl"
TRANSCRIPT_TOKENS = 60_000  # set by the program from its cell config
SEGMENT_MARK = "(learning cycle "  # every per-firing user turn starts so


def _read_transcript() -> list[dict]:
    if not TRANSCRIPT.exists():
        return []
    with open(TRANSCRIPT, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append_transcript(msgs: list[dict], msg: dict) -> None:
    msgs.append(msg)
    TRANSCRIPT.parent.mkdir(exist_ok=True)
    with open(TRANSCRIPT, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")


def compact_transcript(msgs: list[dict], budget: int) -> list[dict]:
    """Drop the oldest whole firing-segments while over budget, never the
    last one (same rule as runtime.agent.compact_once)."""
    from .tokens import count_tokens

    total = sum(count_tokens(m["content"]) for m in msgs)
    if total <= budget:
        return list(msgs)
    seg = [i for i, m in enumerate(msgs)
           if m["role"] == "user" and m["content"].startswith(SEGMENT_MARK)]
    while total > budget and len(seg) > 1:
        cut = seg[1]
        total -= sum(count_tokens(m["content"]) for m in msgs[:cut])
        msgs = msgs[cut:]
        seg = [i - cut for i in seg[1:]]
    return list(msgs)


def _rewrite_transcript(msgs: list[dict]) -> None:
    TRANSCRIPT.unlink(missing_ok=True)
    for m in msgs:
        _append_transcript([], m)


def _verdict(i: int, res: dict, inc: dict, rows: list[dict], beats: bool,
             score: float, inc_score: float, last: bool,
             summarize=None, render_rows=None, unit: str = "wakes") -> str:
    """The replay verdict turn. bnpm's metric line by default; a task
    replay module may supply `summarize(metrics) -> str` and
    `render_diff_rows(rows) -> [str]` (alg=vskills2) and name its
    decision points (`unit`: wakes / rollouts)."""
    m, im = res["metrics"], inc["metrics"]
    if summarize is not None:
        lines = [f"REPLAY RESULT for your reply #{i} (sha {res['sha']}):",
                 "  " + summarize(m),
                 f"  incumbent on the same {unit}: " + summarize(im),
                 f"  {unit} where your version and the incumbent acted "
                 f"differently: {len(rows)}"]
        lines += render_rows(rows) if render_rows else [
            json.dumps(r, separators=(",", ":")) for r in rows]
    else:
        lines = [f"REPLAY RESULT for your block #{i} (sha {res['sha']}):",
                 f"  cov_f1 {m['cov_f1']} | precision {m['precision']} | "
                 f"cov_recall {m['cov_recall']} | claims {m['alerts']} "
                 f"(false alarms {m['false_alarms']}, rejected "
                 f"{m['alerts_rejected']}) | breakpoints covered "
                 f"{m['breakpoints_covered']}/{m['breakpoints_closed']}",
                 f"  incumbent on the same decision points: cov_f1 {im['cov_f1']} "
                 f"| precision {im['precision']} | cov_recall {im['cov_recall']} "
                 f"| claims {im['alerts']} (false alarms {im['false_alarms']})",
                 f"  decision points where your block and the incumbent chose "
                 f"differently: {len(rows)}"]
        lines += [json.dumps(r, separators=(",", ":")) for r in rows]
    if beats:
        lines.append(f"DECISION: adopted — {score} > {inc_score}; this "
                     "version is now the incumbent.")
    else:
        lines.append(f"DECISION: not adopted — {score} is not strictly above "
                     f"the incumbent's {inc_score}; the incumbent stays.")
        if not last:
            lines.append(REPLY_AGAIN)
    return "\n".join(lines)


def reflect_verified(env, state: dict, records_mod, replay_mod,
                     episode_factory, n_iter: int, claim_window_hours: float,
                     price_delay_minutes: float, context_tokens: int,
                     max_calls: int, max_items: int | None = None) -> None:
    """alg=vskills (bnpm): propose one small edit set to the incumbent
    block, replay the edited block on the settled decision points, keep
    it only if it strictly beats the incumbent there; otherwise report
    the verdict into the curation transcript and ask again, up to n_iter
    times. Everything is traced (vskills_* rows), every candidate is
    kept under memory/candidates/<firing>/, and the whole conversation
    is memory/curation_transcript.jsonl."""
    snapshot = prepare_reflection(env, state, records_mod)
    now = env.now()
    items, dropped = replay_mod.build_replay_set(now, claim_window_hours,
                                                 max_items)
    cfg = {"firing": snapshot["n_firing"],
           "claim_window_hours": claim_window_hours,
           "price_delay_minutes": price_delay_minutes,
           "context_tokens": context_tokens, "max_calls": max_calls}
    slots = {"max_edits": str(MAX_EDITS),
             "max_edit_words": str(MAX_EDIT_WORDS),
             "block_tokens": str(skills.BLOCK_TOKENS)}
    _verified_loop(
        env, state, snapshot, records_mod, replay_mod, episode_factory,
        n_iter, cfg, items, dropped, max_items,
        system=render_reflection_prompt(snapshot, VERIFIED_SYSTEM, slots),
        turn=render_reflection_prompt(snapshot, VERIFIED_TEMPLATE, slots),
        start_fields={}, transcript_tokens=TRANSCRIPT_TOKENS)


VERIFIED2_SYSTEM = "prompts/reflect_verified2_system.md"
VERIFIED2_TEMPLATE = "prompts/reflect_verified2.md"
VERIFIED3_SYSTEM = "prompts/reflect_verified3_system.md"  # alg=vskills3
VERIFIED3_TEMPLATE = "prompts/reflect_verified3.md"


def reflect_verified2(env, state: dict, records_mod, replay_mod,
                      episode_factory, n_iter: int, params: dict,
                      context_tokens: int, max_calls: int,
                      max_items: int | None = None,
                      agent_name: str = "actor",
                      system_prompt: str | None = None,
                      template: str | None = None) -> None:
    """alg=vskills2 (verified-curation v2 plan): the same verified loop
    with (1) the actor's trajectory and (2) the formatted memory render
    as the evidence, curating a REMINDER that the actor receives as a
    user message at every wake, replayed on the actor's own past wakes
    (replay_mod.build_replay_set(now, horizon_hours, max_items)) and
    adopted only when its score strictly beats the incumbent's. The
    curation transcript is never compacted (no context limit for this
    arm). `params` are the task constants the
    replay settlement mirrors (REPLAY_PARAMS in the generated main).
    alg=vskills3 passes its day-rollout
    module as `replay_mod` and its own prompt files: the loop is
    unit-agnostic — build_replay_set / evaluate / diff / scorable are
    the module contract, `UNIT` names the decision points."""
    from . import trajectory

    snapshot = prepare_reflection(env, state, records_mod)
    now = env.now()
    items, dropped = replay_mod.build_replay_set(now, params["horizon_hours"],
                                                 max_items)
    cfg = {"firing": snapshot["n_firing"], "params": params,
           "context_tokens": context_tokens, "max_calls": max_calls,
           "version": skills.block_version(), "records_mod": records_mod}
    slots = {"max_edits": str(MAX_EDITS),
             "max_edit_words": str(MAX_EDIT_WORDS),
             "block_tokens": str(skills.BLOCK_TOKENS),
             "trajectory": trajectory.render(agent_name, records_mod),
             "memory_block": memory.render_block() or "(no settled records yet)",
             "n_new_records": str(len(snapshot["records"]))}
    stats = (replay_mod.acting_stats(now, params["horizon_hours"])
             if hasattr(replay_mod, "acting_stats") else {})
    _verified_loop(
        env, state, snapshot, records_mod, replay_mod, episode_factory,
        n_iter, cfg, items, dropped, max_items,
        system=render_reflection_prompt(
            snapshot, system_prompt or VERIFIED2_SYSTEM, slots),
        turn=render_reflection_prompt(
            snapshot, template or VERIFIED2_TEMPLATE, slots),
        start_fields=stats, transcript_tokens=None,
        summarize=getattr(replay_mod, "summarize_metrics", None),
        render_rows=getattr(replay_mod, "render_diff_rows", None),
        lint_extra=getattr(records_mod, "LINT_EXTRA", None),
        unit=getattr(replay_mod, "UNIT", "wakes"))


def _verified_loop(env, state: dict, snapshot: dict, records_mod, replay_mod,
                   episode_factory, n_iter: int, cfg: dict, items: list[dict],
                   dropped: int, max_items: int | None, system: str, turn: str,
                   start_fields: dict, transcript_tokens: int | None,
                   summarize=None, render_rows=None, lint_extra=None,
                   unit: str = "wakes") -> None:
    """The propose -> replay -> adopt-if-better loop shared by
    reflect_verified (bnpm blocks) and reflect_verified2 (reminders).
    transcript_tokens=None keeps the whole curation transcript."""
    firing = snapshot["n_firing"]
    incumbent = (skills.SKILLS.read_text(encoding="utf-8")
                 if skills.SKILLS.exists() else "")
    trace.log(snapshot["t"], "vskills_start", firing=firing,
              n_items=len(items),
              n_wakes=sum(1 for i in items if i["kind"] == "wake"),
              n_decides=sum(1 for i in items if i["kind"] == "decide"),
              n_rollouts=sum(1 for i in items if i["kind"] == "rollout"),
              n_dropped=dropped, max_items=max_items,
              incumbent_sha=replay_mod.block_sha(incumbent),
              incumbent_version=skills.block_version(), n_iter=n_iter,
              **start_fields)
    # guard 1: curation runs only on feedback — nothing settled since the
    # last reflection, or nothing replayable yet, means nothing to learn
    # from and nothing to verify against: no proposer call, no adoption
    seen = state.get("vskills_seen") or {}
    item_ids = sorted(i["id"] for i in items)
    why = None
    if not snapshot["records"] and seen.get("items") == item_ids:
        why = "no_new_evidence"
    elif not snapshot["records"]:
        why = "no_new_records"
    elif not items:
        why = "no_settled_decision_points"
    state["vskills_seen"] = {"items": item_ids}
    if why:
        trace.log(snapshot["t"], "vskills_skip", firing=firing, why=why,
                  n_items=len(items), n_new_records=len(snapshot["records"]))
        finish_reflection(state, snapshot)
        return
    cdir = Path("memory") / "candidates" / str(firing)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "incumbent.md").write_text(incumbent, encoding="utf-8")
    try:
        inc = replay_mod.evaluate(env, incumbent, items, episode_factory,
                                  {**cfg, "cand": "incumbent"})
    except replay_mod.ReplayAborted as e:
        trace.log(snapshot["t"], "vskills_abort", firing=firing,
                  stage="incumbent", error=str(e))
        finish_reflection(state, snapshot)
        return
    _dump(cdir / "incumbent.score.json", inc)
    scorable = getattr(replay_mod, "scorable", None)
    if scorable is not None and not scorable(inc):
        # guard 4: nothing in the replayed span can earn credit for any
        # version — no proposer call, the incumbent stays, and the
        # evidence turn is not added (the next firing renders it afresh)
        trace.log(snapshot["t"], "vskills_skip", firing=firing,
                  why="nothing_scorable", n_items=len(items),
                  metrics=inc["metrics"])
        finish_reflection(state, snapshot)
        return
    board = [{"id": "incumbent", "sha": inc["sha"], "metrics": inc["metrics"]}]
    inc_score = replay_mod.replay_scorer.score_value(inc["metrics"])
    system_msg = {"role": "system", "content": system}
    msgs = _read_transcript()
    _append_transcript(msgs, {"role": "user", "content":
                       f"{SEGMENT_MARK}{firing} — sim time {snapshot['t']})\n\n"
                       + turn})
    adopted = None
    for i in range(1, n_iter + 1):
        if transcript_tokens is not None:
            msgs = compact_transcript(msgs, transcript_tokens)
        (cdir / f"{i}.prompt.md").write_text(
            "\n\n".join(f"### {m['role']}\n{m['content']}"
                        for m in [system_msg] + msgs), encoding="utf-8")
        raw = llm_client.chat(env, [system_msg] + msgs)
        _append_transcript(msgs, {"role": "assistant", "content": raw})
        (cdir / f"{i}.reply.txt").write_text(raw, encoding="utf-8")
        parsed = parse_reply(raw)
        if parsed is None:
            # the reply was not the required JSON object: nothing to
            # replay; tell the proposer and ask again (counts as a try)
            trace.log(snapshot["t"], "vskills_reject", firing=firing, cand=i,
                      why="format", reply_chars=len(raw))
            _append_transcript(msgs, {"role": "user", "content":
                               f"REJECTED before replay: reply #{i} was not "
                               "the required JSON object.\n" + REPLY_AGAIN})
            continue
        analysis, edits = parsed
        (cdir / f"{i}.analysis.md").write_text(analysis, encoding="utf-8")
        (cdir / f"{i}.edits.json").write_text(json.dumps(edits, indent=1),
                                              encoding="utf-8")
        if not edits:
            # guard 3: the proposer declines to change the block — a
            # legitimate verdict on the evidence, not a format error;
            # nothing to replay, the firing ends
            trace.log(snapshot["t"], "vskills_stop", firing=firing, cand=i,
                      why="no_edits", analysis_chars=len(analysis))
            _append_transcript(msgs, {"role": "user", "content":
                               f"Noted — reply #{i} proposes no edits; the "
                               "incumbent stays for this cycle."})
            break
        text, err = apply_edits(incumbent, edits, MAX_EDITS, MAX_EDIT_WORDS)
        hits = lint_block("\n".join(e["new"] for e in edits), lint_extra)
        if err or hits:
            # the edits break the minimal-change contract, or the new
            # text memorises the replay set (ids, timestamps, decision
            # points) instead of stating a policy: refused without replay
            why = "edits" if err else "lint"
            reason = err or (f"the new text names specific ids, timestamps "
                             f"or decision points ({', '.join(hits)}); a "
                             "rule must be a condition on evidence the "
                             "program can observe at decision points it "
                             "has never seen")
            trace.log(snapshot["t"], "vskills_reject", firing=firing, cand=i,
                      why=why, reason=reason, matches=hits,
                      n_edits=len(edits), analysis_chars=len(analysis))
            _append_transcript(msgs, {"role": "user", "content":
                               f"REJECTED before replay: reply #{i} — "
                               f"{reason}.\n" + REPLY_AGAIN})
            continue
        (cdir / f"{i}.md").write_text(text, encoding="utf-8")
        try:
            res = replay_mod.evaluate(env, text, items, episode_factory,
                                      {**cfg, "cand": i,
                                       **({"version": cfg["version"] + 1}
                                          if "version" in cfg else {})})
        except replay_mod.ReplayAborted as e:
            trace.log(snapshot["t"], "vskills_abort", firing=firing,
                      stage=f"cand{i}", error=str(e))
            _append_transcript(msgs, {"role": "user", "content":
                               f"REPLAY RESULT for your reply #{i}: "
                               "verification aborted (the run's LLM "
                               "budget is spent); the incumbent stays."})
            break
        _dump(cdir / f"{i}.score.json", res)
        rows = replay_mod.diff(res, inc, items)
        with open(cdir / f"{i}.diff.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        score = replay_mod.replay_scorer.score_value(res["metrics"])
        beats = score > inc_score
        stop = beats or not rows or i == n_iter
        board.append({"id": f"cand{i}", "sha": res["sha"],
                      "metrics": res["metrics"], "diff": rows})
        _append_transcript(msgs, {"role": "user", "content": _verdict(
            i, res, inc, rows, beats, score, inc_score, last=stop,
            summarize=summarize, render_rows=render_rows, unit=unit)})
        trace.log(snapshot["t"], "vskills_iter", firing=firing, cand=i,
                  block_sha=res["sha"], score=score, incumbent_score=inc_score,
                  beats=beats, n_diff=len(rows),
                  transcript_messages=len(msgs) + 1,
                  prompt_chars=sum(len(m["content"]) for m in [system_msg] + msgs),
                  reply_chars=len(raw), analysis_chars=len(analysis),
                  n_edits=len(edits), block_chars=len(text))
        if beats:
            adopted = (i, text, res)
            break
        if not rows:
            # guard 2: the candidate decided exactly as the incumbent on
            # every point — it cannot beat it; stop this firing
            trace.log(snapshot["t"], "vskills_stop", firing=firing, cand=i,
                      why="no_behavioral_difference")
            break
    _rewrite_transcript(compact_transcript(msgs, transcript_tokens)
                        if transcript_tokens is not None else msgs)
    (cdir / "board.json").write_text(
        json.dumps([{k: v for k, v in b.items() if k != "diff"}
                    for b in board], indent=1), encoding="utf-8")
    if adopted:
        i, text, res = adopted
        v_before = skills.block_version()
        skills.update_skills(text)
        trace.log(snapshot["t"], "reflect", block_version_before=v_before,
                  block_version_after=skills.block_version(),
                  n_new_records=len(snapshot["records"]), skills=text)
        trace.log(snapshot["t"], "vskills_adopt", firing=firing, cand=i,
                  verified=True, block_sha=res["sha"],
                  score=replay_mod.replay_scorer.score_value(res["metrics"]),
                  incumbent_score=inc_score)
    else:
        trace.log(snapshot["t"], "vskills_keep", firing=firing,
                  tried=len(board) - 1, incumbent_score=inc_score,
                  best_candidate=max((replay_mod.replay_scorer.score_value(
                      b["metrics"]) for b in board[1:]), default=None))
    finish_reflection(state, snapshot)


MAX_EDITS = 3  # verified curation: edits accepted per reply
MAX_EDIT_WORDS = 100  # verified curation: new words accepted per reply
REPLY_AGAIN = ('Reply again with one JSON object {"analysis": "...", '
               '"edits": [{"old": "...", "new": "..."}, ...]} — analysis '
               'is your reasoning for the record; each edit replaces one '
               'exact passage of the current block (old) with new text '
               '(old "" appends, new "" deletes); nothing outside the '
               'object.')
# a block that names an article/decision-point id or a timestamp is
# reciting the replay set, not stating a policy
LINT = [("id", re.compile(r"\b[0-9a-f]{12,}\b")),
        ("timestamp", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
        ("decision point", re.compile(r"\w@\d{4}-\d{2}-\d{2}"))]


def parse_reply(text: str) -> tuple[str, list[dict]] | None:
    """The proposer's reply as (analysis, edits), or None when it is not
    a JSON object with an edits list of {old, new} strings. An empty
    edits list is a valid reply: the proposer keeps the incumbent."""
    t = strip_fences(text)
    if t.startswith("json"):
        t = t[4:].lstrip()
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    edits = obj.get("edits")
    if not isinstance(edits, list) or not all(
            isinstance(e, dict) and isinstance(e.get("old", ""), str)
            and isinstance(e.get("new", ""), str) for e in edits):
        return None
    analysis = obj.get("analysis")
    return (analysis if isinstance(analysis, str) else json.dumps(analysis),
            [{"old": e.get("old", ""), "new": e.get("new", "")}
             for e in edits])


def apply_edits(block: str, edits: list[dict], max_edits: int,
                max_words: int) -> tuple[str | None, str | None]:
    """The block after the edits, or (None, reason) when they break the
    minimal-change contract: at most max_edits edits, at most max_words
    new words in total, every non-empty old matching exactly once."""
    if len(edits) > max_edits:
        return None, f"{len(edits)} edits exceed the limit of {max_edits}"
    words = sum(len(e["new"].split()) for e in edits)
    if words > max_words:
        return None, f"{words} new words exceed the limit of {max_words}"
    out = block
    for e in edits:
        old, new = e["old"], e["new"].strip()
        if not old:
            if new:
                out = (out.rstrip() + "\n" + new) if out.strip() else new
            continue
        n = out.count(old)
        if n != 1:
            return None, (f"old text {old[:60]!r} matches the current block "
                          f"{n} times (must match exactly once)")
        out = out.replace(old, new)
    out = "\n".join(l for l in out.splitlines() if l.strip()).strip()
    if not out:
        return None, "the edits leave the block empty"
    return out, None


def lint_block(block: str, extra=None) -> list[str]:
    """Names of the generality rules the block violates, with the first
    offending token each: [] when clean. `extra` = task-declared
    (name, regex) pairs (records.LINT_EXTRA) checked as well."""
    hits = []
    for name, rx in list(LINT) + list(extra or []):
        m = rx.search(block)
        if m:
            hits.append(f"{name} '{m.group(0)}'")
    return hits


def strip_fences(text: str) -> str:
    """Drop the template's own <<<SKILL / SKILL>>> delimiters (and a code
    fence) when the model echoes them around the block."""
    t = (text or "").strip()
    for head in ("<<<SKILL", "```"):
        if t.startswith(head):
            t = t[len(head):].lstrip("\n")
    for tail in ("SKILL>>>", "```"):
        if t.endswith(tail):
            t = t[:-len(tail)].rstrip()
    return t.strip()


def _dump(path: Path, res: dict) -> None:
    key = "alerts" if "alerts" in res else "recs"
    path.write_text(json.dumps(
        {"sha": res["sha"], "metrics": res["metrics"],
         key: res[key],
         # the settled view minus the floats the scorer derived
         "settled_actions": [{k: v for k, v in a.items()
                              if not isinstance(v, float)}
                             for a in res["settled"][key]],
         "per_item": res["per_item"]}, indent=1, default=str),
        encoding="utf-8")
