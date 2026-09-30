"""Agent: context management + one behavior method, turn(). Program library.

An Agent owns a transcript file, the deterministic compaction rule, the
system-prompt render (INSTRUCTION.md + tool docs + learned block), and a
tool registry. `turn()` does one LLM call and executes the tool call it
returns. That is the whole interface: loops are program code written around
the agent (`while actor.turn(): pass` is an entire ReACT cell), and waiting
is just a tool in the registry — its implementation blocks in the env's
wait endpoint, which fast-forwards the sim clock, and the wake payload
comes back as an ordinary tool result in the transcript.

Context policy: when the transcript exceeds
context_tokens, the oldest whole wake-segments are dropped — deterministic
truncation, no summarizer call; the learned block lives in the system
prompt and is preserved verbatim by construction. The compactor is part of
the spec — not a learnable component.

Guards: max_calls_per_wake breaches and LLM-budget 503s
log a note and exit the process — never silently override the agent's own
timing; the supervisor resumes at the next trigger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import llm_client, trace
from .env_client import EnvError, iso
from .tokens import count_tokens


def env_tools(env) -> dict[str, dict]:
    """Build a tool registry from the environment's manifest: every
    provisioned tool becomes callable, docs verbatim from the manifest."""
    out: dict[str, dict] = {}
    for t in env.tools():
        out[t["name"]] = {
            "doc": t["doc"],
            "tags": set(t.get("tags") or ()),
            "fn": (lambda args, _n=t["name"]: env.call(_n, **args)),
        }
    return out


def done_tool() -> dict[str, dict]:
    """A `done` tool for run-to-completion agents (run_until_done)."""
    return {
        "done": {
            "doc": "done(summary?: str) -> finish this run",
            "fn": lambda args: {"done": True,
                                "summary": args.get("summary", "")},
        },
    }


def brief_line(costs: dict) -> str:
    """The daily cost & budget brief as one fixed line: numbers only, identical across arms."""
    def money(block: dict) -> str:
        return (f"${block['spent_usd']:.2f} spent of "
                f"${block['budget_usd']:.2f} (${block['remaining_usd']:.2f} left)")
    by_type = ", ".join(f"{k} ${v:.2f}"
                        for k, v in costs.get("spend_by_type", {}).items())
    line = (f"Day {costs['day']} of {costs['of']}. Budget: {money(costs)}. "
            f"LLM budget: {money(costs['llm'])}.")
    for t, block in costs.get("domains", {}).items():
        line += f" {t} budget: {money(block)}."
    if by_type:
        line += f" By type: {by_type}."
    return line


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def context_sha(messages: list[dict]) -> str:
    """Digest of a message list as the LLM would receive it."""
    return sha(json.dumps(messages, separators=(",", ":")))


def compact_once(messages: list[dict], context_tokens: int) -> list[dict]:
    """The compaction rule applied once to a message list: drop the
    oldest whole wake-segments while over budget, never the last one.
    Applying it incrementally after every turn (Agent._compact) and once
    to the full history prefix give the same suffix — drops are monotone
    and a dropped prefix never comes back — which is what lets a replay
    rebuild a past wake's context from logs/history_<name>.jsonl."""
    total = sum(count_tokens(m["content"]) for m in messages)
    if total <= context_tokens:
        return list(messages)
    wake_idx = [i for i, m in enumerate(messages)
                if m["content"].startswith("(woke at ")]
    while total > context_tokens and len(wake_idx) > 1:
        cut = wake_idx[1]
        total -= sum(count_tokens(m["content"]) for m in messages[:cut])
        messages = messages[cut:]
        wake_idx = [i - cut for i in wake_idx[1:]]
    return list(messages)


class Agent:
    def __init__(self, env, name: str, tools: dict[str, dict],
                 transcript: str | Path, block_fn=None,
                 wait_tool: str | None = None,
                 instruction: str | Path = "INSTRUCTION.md",
                 context_tokens: int = 60_000,
                 max_calls_per_wake: int = 200,
                 reminder_fn=None, event_fn=None):
        self.env = env
        self.name = name
        self.tools = tools
        self.transcript_path = Path(transcript)
        # reminder channel: () -> str,
        # delivered as ONE user message right after every wake marker —
        # never in the system prompt. Empty text = nothing delivered.
        # The wake row's context_sha is taken BEFORE delivery, so a
        # replay rebuilds the same context and appends its own
        # candidate reminder in the same place.
        self.reminder_fn = reminder_fn
        # Optional harness-owned structured observability sink. It must be
        # best-effort: logging can never change the actor's control flow.
        self.event_fn = event_fn
        # append-only twin of the transcript: every message ever appended,
        # never compacted (logs/history_<name>.jsonl). The transcript is
        # the live context and gets rewritten by the compactor; the
        # history is what a later replay reconstructs a past wake's
        # context from.
        self.history_path = Path("logs") / f"history_{name}.jsonl"
        self.block_fn = block_fn  # () -> str; rendered fresh every turn
        self.wait_tool = wait_tool  # name of the wait tool, if any
        self.instruction_path = Path(instruction)
        self.context_tokens = context_tokens
        self.max_calls_per_wake = max_calls_per_wake
        self.messages = self._load_transcript()
        self.turns = 0  # completed turns this process
        self.done = False  # set by the done tool (run_until_done)
        self._calls_since_wait = 0
        self._last_wait_now: str | None = None

    def _observe(self, event: str, **fields) -> None:
        if not self.event_fn:
            return
        try:
            self.event_fn(event, **fields)
        except Exception:
            pass

    # -- transcript ---------------------------------------------------------------

    def _load_transcript(self) -> list[dict]:
        if not self.transcript_path.exists():
            return []
        with open(self.transcript_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, msg: dict) -> None:
        self.messages.append(msg)
        line = json.dumps(msg) + "\n"
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(line)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _rewrite_transcript(self, messages: list[dict]) -> None:
        """Replace the live transcript (compaction); the history keeps
        every line."""
        self.transcript_path.unlink(missing_ok=True)
        self.messages = []
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            for m in messages:
                self.messages.append(m)
                f.write(json.dumps(m) + "\n")

    def wake(self, trigger: dict | None = None) -> None:
        """Append the wake marker for this process invocation (compaction
        segments key on it)."""
        trigger = trigger or {}
        marker = (f"(woke at {self.env.now().isoformat()} for "
                  f"trigger {trigger.get('id')}/{trigger.get('kind')})")
        if trigger.get("note") is not None:  # an agent-owned schedule (TM-D)
            marker += f"\nYour note for this schedule: {trigger['note']}"
        if trigger.get("costs"):  # first wake of a sim date
            marker += "\n" + brief_line(trigger["costs"])
        self._append({"role": "user", "content": marker})
        # the wake row: where this wake's marker sits in the history and
        # what the LLM context looked like — a replay of this wake
        # reconstructs both and must match
        reminder = self._reminder()
        trace.log(iso(self.env.now()), "wake", agent=self.name,
                  trigger_id=trigger.get("id"),
                  trigger_kind=trigger.get("kind"),
                  history_index=self._history_len() - 1,
                  n_messages=len(self.messages),
                  context_sha=context_sha(self.messages),
                  prefix_sha=sha(self.system_prompt(with_block=False)),
                  reminder_sha=sha(reminder) if reminder else "")
        self._deliver(reminder)
        self._observe("wake", sim_time=iso(self.env.now()),
                      trigger_id=trigger.get("id"),
                      trigger_kind=trigger.get("kind"),
                      messages=len(self.messages),
                      reminder=bool(reminder))

    def _reminder(self) -> str:
        return (self.reminder_fn() or "") if self.reminder_fn else ""

    def _deliver(self, reminder: str) -> None:
        if reminder:
            self._append({"role": "user", "content": reminder})

    def _history_len(self) -> int:
        if not self.history_path.exists():
            return 0
        with open(self.history_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def note(self, text: str) -> None:
        """Inject a visible note into the transcript (e.g. to make an
        out-of-band curation run agent-aware)."""
        self._append({"role": "user", "content": text})

    def _compact(self) -> None:
        """Deterministic truncation: drop the oldest whole wake-segments
        (never the current wake) once over budget."""
        messages = compact_once(self.messages, self.context_tokens)
        dropped = len(self.messages) - len(messages)
        if not dropped:
            return
        self._rewrite_transcript(messages)
        trace.log(iso(self.env.now()), "note", what="compacted",
                  agent=self.name, dropped_messages=dropped)
        self._observe("compaction", sim_time=iso(self.env.now()),
                      dropped_messages=dropped,
                      remaining_messages=len(self.messages))

    # -- prompt -------------------------------------------------------------------

    def system_prompt(self, with_block: bool = True) -> str:
        instruction = self.instruction_path.read_text(encoding="utf-8") \
            if self.instruction_path.exists() else ""
        tool_docs = "\n".join(f"- {t['doc']}" for t in self.tools.values())
        wait_line = (f"When there is nothing to do now, call "
                     f"{self.wait_tool} — time only passes while you wait. "
                     if self.wait_tool else "")
        block = self.block_fn() if (self.block_fn and with_block) else ""
        return (
            f"{instruction}\n\n# How you run\n"
            f"You are a long-running agent in simulated time. Reply with "
            f"ONLY a JSON object: {{\"tool\": <name>, \"args\": {{...}}, "
            f"\"thought\": <brief>}}. {wait_line}"
            f"Available tools:\n{tool_docs}\n\n"
            + (f"{block}\n" if block else ""))

    # -- the behavior method ------------------------------------------------------

    def turn(self) -> bool:
        """One LLM call + execute its tool call. Returns False when the
        experiment is over (or this agent called done); True otherwise.
        Raises SystemExit on the runaway/budget guards."""
        self._observe("llm_request", sim_time=iso(self.env.now()),
                      turn=self.turns, messages=len(self.messages),
                      calls_since_wait=self._calls_since_wait)
        try:
            reply = llm_client.chat_json(
                self.env,
                [{"role": "system", "content": self.system_prompt()}]
                + self.messages)
        except EnvError as e:
            self._observe("llm_error", sim_time=iso(self.env.now()),
                          error=str(e))
            if "503" in str(e):  # LLM budget exhausted: exit; the
                self._exit_note("budget_exhausted")  # supervisor resumes
            raise                                    # at the next trigger
        if not isinstance(reply, dict) or "tool" not in reply:
            self._observe("llm_reply", sim_time=iso(self.env.now()),
                          valid=False, reply_type=type(reply).__name__)
            self._append({"role": "user", "content":
                          "Reply with ONLY {\"tool\": ..., \"args\": {...}, \"thought\": ...(optional)}"})
            self._bump_guard()
            return True

        self._append({"role": "assistant",
                      "content": json.dumps(reply, separators=(",", ":"))})
        name, args = reply.get("tool"), reply.get("args") or {}
        self._observe("llm_reply", sim_time=iso(self.env.now()), valid=True,
                      tool=name, thought=reply.get("thought"), turn=self.turns)
        tool = self.tools.get(name)
        t = iso(self.env.now())
        if tool is None:
            result = {"error": f"unknown tool {name!r}"}
        else:
            try:
                result = tool["fn"](args)
            except EnvError as e:
                result = {"error": str(e)}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
        trace.log(t, "action", agent=self.name, tool=name, args=args,
                  ok="error" not in (result or {})
                  if isinstance(result, dict) else True)
        self._observe("tool_result", sim_time=t, tool=name, args=args,
                      ok=("error" not in result
                          if isinstance(result, dict) else True),
                      error=(result.get("error")
                             if isinstance(result, dict) else None))

        if isinstance(result, dict) and result.get("experiment_over"):
            return False
        self._append({"role": "user", "content": json.dumps(
            result, separators=(",", ":"), default=str)})

        if isinstance(result, dict) and result.get("done"):
            self.done = True
        # wait-class = the declared wait tool or any wait-tagged tool
        # (run_program advances sim time exactly like wait_until does)
        is_wait = (name == self.wait_tool
                   or (tool is not None and "wait" in (tool.get("tags") or ())))
        advanced = (is_wait
                    and isinstance(result, dict) and "error" not in result
                    and result.get("now") not in (None, self._last_wait_now))
        if advanced:
            self._last_wait_now = result["now"]
            self._calls_since_wait = 0
            # segment boundary: a wake is a sim-time event, not a process
            # invocation — the react loop waits in-process, so without
            # this marker the whole run is one segment and the compactor
            # can never fire (its guard keeps the current segment whole)
            marker = f"(woke at {result['now']} from {name})"
            if result.get("costs"):  # first wake of a sim date
                marker += "\n" + brief_line(result["costs"])
            self._append({"role": "user", "content": marker})
            trig = result.get("trigger") or {}
            reminder = self._reminder()
            trace.log(iso(self.env.now()), "wake", agent=self.name,
                      trigger_id=trig.get("id"), trigger_kind=name,
                      history_index=self._history_len() - 1,
                      n_messages=len(self.messages),
                      context_sha=context_sha(self.messages),
                      prefix_sha=sha(self.system_prompt(with_block=False)),
                      reminder_sha=sha(reminder) if reminder else "")
            self._deliver(reminder)
            self._observe("wait_advanced", sim_time=iso(self.env.now()),
                          tool=name, woke_at=result.get("now"),
                          reminder=bool(reminder))
        else:  # a wait that does not move the clock is not a wait
            self._bump_guard()
        self._compact()
        self.turns += 1
        return not self.done

    def _bump_guard(self) -> None:
        self._calls_since_wait += 1
        if self._calls_since_wait >= self.max_calls_per_wake:
            self._exit_note("runaway")

    def _exit_note(self, why: str) -> None:
        self._append({"role": "user", "content":
                      f"(process exited: {why}; resumed at next trigger)"})
        trace.log(iso(self.env.now()), "note", what=why, agent=self.name)
        self._observe("agent_exit", sim_time=iso(self.env.now()), reason=why)
        raise SystemExit(0)


def run_until_done(agent: Agent, max_calls: int) -> None:
    """Run a triggered agent to completion at the current sim instant: turn
    until its done tool fires or the per-firing call cap is hit. The agent
    must hold no wait tool — it cannot move sim time."""
    agent.done = False
    for _ in range(max_calls):
        if not agent.turn():
            break
