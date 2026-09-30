# TM-A reflection-only actor: components and wiring

This document describes the current source candidate in this directory:

```text
tasks/breakout_news_pm/agent/ext_candidates/tmA-reflection-only/
```

It explains what changed relative to `tmA-base` (the per-market base program,
`tasks/breakout_news_pm/agent/scaffolds/per_market_main.py`), how the pieces are
connected at runtime, what files a run produces, and where to look when a
component behaves unexpectedly.

## 1. Scope and important terminology

There are two similarly named layers:

- `runtime/actor.py` is the protected process runner. It obtains triggers,
  starts `python main.py`, enforces the watchdog, and reports process exits.
  This file was **not modified**.
- `main.py` and `runtime/agent.py` implement the actor agent program that runs
  inside the process. These were modified to add observation and reflection.

The reflective system added here is separate from the repository's older
`ALG` / `SIG` learning stack. The current TM-A base injects:

```python
TM = "A"
SIG = "none"
ALG = "none"
WAIT_TOOL = "sleep"
```

Consequently, the old `LEARNING` branch retained in `main.py` is inactive.
The new `reflective.py` path is wired unconditionally and works under
`algnone/signone`.

The candidate has no scorer or oracle feedback tool. “Hindsight feedback” in
this design means provisional feedback inferred from:

1. the actor's own past notifications;
2. prices that become visible after the fact;
3. article metadata the actor already fetched.

It is not the scorer's settled label and is deliberately described to the
model as heuristic.

## 2. File map

| File | Status | Responsibility |
|---|---|---|
| `main.py` | modified | Builds per-market agents, wraps selected tools, creates the concurrent wait party, starts the daily curator, and injects reflective memory into agents. |
| `reflective.py` | added | Persistent state, observability writer, action/news capture, price-derived feedback, memory curation, and dynamic memory rendering. |
| `runtime/agent.py` | modified | Adds an optional best-effort event callback around the existing LLM/tool/wait loop. |
| `ORIGIN.md` | modified | Records the candidate's origin and the high-level changes. |
| `runtime/actor.py` | unchanged | Protected environment runner; still byte-identical to the mounted runtime. |

Files such as `INSTRUCTION.md`, `cell_config.py`, `instructions/`, `logs/`,
and `memory/` are not source configuration in the candidate. The runner or
the running program creates them inside each run workspace.

## 3. Source candidate versus run workspace

When the launcher starts a run, it copies this candidate into:

```text
runs/task_breakout_news_pm/gpt-5.6-luna/<run-id>/workspace/
```

The runner then injects the base's current `INSTRUCTION.md` and
`cell_config.py` into that copy. All runtime state and logs belong to the
copy, not to the source candidate.

For a completed reflection-only run, `<run-id>` below is its launcher id:

```text
runs/task_breakout_news_pm/gpt-5.6-luna/
  <run-id>/
```

Its important generated files are:

```text
workspace/
  INSTRUCTION.md
  cell_config.py
  instructions/m-<market-id>.md
  logs/observability.jsonl
  logs/transcript_m-<market-id>.jsonl
  logs/history_m-<market-id>.jsonl
  logs/trace.jsonl
  memory/reflective_state.json
results.json
ledger.jsonl
llm_log.jsonl
actor.log
```

Every fresh run starts without reflective state. Learned lessons from one run
are not seeded into another candidate or run.

## 4. Process and wait-party topology

The baseline program used eight market-agent threads plus the main-thread
coordinator. This candidate adds one curator thread:

```text
runtime/actor.py
  |
  +-- starts python main.py
        |
        +-- market thread m-<id 1> -- Agent.turn() loop -- sleep(waiter_id)
        +-- market thread m-<id 2> -- Agent.turn() loop -- sleep(waiter_id)
        +-- ... one thread per get_markets() row
        +-- curator thread ---------- daily review ---- sleep(waiter_id)
        +-- main coordinator -------- parks to SIM_END - sleep(waiter_id)
```

`main.py` calls `get_markets()`, then declares all market ids plus `curator`
and `coordinator` through `set_party`. The coordinator is the party's
`trigger_waiter`.

The party is necessary because the server advances simulated time only when
every declared waiter is parked. Normal behavior at a simulated instant is:

1. Market agents perform LLM/tool work at frozen simulated time.
2. Each market agent eventually calls `sleep` with its own waiter id.
3. The curator parks until its next daily deadline.
4. The coordinator is parked until the experiment end.
5. Once everyone is parked, the server advances to the earliest deadline.
6. The relevant waiter wakes and resumes work.

The curator therefore does not create a separate cron trigger or a fresh
process. It is a resident member of the same TM-A wait party.

`leave_party(name)` removes a market agent or the curator if it exits. This
prevents the remaining threads from deadlocking behind a waiter that no
longer exists.

Relevant source locations:

- constants and sim bounds: `main.py`, near `COORDINATOR`, `CURATOR`, and
  `CURATION_HOURS`;
- curator loop: `main.py::run_curator`;
- party declaration and thread startup: the bottom of `main.py`;
- party removal: `main.py::leave_party`.

## 5. Prompt construction

Each market agent receives a prompt assembled in several layers. The order is
important:

```text
base INSTRUCTION.md supplied by the environment
  + the base per-market assignment
  + runtime “How you run” protocol
  + live tool documentation from GET /tools
  + freshly rendered reflective-memory block
```

### 5.1 Base prompt parity

This candidate has no `strategy.md`. Its task instruction and per-market
assignment are the same as `tmA-base`. Before the first curation, the only
additional prompt content is an empty reflective-memory block explaining that
there are no lessons yet and that delayed prices are retrospective feedback,
not confirmation for a current alert.

### 5.2 Per-market instruction

`main.py::write_instruction` creates:

```text
instructions/m-<market-id>.md
```

It concatenates the environment instruction and the same dynamically generated
assignment as `tmA-base`: market id, question, and window.

Nothing in this assignment is hard-coded for the search roster. It is derived
from the live market row.

### 5.3 Runtime protocol and tool descriptions

`runtime/agent.py::Agent.system_prompt` reads the generated instruction and
appends:

- the JSON-only tool-call response contract;
- the instruction to call `sleep` when idle;
- the currently provisioned tool descriptions obtained from `GET /tools`.

### 5.4 Dynamic reflective block

`main.py::build_agent` passes a `block_fn` that calls:

```python
reflective.render_memory(agent_name)
```

`Agent.system_prompt()` invokes this function on every LLM turn. It is not
rendered once at agent creation, so a lesson written by today's curator is
visible on the next model call without rebuilding the agent.

The block contains:

- at most six global generalized lessons;
- the five most recent inferred outcomes for that particular market agent;
- up to three of that agent's claims still awaiting mature review;
- a final safety statement that prices are retrospective only and must never
  be used as pre-notification confirmation.

## 6. Per-market tool wiring

`main.py::market_tools` begins with the normal registry produced by
`runtime.agent.env_tools(env)`. It then applies these layers:

```text
environment tool
  -> market ownership/waiter binding
  -> reflective capture wrapper
  -> Agent.turn() dispatch
```

### 6.1 Tools removed from market agents

Program-owned scheduling/party tools are removed from each market agent:

```text
get_crontab, set_crontab, run_at, set_party
```

The agents retain the TM-A `sleep` tool.

### 6.2 Sleep wrapper

The sleep wrapper adds:

```json
{"waiter_id": "m-<market-id>"}
```

It also stores the most recent wait arguments in the baseline `state.json`
structure. The actual clock movement remains server-owned.

### 6.3 Search wrapper

The `search_news` wrapper:

1. calls the original environment tool;
2. records returned article id/title/domain/publish-time metadata in
   `reflective_state.json`;
3. writes a compact `search` observability event;
4. returns the unmodified result to the agent.

### 6.4 Article wrapper

The `get_article` wrapper updates article metadata and writes an `article`
event. It does not copy the full article text into the observability log or
reflective state.

### 6.5 Notification wrapper

After a successful `notify`, the wrapper records:

- agent and market;
- cited article id and available metadata;
- direction;
- server-returned notification time;
- `reviewed: false`.

Rejected notifications are not added to reflective action state. The agent's
ordinary tool-result event still records that the call failed.

The wrappers do not alter the environment's tool result, score, billing, or
standing-claim rules.

## 7. Structured observability

### 7.1 Destination

All added events go to:

```text
workspace/logs/observability.jsonl
```

Each line is a JSON object. Common fields are:

```json
{
  "real_time": "...",
  "thread": "m-...",
  "event": "tool_result",
  "agent": "m-...",
  "sim_time": "..."
}
```

`real_time` is useful for diagnosing model latency and thread overlap;
`sim_time` is the frozen simulated timestamp relevant to behavior.

### 7.2 Event source and event types

`runtime/agent.py` emits lifecycle events through its optional `event_fn`:

| Event | Meaning | Selected fields |
|---|---|---|
| `wake` | Agent received the bootstrap/process wake. | trigger id/kind, message count, reminder presence |
| `llm_request` | Immediately before a market-agent LLM call. | turn, message count, calls since wait |
| `llm_reply` | Model reply parsed or failed validation. | valid, selected tool, thought, turn |
| `llm_error` | LLM proxy raised an environment error. | error |
| `tool_result` | A selected tool completed or failed. | tool, args, ok, error |
| `wait_advanced` | A valid wait advanced simulated time. | tool, wake time, reminder presence |
| `compaction` | Old wake segments were removed from live context. | dropped and remaining message counts |
| `agent_exit` | Runaway or budget guard ended an agent loop. | reason |

`reflective.py` emits domain/reflection events:

| Event | Meaning |
|---|---|
| `search` | Query and number of results captured. |
| `article` | Article metadata captured. |
| `notification` | A successful notification entered action state. |
| `curation` | Daily review completed; includes claim/move counts and whether memory changed. |
| `feedback_error` | A price query used for review failed. |
| `curation_error` | The curator's LLM call failed. |
| `curator_exception` | Unexpected review error caught by the curator thread. |
| `curator_env_error` | The curator's wait call failed. |

Curator LLM calls go directly through `runtime.llm_client.chat_json`, so they
appear in the server's `ledger.jsonl` and `llm_log.jsonl`, but not as local
`llm_request` / `llm_reply` events. A successful daily curation is the local
summary of that call.

### 7.3 Failure isolation and size controls

Observability is intentionally best-effort:

- `Agent._observe` catches callback exceptions;
- `reflective.observe` catches all logging exceptions;
- strings are clipped to 1,200 characters;
- lists are clipped to 30 items;
- full article bodies and full LLM prompts/responses are not duplicated.

A logging failure should therefore not crash or change an agent decision.
For full LLM bodies, use the server-owned `llm_log.jsonl`.

## 8. Reflective state

### 8.1 Destination and schema

Persistent reflective data lives at:

```text
workspace/memory/reflective_state.json
```

The top-level schema is:

```json
{
  "articles": {
    "<news-id>": {
      "title": "...",
      "domain": "...",
      "published": "..."
    }
  },
  "actions": [
    {
      "agent": "m-<market-id>",
      "market_id": "...",
      "news_id": "...",
      "title": "...",
      "published": "...",
      "direction": "up",
      "at": "...",
      "reviewed": false
    }
  ],
  "feedback": [],
  "lessons": [],
  "diagnosis": "",
  "last_review": null,
  "curations": 0
}
```

Retention limits in `reflective.py` are:

- 1,200 article metadata rows;
- 160 feedback rows;
- six lessons.

Actions are retained for the run so later review and prompt rendering can
refer to their status.

### 8.2 Atomic persistence and locking

`reflective.py` uses a process-wide reentrant lock around state and log
writes. State is written to:

```text
memory/reflective_state.json.tmp
```

and then atomically replaced with `os.replace`. This avoids a curator or
market thread observing a partially written JSON document.

Price calls and curator LLM calls occur outside the state lock. Before saving
review results, the curator reloads state and merges actions added by market
threads during the review.

The older `state.json` used by baseline code has its own lock in `main.py`.
It is not the reflective-memory file.

## 9. Daily curation pipeline

The curator first sleeps for 24 simulated hours and then reviews once per
simulated day; the final sleep reaches the experiment end and returns
without attempting another review.

Each curation runs this pipeline:

```text
load state
  |
  +-- review every unreviewed claim at least 24h old
  |     `-- query delayed prices over its claim window
  |
  +-- scan each market since the previous curation
  |     `-- identify durable price displacements
  |
  +-- append heuristic feedback and mark mature claims reviewed
  |
  +-- ask the same metered LLM to curate generalized lessons
  |
  +-- filter unsafe lessons and atomically save memory
  |
  `-- write one curation observability event
```

The curation LLM uses the same `ENV_MODEL`, `/llm` proxy and wallet as the
market agents; reflection is not free (one metered call per curation).

## 10. How heuristic feedback is inferred

### 10.1 Durable-move detector

The environment returns sparse price changes. `reflective._price_points`
normalizes them into `(time, price)` pairs.

`reflective._jumps` calculates the absolute change between adjacent visible
points and a generic threshold:

```text
max(0.02, 6 * median absolute adjacent change)
```

A candidate displacement is retained only when:

1. its absolute change exceeds that threshold;
2. at least 30 minutes of later price data exists;
3. the price still has the same sign of displacement after 30 minutes;
4. at least 70% of the original displacement remains.

Signals occurring within six hours are grouped, keeping the largest. Without
the persistence test, raw adjacent-price jumps mistake short-lived quote
flicker for missed moves.

This detector is intentionally not the scorer's private breakpoint algorithm.
Its output is approximate actor-side feedback.

### 10.2 Mature claim review

Once a claim is at least 24 hours old, the curator requests prices from one
hour before the claim through its 24-hour window. It classifies the claim as:

- `likely covered a directional move` when a durable same-direction move was
  seen after the claim;
- `likely wrong direction` when only an opposite durable move was seen;
- `likely false alarm` when neither was seen.

These labels do not know causal article attribution or the scorer's exact
breakpoint boundaries. They are stored as `claim_review` feedback.

### 10.3 Possible missed-move review

The curator also scans prices since its prior review. For each durable move,
it looks for an actor claim on the same market, in the same direction, made
within the preceding 24 hours.

The resulting `move_review` is either:

- `likely anticipated move`; or
- `possible missed move`.

A stable `market@time` key prevents the two-hour overlap between daily scans
from duplicating an already recorded move.

## 11. LLM memory curation and safeguards

The curation prompt receives:

- the previous lesson list;
- up to 30 new heuristic feedback rows.

It asks for at most six short generalized rules and a short diagnosis. It
explicitly forbids dates, market ids, article ids, proper names, exact price
levels, and event-specific hints.

An important failure mode: a curator may suggest waiting for a sustained
market move before issuing a claim. That is invalid because a notification
must precede the move.

The current source has two defenses:

1. The curation system prompt says prices are retrospective evaluation only.
2. `_safe_lesson` rejects lessons combining price/movement language with
   prerequisite language such as “confirm,” “require,” “wait for,” or
   “before issuing.”

The final line of every rendered memory block repeats the constraint after
the learned bullets, ensuring the safety rule has the most recent position in
that block.

## 12. Context and compaction interaction

Each market has its own transcript and uncompacted history:

```text
logs/transcript_m-<market-id>.jsonl
logs/history_m-<market-id>.jsonl
```

The existing deterministic compactor retains the newest complete wake
segments within `CONTEXT_TOKENS`. Reflective lessons are not stored only in a
transcript segment; they are rendered into the system prompt on every turn.
They therefore survive transcript compaction.

The observability `compaction` event shows when and how much history was
dropped from live context. The uncompacted history remains available for
post-run analysis.

## 13. Tunable controls

| Control | Current value | Location | Effect |
|---|---:|---|---|
| Curation cadence | 24 hours | `main.py::CURATION_HOURS` | Code-enforced curator deadline. |
| Lessons in prompt | 6 | `reflective.py::MAX_LESSONS` | Controls memory size and curation output. |
| Article metadata cap | 1,200 | `reflective.py::MAX_ARTICLES` | Bounds cached title/domain/publish metadata. |
| Feedback cap | 160 | `reflective.py::MAX_FEEDBACK` | Bounds historical inferred feedback. |
| Absolute move floor | 0.02 | `reflective.py::_jumps` | Generic minimum displacement for hindsight review. |
| Noise multiplier | 6 | `reflective.py::_jumps` | Raises the threshold in volatile markets. |
| Persistence test | 30 minutes, 70% | `reflective.py::_jumps` | Rejects quickly reversing quote flicker. |
| Move deduplication | 6 hours | `reflective.py::_jumps` | Collapses one repricing episode. |
| Own feedback shown | 5 rows | `reflective.py::render_memory` | Per-market prompt detail. |
| Pending claims shown | 3 rows | `reflective.py::render_memory` | Reminds an agent about unresolved local actions. |

Agent cadence is unchanged from `tmA-base`; malformed behavior is still
governed by the existing 200-calls-per-wake guard in `Agent`.

## 14. How to inspect a run

Set a shell variable to a run directory, for example:

```sh
RUN=runs/task_breakout_news_pm/gpt-5.6-luna/<run-id>
```

### Result and resource summary

```sh
jq '{performance,resources,flags}' "$RUN/results.json"
```

### Count observability events

```sh
jq -r '.event' "$RUN/workspace/logs/observability.jsonl" \
  | sort | uniq -c | sort -nr
```

### Show daily curator outcomes and errors

```sh
jq -c 'select(.event | test("curation|exception|error"))' \
  "$RUN/workspace/logs/observability.jsonl"
```

### Read the final learned memory

```sh
jq '{curations,last_review,lessons,diagnosis,
     actions:(.actions|length),feedback:(.feedback|length)}' \
  "$RUN/workspace/memory/reflective_state.json"
```

### Inspect one market's decisions

```sh
jq -c 'select(.agent=="m-<market-id>")' \
  "$RUN/workspace/logs/observability.jsonl"
```

### Find failed tool calls

```sh
jq -c 'select(.event=="tool_result" and .ok==false)' \
  "$RUN/workspace/logs/observability.jsonl"
```

### Compare local agent calls with server LLM billing

```sh
jq -r 'select(.event=="llm_request") | .agent' \
  "$RUN/workspace/logs/observability.jsonl" | wc -l

jq '.resources.counts.llm' "$RUN/results.json"
```

The server total should exceed the market-agent request count by the number
of successful curator LLM calls.

## 15. Test evidence

`tests/tasks/breakout_news_pm/test_ext_tmb_reflection_candidate.py` and
`test_ext_tmd_reflection_candidate.py` pin the TM-B and TM-D ports of this
mechanism to their sources and run each end to end with a fake LLM; the TM-A
program is exercised the same way through the ext: provisioning tests in
`tests/harness/test_task_scaffolds.py`.
