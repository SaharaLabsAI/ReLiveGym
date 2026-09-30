# tmB-reflection-only

The TM-B counterpart of `tmA-reflection-only`: the ext: TM-B
program `tmB-base` (`main.py` = `scaffolds/per_market_main.py` plus the
client-side TM-B affordances of `local_tools.py`) with the reflective
memory mechanism of `../tmA-reflection-only` wired in. Its
`ARCHITECTURE.md` describes the mechanism; this file records only what
differs here.

Files:

- `main.py` — `tmB-base/main.py` plus the reflection hunks of
  `tmA-reflection-only/main.py` (capture wrappers on `search_news`,
  `get_article`, `notify`; the `curator` party member; per-turn memory
  block; `event_fn` observability), unchanged where they could be.
- `local_tools.py` — byte-identical to `tmB-base`.
- `reflective.py` — `tmA-reflection-only/reflective.py` plus one function,
  `register_program_call`, routing program-side calls to the same three
  captures. Curation, feedback inference, lesson filtering and rendering
  are untouched.
- `runtime/` — the current `scaffolds/runtime` with two edits:
  `agent.py` carries the `event_fn` hook of `tmA-reflection-only`
  (identical file); `program.py` gains a module-level `CALL_OBSERVER`
  called after every billed envkit fetch.
- `authored_skill.md`, `jail_seed/` — the tm=B skill appendix
  (`harness/authored_skill.md`) and the seeded `sleep.py` /
  `example_gatekeeper.py`, which the constructor's provisioning writes
  into every TM-B jail; an ext: program carries its own copies.

Two TM-B specifics:

1. **Program-side capture.** Under TM-B the authored programs do most of
   the searching and a substantial share of the notifying, and those
   envkit calls never pass the agent's tool
   registry. `runtime.program.CALL_OBSERVER` is set to
   `reflective.register_program_call`, so a program's searches, article
   reads and notifications enter the reflective state under the owning
   waiter's name exactly like the agent's own tool calls. Without this the
   curator would review an incomplete action list and label anticipated
   moves as missed.
2. **Curator park.** TM-B has no sleep tool; the curator parks like the
   coordinator does — a `park.py` (`envkit.wait(envkit.deadline())`) in
   its own jail, run with `until` = the next review time. Under the
   constructor's provisioning the same goes through the environment's
   `run_program`; under a TM-A cell it falls back to `sleep`.

Every run starts with empty reflective state; nothing is seeded from
earlier runs. Cell `ALG=none`; the repo's learning stack is inactive.

Full-episode bases: `tasks/breakout_news_pm/configs/ext_bases/{w10,w13,w17}full-tmB-ext.yaml`
(the TM-B no-learning per-market cells with only `run_id` and
`agent.scaffold` changed).
Launch: README.md, "Memory arms".
