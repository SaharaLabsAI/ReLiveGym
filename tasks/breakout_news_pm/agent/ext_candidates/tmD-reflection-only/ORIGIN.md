# tmD-reflection-only

The TM-D counterpart of `tmA-reflection-only` / `tmB-reflection-only`: the
per-market cron program `scaffolds/per_market_cron_main.py` (= `tmD-base`)
with the reflective memory mechanism of `../tmA-reflection-only` wired in.
Its `ARCHITECTURE.md` describes the mechanism; this file records only what
differs here.

Files:

- `main.py` — `per_market_cron_main.py` plus the reflection hunks of
  `tmA-reflection-only/main.py` (capture wrappers on `search_news`,
  `get_article`, `notify`; per-turn memory block; `event_fn`
  observability), unchanged where they could be.
- `reflective.py` — byte-identical to `tmA-reflection-only`.
- `runtime/` — the current `scaffolds/runtime`; `agent.py` carries the
  `event_fn` hook of `tmA-reflection-only` and nothing else.

One TM-D specific: **the curator is a cron row, not a party member.** TM-D
has no wait party and no resident process — every firing is its own
`python main.py` — so the 24-hour curator thread becomes a program-owned
crontab row, id `learn`, `0 0 * * *`, dispatched to `reflective.curate`.
The id is the one the cron mains already reserve for learning plumbing: the
agents' `list_schedules` hides it and they can neither create nor edit it.
A row never fires at its own install instant, so the first curation lands
at sim_start + 24 h and then daily — the instants of the TM-A/B curator —
and the 00:05 firings already see the lessons from the previous day. The
reflective state is re-read from `memory/reflective_state.json` on every
call, and the actor runs one firing at a time, so nothing else changes.

**Base cadence as a read-only row.** `READONLY_BASE = True` (top of
`main.py`): each market's 6-hourly cadence is a program-installed,
read-only row and the instruction says so — the cadence of the TM-D
no-learning per-market cells this arm is compared against. Agents still
create, update and delete schedules of their own. `False` gives the
agent-modifiable default schedule of `per_market_cron_main.py`.

The curator does not see schedule CRUD; curation prompt, feedback inference
and lesson filter are those of the TM-A/B candidates.

Every run starts with empty reflective state; nothing is seeded from
earlier runs. Cell `ALG=none`; the repo's learning stack is inactive.

Full-episode bases: `tasks/breakout_news_pm/configs/ext_bases/{w10,w13,w17}full-tmD-ext.yaml`
(the TM-D no-learning per-market cells with only `run_id` and
`agent.scaffold` changed).
Launch: README.md, "Memory arms".
