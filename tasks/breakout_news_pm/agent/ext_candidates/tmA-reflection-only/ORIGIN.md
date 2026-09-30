# tmA-reflection-only

For the detailed component map, control flow, prompt/tool wiring, persistent
state schemas, and curation algorithm, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

This program is the reflection-memory arm under TM-A (sleep): the per-market
base program `tasks/breakout_news_pm/agent/scaffolds/per_market_main.py`
(referred to as `tmA-base` below and in `ARCHITECTURE.md`) with a reflective
memory mechanism added, and nothing else changed. Relative to the base it adds
only:

- structured, write-only `logs/observability.jsonl` events;
- capture of article metadata and successful notifications as reflection
  evidence;
- a party-aware curator that wakes every simulated day;
- heuristic hindsight feedback inferred from delayed prices and prior claims;
- LLM-curated, generalized per-run memory injected into every agent prompt.

There is no static cadence, query, evidence-selection, or notification policy:
the market agents receive the same assignment as under the base program. The
reflective curation and rendering prompts contain safeguards that prevent
retrospective prices from being used as confirmation for current alerts; those
safeguards are intrinsic to the memory mechanism, not an operating strategy.

Every run starts with empty reflective state. Nothing is seeded from earlier
runs, ground truth, or held-out data. `runtime/actor.py` remains unchanged and
byte-identical to the shared `scaffolds/runtime/actor.py`.

Full-episode bases: `tasks/breakout_news_pm/configs/ext_bases/{w10,w13,w17}full-tmA-ext.yaml`
(the TM-A no-learning per-market cells with only `run_id` and `agent.scaffold`
changed). Launch: README.md, "Memory arms".
