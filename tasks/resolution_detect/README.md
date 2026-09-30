# resolution_detect

Market-resolution **detection** over Polymarket questions with billed
CC-NEWS evidence: the agent watches a roster of standing questions and
commits at most ONE claim per question, ever — "this question's outcome
is already decided in the world" — scored as event-settled TC-F1 in
bnpm's shape. The rung below forecasting on the ladder: forecasting skill
earns *nothing* here by construction (any pre-settlement claim is a false
alarm, however good the forecast), and the anti-anticipation cliff
mirrors bnpm's anti-reactivity cliff with the clock shifted
(world-settled → official resolution vs cause-published → move start).

## Task structure

- **World**: the bnpm 2026-03→07 substrate — CC-NEWS BM25 index
  (`tasks/breakout_news_pm/news/tantivy_index_v3`, publish-time
  visibility) plus the far-deadline roster (`data/sample_detect_v1.jsonl`,
  609 questions: 700 sampled hindsight-free at `scheduled_end ≥
  sim_end+30d` minus a documented 14-row easy-at-activation exclusion and
  a 77-row crypto-FDV exclusion, `data/README.md`) → `data/build.py` →
  `data/built/`. Every in-window resolution is event-triggered by
  construction; ~80% of the roster never resolves in-window (the quiet
  majority — the precision pressure). No price surface exists anywhere;
  prices drive only the build-time ground truth.
- **Questions** are visible from their real market-open date (staggered
  arrivals) and carry NO resolution signal of any kind: no outcome or
  resolution time, no `resolved` status flip, no already-resolved reject
  — a deployed resolution-detection agent has no oracle saying "this is
  settled"; that determination IS the task. Resolved questions stay
  listed indistinguishable from open ones; post-resolution claims are
  accepted and priced by the same 48 h decay clock (copying pays no more
  than claiming just before the close, which was always possible).
  Consequence: unclaimed questions settle only at run end (a late claim
  must stay recordable), so mid-run `close_due` books only claimed
  questions.
- **News clock**: the index bakes each article's `pub_ts` at build time
  as the self-reported publish time when the CC-NEWS crawl corroborates
  it within 2 h, otherwise crawl time − 2 h (one crawl cycle before
  discovery — the tightest bound the crawl supports); self-reports inside
  the crawler outage of 2026-04-01 04:00 → 04-05 20:00 UTC (the window's
  single crawl silence) are trusted as-is, since no corroboration was
  possible there (`news/build_index.py --ts-v3`, `V2_CORROBORATION_S` /
  `OUTAGE_START` / `OUTAGE_END`). Baked in, so range queries,
  `earliest_match` and pollers all follow this clock natively; the engine
  has no timestamp logic.
- **Observation tools** (env/apps.py): `list_questions(status?,
  added_after?)` compact index, `get_question(id)` detail (verbatim
  resolution criteria), `get_marks(id?)` own-claim echo — all free.
  Neither endpoint reveals the scheduled close date (with it visible,
  agents clock-watch deadline-wave questions and claim at the deadline
  instead of at determination); `search_news` / `get_article` billed at
  bnpm's commercial anchor rates on the shared "news" limiter.
- **Action**: `mark_outcome(question_id, outcome, news_id?)`, free, ONE
  accepted claim per question for the run, no retraction. Free-rejected
  (claim not consumed) on unknown/unactivated id (byte-identical
  errors), already-claimed question, outcome not in the outcome set, or
  an unknown/unpublished cited article. No already-resolved reject (it
  would be a resolution-status probe channel; the decay clock prices
  post-resolution copying instead). `news_id` is recorded, unscored
  (premature-claim forensics).

## Ground truth & scoring

`t_det` = the price's dip-tolerant entry into the winning outcome's
0.99 band (final entry judged against a 0.95 exit band), computed once
at build time and pinned into `data/built/questions.jsonl`.
`t_res = min(closedTime, scheduled_end)`. A claim (τ, outcome) settles
at the question's resolution under the **soft credit rule** — both
decay clocks fixed, independent of the question's gap (`task.py`:
`EARLY_DECAY_S` 12 h / `LATE_DECAY_S` 48 h):

| category | rule | credit |
|---|---|---|
| covered | outcome = y, τ ≥ t_det | exp(−(τ − t_det)/48 h) |
| early | outcome = y, τ < t_det, winnable | exp(−(t_det − τ)/12 h) |
| fa_premature | no determination exists (incl. every never-resolving question, settled at run end) | 0 |
| fa_wrong | outcome ≠ y | 0 |
| miss | winnable, unclaimed | 0 |

`tc_f1` (primary, max) = harmonic mean of recall = Σcredits/n_winnable
and precision = Σcredits/n_claims_settled (credit-weighted); `cov_f1`
(binary, covered only) is the bnpm-comparable companion. The early/late
asymmetry (4× per hour) keeps detection, not forecasting, the paying
skill; the quiet majority — claims on never-resolving questions are
hard FAs — carries the precision pressure, so the oracle optimum stays
at mark@0.99 and speculation still dies (see anchors).

## Anchors (full window, scripted actors through the real scorer — pinned in tests)

silence **0.000** · blanket-No @activation **0.014** (56 early claims
earn exp-decayed slivers; 59 wrong + 492 quiet FAs, zero covered) ·
one-shot price-band follower mark@0.95 0.166 · mark@0.99 **0.616** (the
quiet majority annihilates price-extremity following — most of its FAs
are longshot touches on questions whose world never settled; its 11
winnable-too-early burns are episodes early, credit ≈ 0) · +6h-late
syncer 0.882 · +24h-late syncer 0.607 (closed-form exp(−d/48 h): with
resolutions invisible, post-resolution claims are accepted and
decay-priced, so fixed-delay syncers cover everything) · perfect
(claim at t_det) **1.000**. The silent baseline (`agent/baselines/silent`)
is the runnable floor: 0.000, $0.

## Layout

- `data/` — frozen sample + `build_sample.py` (roster protocol + the two
  exclusions) + `build.py` (built world) + `audit_sample.py` (roster
  audit; reference t_det implementation) + README with the pinned audit
  block
- `task.py`, `env/apps.py`, `INSTRUCTION.md` (contract only — how
  "decided" is measured is never stated), `agent/` (`cron_react` main,
  TM-B example gatekeeper, silent baseline), `configs/cells/{e1,e2,e3}/`
  (three 60-question episodes)
- tests: `tests/tasks/resolution_detect/` (settlement math, free-reject
  matrix, leakage walk, calibration pins incl. anchor replication, e2e
  silent baseline)
