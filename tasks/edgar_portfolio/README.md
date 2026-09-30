# edgar_portfolio

Web-agent task: keep the
fund's filings sheet current from a replayed EDGAR. The world is two
HTTP hosts and **no task tools** — the env MCP carries only the clock,
the wait tool and the cost dashboard.

| host | access | what |
|---|---|---|
| `sec` | read-only, curl or browser | data.sec.gov's JSON endpoints as of the sim instant (`web/sec_app.py`, `edgar.py`) |
| `portal` | writable, browser (form) | the fund's filings sheet: holdings page, per-ticker filing records, audit log (`web/portal_app.py`, `web/templates/`) |

Scorable events: for each held ticker, the earnings 8-K (item 2.02) and
the 10-Q/10-K accepted inside the run window. Each is a grader probe at
its deadline (next NYSE open after acceptance, `edgar.py::next_market_open`;
2026 NYSE holidays pinned there). Credit 1/0 per event: the row keyed by
(ticker, accession) exists, matches the XBRL gold (form, acceptance time
±60 s, fy/fp, revenue ±0.05 %, diluted EPS ±0.005 — fields the filing
does not report are unscored), its last save lies in
[accepted_at, deadline), and the ledger holds a `web` row on host `sec`
for that CIK between acceptance and the save (source-visible-before-write).
Primary metric `filing_score` = mean credit (max). Report: per-form and
sharp (< 6 h to deadline) vs overnight scores, status counts
(ok / miss / wrong / premature / no_evidence), delays.

State model: `harness/webworld.py` — the journal holds the holdings sheet
(world-written: initial book + scripted `blotter`) and the filings sheet
(agent-written, each save one `notify` ledger row); the timeline runs the
blotter and the probes inside `close_due(now)`. World writes regenerate
from config on restore; agent writes replay from the ledger.

## Data

`data/raw/` (gitignored): one snapshot of `company_tickers.json`,
`submissions/CIK*.json` and `companyfacts/CIK*.json` for the roster in
`roster.py` — rebuild with `SEC_USER_AGENT="<name> <email>" python -m
tasks.edgar_portfolio.data.fetch_edgar` (see `data/README.md`). The
snapshot of 2026-09-16 yields 66 scorable filings over Apr 1 – Jun 1
2026 with the full roster held (deadline gaps: 19 events within 3 h,
most of the rest ~17 h).

## Config (`task:` section)

| key | meaning |
|---|---|
| `holdings` | ticker → shares at sim_start |
| `blotter` | scripted trades `[{at, ticker, action: buy|sell, shares}]` |
| `portal_access` | `browser` (default; the form is the only write path) or `api` (adds `PUT /portfolio/api/rows/<T>/<ACCN>` — the UI-tax control) |
| `sec_rate_limit` | `{"window": "none"}` by default |
| tolerances | `acceptance_tolerance_s`, `revenue_tolerance_frac`, `eps_tolerance` |

Cells: `configs/cells/tm{A,B,D}-tlrnnone-signone-algnone.yaml`
(33 holdings + a 4-trade blotter, Apr 1 → Jun 1 2026, $20).

## Running

Episodes only (a browser actor): `python scripts/prepare_episode.py
--task edgar_portfolio --tm A --app opencode --model openai:gpt-5.6-luna`
then the printed fenced `opencode run` line (TM-D: `python -m harness.mcp
drive`), and settle with `python -m harness.mcp settle` (`../../README.md`,
"Browser tasks"). The constructor's `react` program has no HTTP client, so
this task runs as episodes only.

## Tests

`tests/tasks/edgar_portfolio/` — as-of leak pins on every event, the
calendar, scoring anchors through a served sim (perfect client 1.000,
premature 0.000, a 07:00-ET daily poller = the pre-open miss rate),
determinism, restore, the non-browser write flag. They skip when the
snapshot is absent.
