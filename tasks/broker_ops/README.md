# broker_ops

The broker task with a **fixed ground truth**. Three HTTP hosts, no task tools:

| host | access | what |
|---|---|---|
| `api` | read-only (browser, curl, watcher programs) | the broker's account API: `/api/events` (fills, margin calls, requirement changes, treasury wires, `?since=`), `/api/account`, `/api/orders`, `/api/notices` (`web/api_app.py`) |
| `mail` | read-only | the ops inbox: the PM's note and one-time login codes (`web/mail_app.py`) |
| `broker` | writable, login-gated, browser forms | positions, the PM's orders (read-only), notices, risk desk (`web/broker_app.py`, `web/templates/`) |

The PM's book is scripted (`world.canonical_schedule`): the desk algorithm
enters one day-order per symbol at `pm_orders.hours_utc`, `offset_pct` off
the mark, side alternating by placement, unfilled ones lapsing at the next
placement; fills are the first candle through the price; margin calls are
the first 5-minute instant with ratio < maintenance (no open call, cooldown
elapsed); the treasury wires the required amount at the deadline in every
run. The agent's actions (funding instruction, protective-stop
registration) are journaled and probed but move neither positions nor
cash, so the roster of duty instances is identical across runs of a config
(`results.json` → `task.roster`, `task.schedule`).

Duties (credit 1/0 each; primary `routine_score` = mean over the roster):

| routine | event | deadline |
|---|---|---|
| margin-call response (funding instruction ≥ required) | ratio < maintenance: prices or a hike | +`call_response_hours` (4 h) |
| protective stop for a fill (opposite side, qty > 0, trigger on the protective side of the mark) | one of the PM's orders fills | +`stop_after_fill_hours` (2 h) |

Sessions expire in sim time (30 idle min; daily cutoff 00:00 UTC); codes
last 10 min; 5 failed logins lock the account 24 h and every probe inside
the lockout scores 0 (`locked_out`).

Cells: `configs/cells/tm{A,B,D}-tlrnnone-signone-algnone.yaml`
(Apr 1 → Apr 15 2026, BTC + ETH, $20; 29 fills + 2 margin calls = 31
instances). Print the schedule: `python -m tasks.broker_ops.world <yaml>`.
Run as an episode with a browser actor (`../../README.md`, "Browser
tasks"); the grid: `scripts/run_web_episodes.py`.

Tests: `tests/tasks/broker_ops/` — the roster is the same whatever the
client does; a 25-minute reference client scores 1.000; a client that
never signs in scores 0 on the same roster; a lockout zeroes the probes
inside it; restore reproduces the fold.
