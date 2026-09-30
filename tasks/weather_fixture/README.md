# Weather high-temperature alerting

The harness's reference task: the test suite and its fixture agents run
against it (it is not a benchmark task). An agent watches an hourly
temperature stream and must notify each UTC day whose temperature crosses
a threshold, promptly and cheaply. This file is for experimenters; the
agent-facing task spec is [INSTRUCTION.md](INSTRUCTION.md) (a template —
`${...}` placeholders are rendered from the run config into the workspace
at run init).

## Task structure

- **Signal**: hourly 2 m temperature from a fixed location, replayed from
  historical data. Hour H's reading becomes visible at H+1 (1-hour data
  delay), so the best achievable notification delay is 1 h.
- **Action**: at most one notification per UTC calendar day, naming the day
  (`{"date": "YYYY-MM-DD"}`).
- **Ground truth**: a day is a *crossing day* iff any hour reaches
  `threshold_c`; the target time is the first such hour (h*). Ground truth is
  computed by the scorer from the full dataset and never exposed through the
  agent API.

## Evaluation

The run's score is **TC-F1** — the harmonic mean of precision (properly
notified crossing days / those + false-alarm days) and TC-recall (mean
timeliness credit `max(0, 1 − delay/credit_hours)` over crossing days).
Higher is better. Per-day settlement details in the [task.py](task.py)
docstring. Resources are constraints, not score: the `budget_usd` wallet
(LLM at real token rates; the weather API is free) and the documented
open-meteo quota (10k calls/day, enforced 429). Pinned anchor (baseline
poller, 2021-06 → 2024-06, `tests/tasks/weather_fixture/test_anchor.py`): tc_f1
0.8377, recall 0.991, precision 1.0.

A day closes `grace_hours` (default 24 h) after it ends; `/feedback` reveals
only closed days. Premature notifications (before the crossing data was
visible) are false alarms but do not consume the day — it can still be
properly notified.

## Dataset

`data/` holds open-meteo hourly CSVs (see [data/README.md](data/README.md) for
provenance and download): LA (34.06N, 118.24W) and Berlin (52.55N, 13.41E),
2016–2026. Calibration facts pinned by tests: the window 2021-06-01 →
2024-06-01 has **111 crossing days** at 33 °C, concentrated in Aug–Oct;
the window 2023-06-01 → 2023-09-01 has **24 crossing days**.

## Config

Task section of the run YAML (see [configs/](configs/)):

```yaml
task:
  name: weather_fixture
  location: LA            # or Berlin, or weather_csv: <path>
  data_cutoff: 2023-06-01T00:00:00Z   # earliest visible history
  threshold_c: 33.0
  # grace_hours: 24
  cost:
    weather_call: 0.01
    delay_factor: 0.02
    miss_cap: 11.52       # = delay_factor * 24²; also caps the delay penalty
    false_alarm: 11.52
    # duplicate_penalty: defaults to false_alarm / 10
```

`configs/baseline.yaml` runs the no-LLM baseline poller over two
weeks:

```
python -m harness.run --config tasks/weather_fixture/configs/baseline.yaml
```

## Baseline

`agent/baselines/baseline_poller/` — a daily 23:00 UTC poll, no LLM. The
task's tools are served by `env/apps.py` and discovered by programs
through the environment's manifest (no task client is copied into
workspaces).
