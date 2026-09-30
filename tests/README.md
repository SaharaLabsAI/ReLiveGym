# Tests

```sh
pytest                 # default: fast tests only (~1 min) — anchor + slow deselected
pytest -m slow         # e2e over many process firings (cron cells, replays): a few minutes
pytest -m anchor       # the weather calibration anchor (~2 min)
pytest -m "slow or not slow"   # everything except the anchor
```

The default selection lives in `pyproject.toml` (`addopts`). The harness
suite alone (`pytest tests/harness`) takes about 10 s.

- `tests/harness/` — clock, schedule store, wait party, ledger and wallet,
  LLM proxy, tool manifest, the constructor and its mounts, reflection
  rendering (golden file under `golden/`).
- `tests/control_plane/` — the detached server/runner contract, the
  launcher, checkpoint/resume, episode mode (MCP bridge, cron driver,
  party episodes). Server processes are spawned for real; tests that need
  a provider key run against the mock LLM upstream with a placeholder key
  (`conftest.py`).
- `tests/tasks/<task>/` — scorers, visibility rules, env tools and one
  end-to-end smoke per program shape, on synthetic mini-worlds. Tests that
  need the real built data (`DATA.md`) skip when it is absent.
- `tests/fixture_agents/` — minimal actor programs (sleeper, poller,
  crasher, ...) that exercise the runner contract against `tasks/weather_fixture`.

The macOS network fence (`sandbox-exec`) is exercised where available;
on Linux the fenced tests skip.
