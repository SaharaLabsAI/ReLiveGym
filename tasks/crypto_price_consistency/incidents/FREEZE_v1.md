# Incident stream v1 — FREEZE

Frozen **2026-08-04**, before any system under test was built.
Nothing below may be edited; any change to the generator, its parameters, or
the scenario files creates a `v2` sibling reported alongside this one.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `calibration/manifest.json` | `66e44354e57164bb102a629749a9a245e6e851d0dc54c9798f7a5956c76065bb` |
| `generator_params_v1.json` | `42ff9d51c9dae19e75d51e4c41e3c13cb55bd92f7abaa85bfd368ca453cbf896` |
| `limiters.yaml` | `97a5a62ec97f931835364052489f1c5e3fc057094a476db15c30334df76566e0` |
| `scenarios/manifest.json` | `cd4d3f1f686cd34e4cc317ccbb825f2cb9086b7a62e59292cf02401c99a8a995` |

Per-source calibration event hashes are inside `generator_params_v1.json`
(`calibration.event_file_sha256`); per-trace hashes are inside
`scenarios/manifest.json`. Regeneration is deterministic:
`python -m tasks.crypto_price_consistency.incidents.generate_traces`
reproduces every scenario file byte-identically (verified at freeze time).

## Trace set (68 traces, 88,246 events)

| Group | Files | Notes |
|---|---|---|
| `zero` | 1 | no incidents |
| `real_replay` | 1 | 46 archived events on true timestamps (coinbase/bitstamp/okx; CoinGecko's archive is genuinely quiet Mar–Jul 2026, latest event 2026-01-13) |
| `fitted_dev_s001–s020` | 20 | development; seeds 1–20 |
| `fitted_held_s101–s120` | 20 | held out — evaluate once per system version, never iterate against |
| `grid_r{05,1,4}_d{05,1,4}_{coff,con}` | 18 | rate × duration × correlation sweep; seeds 500–517 |
| `stress_*` | 8 | authored tail scenarios, pass/fail style |

## Evaluation discipline

- Develop against `fitted_dev_*`, `zero`, and `real_replay` only.
- `fitted_held_*` and `stress_*` results are computed **once** per system
  version and reported as-is.
- Robustness claim = ranking stability across the grid; any flip is
  reported, not hidden.
- `stress_maintenance_in_vol` pins Binance down on 2026-06-05, the window's
  highest-realized-volatility day (chosen from the recorded data by
  `generate_traces.highest_vol_day`, documented in the trace header).
