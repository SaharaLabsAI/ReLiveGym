# Weather data

Hourly 2 m temperature CSVs exported from [open-meteo](https://open-meteo.com/)
(historical weather API), UTC timestamps, °C. The files are gitignored (see
repo .gitignore); re-download from https://open-meteo.com/en/docs/historical-weather-api
with hourly `temperature_2m`, the coordinates below, and the full date range.

| file | location | coordinates | coverage |
|---|---|---|---|
| `open-meteo-34.06N118.24W91m.csv` | LA | 34.06 N, 118.24 W, 91 m | 2016 – 2026, hourly |
| `open-meteo-52.55N13.41E38m.csv` | Berlin | 52.55 N, 13.41 E, 38 m | 2016 – 2026, hourly |

Format: open-meteo CSV export — a metadata header, then a `time,temperature_2m (°C)`
table. The loader (`tasks/weather_fixture/task.py`) requires a contiguous hourly grid.

Calibration facts pinned by `tests/tasks/weather_fixture/test_calibration.py`:
LA @ 33 °C over 2021-06-01 → 2024-06-01 has 111 crossing days, Aug/Sep/Oct-heavy.
