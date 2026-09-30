"""example_gatekeeper.py — a minimal, runnable wake-condition program:
sleep to just past the top of the hour, fetch the freshest candle, hand
over the parsed close for the hour's decision.

Copy and edit it (write_file / edit_file), dry-run with
run_program(path, validate=true), then run_program(path). Wake times
live in code — wait(...); run_program's `until` is only a hard timeout
backstop. Each run watches one hour and ends at handover, returning
control to your ReACT turn — re-run the same program for the next hour.

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
from datetime import timedelta

from envkit import handover, http_fetch, now, wait

SYMBOL = "BTC"       # binance spelling BTCUSDT; other venues' paths,
                     # params, and reply shapes: get_exchange_docs()
WAKE_MINUTE = 6      # the hour's first 5m candle has closed by then
RETRY_MINUTES = 3    # re-poll cadence while the fetch comes back unusable

t = now()
due = t.replace(minute=WAKE_MINUTE, second=0, microsecond=0)
if due <= t:
    due += timedelta(hours=1)
wait(due)
while True:
    res = http_fetch(venue="binance", path="/api/v3/klines",
                     params={"symbol": SYMBOL + "USDT",
                             "interval": "5m", "limit": 2})
    if res.get("status") == 200:
        try:
            rows = json.loads(res["body"])  # ascending klines
        except ValueError:
            rows = None
        if isinstance(rows, list) and rows:
            last = rows[-1]
            handover({"symbol": SYMBOL, "venue": "binance",
                      "open_time_ms": last[0], "close": float(last[4]),
                      "asof": now().strftime("%Y-%m-%dT%H:%M:%SZ")})
    wait(now() + timedelta(minutes=RETRY_MINUTES))
# no deadline handling needed: a trigger or run_program's `until` exits
