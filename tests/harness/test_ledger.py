import json
from datetime import datetime, timezone

import pytest

from harness.ledger import Ledger

UTC = timezone.utc
T = datetime(2021, 6, 1, 12, tzinfo=UTC)


def test_append_and_totals(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("weather", T, cost=0.01, start="a", end="b")
    ledger.append("weather", T, cost=0.01)
    ledger.append("llm", T, cost=0.015, model="m")
    ledger.append("notify", T, day="2021-06-01")
    assert ledger.total_cost() == pytest.approx(0.035)
    assert ledger.cost_by_type() == {"weather": 0.02, "llm": 0.015, "notify": 0.0}
    assert ledger.count_by_type() == {"weather": 2, "llm": 1, "notify": 1}
    ledger.close()

    lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert first["seq"] == 1
    assert first["type"] == "weather"
    assert first["sim_time"] == "2021-06-01T12:00:00Z"
    assert first["start"] == "a"


def test_monthly_cost_buckets():
    ledger = Ledger()
    ledger.append("weather", datetime(2021, 6, 5, tzinfo=UTC), cost=0.01)
    ledger.append("weather", datetime(2021, 7, 5, tzinfo=UTC), cost=0.01)
    ledger.append("llm", datetime(2021, 7, 6, tzinfo=UTC), cost=0.5)
    assert ledger.monthly_cost_buckets() == {"2021-06": 0.01, "2021-07": 0.51}


def test_keep_events_false_is_write_through_only(tmp_path):
    from datetime import datetime, timezone
    ledger = Ledger(tmp_path / "l.jsonl", keep_events=False)
    ledger.append("x", datetime(2026, 1, 1, tzinfo=timezone.utc), cost=1.0, big="payload")
    ledger.append("x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert ledger.events == []
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["big"] == "payload"
    assert json.loads(lines[1])["seq"] == 2
