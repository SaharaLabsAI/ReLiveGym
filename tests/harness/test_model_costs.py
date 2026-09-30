"""The global cost table, the model spec, and launch-time resolution
(harness/model_costs.py)."""

import pytest

import harness.model_costs as mc
from tests.conftest import make_config


def test_parse_model_spec():
    assert mc.parse_model_spec("gpt-5.6-luna") == ("openai", "gpt-5.6-luna")
    assert mc.parse_model_spec("openai:gpt-5.6-luna") == \
        ("openai", "gpt-5.6-luna")
    assert mc.parse_model_spec("openrouter:deepseek/deepseek-v4-pro") == \
        ("openrouter", "deepseek/deepseek-v4-pro")
    for bad in ("bedrock:m", "openrouter:"):
        with pytest.raises(ValueError):
            mc.parse_model_spec(bad)


def test_model_slug():
    assert mc.model_slug("gpt-5.6-luna") == "gpt-5.6-luna"
    assert mc.model_slug("deepseek/deepseek-v4-pro") == \
        "deepseek-deepseek-v4-pro"


def test_parse_costs_file_converts_and_validates(tmp_path):
    p = tmp_path / "costs.yaml"
    p.write_text("models:\n  m1: {in: 0.20, out: 1.20, cached_in: 0.02}\n"
                 "  m2: {in: 3.0, out: 15.0}\n")
    table = mc.parse_costs_file(p)
    assert table["m1"] == pytest.approx({"in": 2e-7, "out": 1.2e-6,
                                         "cached_in": 2e-8})
    assert table["m2"] == pytest.approx({"in": 3e-6, "out": 1.5e-5})

    p.write_text("models:\n  m: {in: 0.2, output: 1.2}\n")  # bad key
    with pytest.raises(ValueError):
        mc.parse_costs_file(p)
    p.write_text("models:\n  m: {in: 0.2}\n")  # out missing
    with pytest.raises(ValueError):
        mc.parse_costs_file(p)


def test_checked_in_table_parses():
    table = mc.parse_costs_file(mc.COSTS_PATH)
    for model, rates in table.items():
        assert 0 < rates["in"] < 1e-3, model  # $/token, not $/Mtok
        assert rates["out"] >= rates["in"], model
        if "cached_in" in rates:
            assert rates["cached_in"] < rates["in"], model


def test_resolve_rates_precedence(monkeypatch):
    monkeypatch.setattr(mc, "_cache", {"m": {"in": 1e-6, "out": 2e-6}})
    cfg = make_config(cost=dict(llm_token_rates={"m": {"in": 9e-6,
                                                       "out": 9e-6}}))
    assert mc.resolve_rates(cfg, "m")["in"] == 9e-6  # per-run override wins
    cfg = make_config()
    assert mc.resolve_rates(cfg, "m")["in"] == 1e-6  # global table
    assert mc.resolve_rates(cfg, "other") is None


def test_apply_launch_model(monkeypatch):
    monkeypatch.setattr(mc, "_cache", {"deepseek/deepseek-v4-pro":
                                       {"in": 1e-6, "out": 2e-6}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    cfg = make_config()
    mc.apply_launch_model(cfg, "openrouter:deepseek/deepseek-v4-pro")
    assert cfg.agent.provider == "openrouter"
    assert cfg.agent.model == "deepseek/deepseek-v4-pro"
    assert cfg.run_id == "test-run-deepseek-deepseek-v4-pro"

    # no model at all: launches untouched (silent baselines)
    cfg = make_config()
    mc.apply_launch_model(cfg, None)
    assert cfg.run_id == "test-run" and cfg.agent.model is None

    # unpriced model: fail fast
    cfg = make_config()
    with pytest.raises(SystemExit, match="no entry"):
        mc.apply_launch_model(cfg, "openai:unpriced-model")

    # priced but the provider key is missing: fail fast
    monkeypatch.delenv("OPENROUTER_API_KEY")
    cfg = make_config()
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        mc.apply_launch_model(cfg, "openrouter:deepseek/deepseek-v4-pro")


def test_yaml_model_spec_normalizes(monkeypatch):
    """agent.model in a yaml may carry the api-provider prefix; the
    config model splits it so agents only ever see the bare name."""
    cfg = make_config(agent=dict(scaffold="baseline_poller",
                                 model="openrouter:moonshotai/kimi-k3"))
    assert cfg.agent.provider == "openrouter"
    assert cfg.agent.model == "moonshotai/kimi-k3"
