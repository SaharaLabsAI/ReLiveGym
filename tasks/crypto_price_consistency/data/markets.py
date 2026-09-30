"""Comparable spot-market cohorts and their venue-specific symbols."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketSource:
    source: str
    symbol: str
    market_type: str | None = None
    enabled_by_default: bool = True


@dataclass(frozen=True, slots=True)
class MarketCohort:
    name: str
    description: str
    base: str
    quote: str
    market_type: str
    sources: tuple[MarketSource, ...]
    default_source_interval_minutes: int = 1
    default_output_interval_minutes: int = 5


COHORTS: dict[str, MarketCohort] = {
    "btc_usdt_spot": MarketCohort(
        name="btc_usdt_spot",
        description="Bitcoin/Tether spot prices across keyless CEX APIs",
        base="BTC",
        quote="USDT",
        market_type="spot",
        sources=(
            MarketSource("binance", "BTCUSDT"),
            MarketSource("okx", "BTC-USDT"),
            MarketSource("kucoin", "BTC-USDT"),
            MarketSource("bybit", "BTCUSDT", enabled_by_default=False),
        ),
    ),
    "eth_usdt_spot": MarketCohort(
        name="eth_usdt_spot",
        description="Ether/Tether spot prices across keyless CEX APIs",
        base="ETH",
        quote="USDT",
        market_type="spot",
        sources=(
            MarketSource("binance", "ETHUSDT"),
            MarketSource("okx", "ETH-USDT"),
            MarketSource("kucoin", "ETH-USDT"),
            MarketSource("bybit", "ETHUSDT", enabled_by_default=False),
        ),
    ),
    "btc_usd_spot": MarketCohort(
        name="btc_usd_spot",
        description="Bitcoin/US dollar spot and reference prices from keyless APIs",
        base="BTC",
        quote="USD",
        market_type="spot",
        sources=(
            MarketSource("coinbase", "BTC-USD"),
            MarketSource("bitstamp", "btcusd"),
            MarketSource("defillama", "coingecko:bitcoin", "reference"),
        ),
    ),
    "eth_usd_spot": MarketCohort(
        name="eth_usd_spot",
        description="Ether/US dollar spot and reference prices from keyless APIs",
        base="ETH",
        quote="USD",
        market_type="spot",
        sources=(
            MarketSource("coinbase", "ETH-USD"),
            MarketSource("bitstamp", "ethusd"),
            MarketSource("defillama", "coingecko:ethereum", "reference"),
        ),
    ),
    "btc_usd_hourly": MarketCohort(
        name="btc_usd_hourly",
        description="Hourly Bitcoin/USD spot and aggregate reference prices",
        base="BTC",
        quote="USD",
        market_type="spot",
        sources=(
            MarketSource("coinbase", "BTC-USD"),
            MarketSource("bitstamp", "btcusd"),
            MarketSource("defillama", "coingecko:bitcoin", "reference"),
            MarketSource("coingecko", "bitcoin", "reference"),
        ),
        default_source_interval_minutes=60,
        default_output_interval_minutes=60,
    ),
    "eth_usd_hourly": MarketCohort(
        name="eth_usd_hourly",
        description="Hourly Ether/USD spot and aggregate reference prices",
        base="ETH",
        quote="USD",
        market_type="spot",
        sources=(
            MarketSource("coinbase", "ETH-USD"),
            MarketSource("bitstamp", "ethusd"),
            MarketSource("defillama", "coingecko:ethereum", "reference"),
            MarketSource("coingecko", "ethereum", "reference"),
        ),
        default_source_interval_minutes=60,
        default_output_interval_minutes=60,
    ),
}


def get_cohort(name: str) -> MarketCohort:
    try:
        return COHORTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(COHORTS))
        raise ValueError(f"unknown cohort {name!r}; choose one of: {choices}") from exc
