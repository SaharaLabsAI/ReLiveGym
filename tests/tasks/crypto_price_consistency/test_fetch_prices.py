from argparse import Namespace

import pytest

from tasks.crypto_price_consistency.data.fetch_prices import select_sources


def _args(*, source=None, exclude_source=None) -> Namespace:
    return Namespace(
        cohort="btc_usdt_spot",
        source=source,
        exclude_source=exclude_source,
    )


def test_default_selection_omits_opt_in_bybit() -> None:
    _, selected = select_sources(_args())

    assert [source.source for source in selected] == ["binance", "okx", "kucoin"]


def test_source_can_select_opt_in_bybit() -> None:
    _, selected = select_sources(_args(source=["bybit"]))

    assert [source.source for source in selected] == ["bybit"]


def test_exclude_source_filters_default_selection() -> None:
    _, selected = select_sources(_args(exclude_source=["okx"]))

    assert [source.source for source in selected] == ["binance", "kucoin"]


def test_source_cannot_be_included_and_excluded() -> None:
    with pytest.raises(ValueError, match="both included and excluded"):
        select_sources(_args(source=["okx"], exclude_source=["okx"]))
