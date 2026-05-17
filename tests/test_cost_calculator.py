"""Tests for cost_calculator pure pricing logic."""
from __future__ import annotations

import pytest

from clorag.core.cost_calculator import calculate_cost  # type: ignore[import-untyped]


class FakeSettings:
    """Stub for get_settings_manager().get_float() — avoids DB."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def get_float(self, key: str) -> float:
        return self._prices[key]


@pytest.fixture
def prices() -> FakeSettings:
    return FakeSettings({
        "pricing.input_price_per_mtok": 3.00,
        "pricing.output_price_per_mtok": 15.00,
        "pricing.cache_read_price_per_mtok": 0.30,
        "pricing.cache_write_price_per_mtok": 3.75,
    })


def test_basic_input_output_cost(prices: FakeSettings) -> None:
    """1M input + 1M output = $3 + $15 = $18."""
    result = calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        settings=prices,
    )
    assert result.input_cost == pytest.approx(3.00)
    assert result.output_cost == pytest.approx(15.00)
    assert result.cache_read_cost == 0.0
    assert result.cache_write_cost == 0.0
    assert result.total_cost_usd == pytest.approx(18.00)
    assert result.cache_hit_pct == 0.0


def test_cache_read_and_write_priced_separately(prices: FakeSettings) -> None:
    """500K cache-read + 200K cache-write at sonnet prices."""
    result = calculate_cost(
        input_tokens=100_000,
        output_tokens=50_000,
        cache_read_tokens=500_000,
        cache_creation_tokens=200_000,
        settings=prices,
    )
    assert result.input_cost == pytest.approx(0.30)
    assert result.cache_read_cost == pytest.approx(0.15)
    assert result.cache_write_cost == pytest.approx(0.75)
    assert result.output_cost == pytest.approx(0.75)
    assert result.total_cost_usd == pytest.approx(1.95)


def test_cache_hit_pct_computed_over_input_side(prices: FakeSettings) -> None:
    """Cache hit % = cache_read / (input + cache_read + cache_creation)."""
    result = calculate_cost(
        input_tokens=200,
        output_tokens=500,
        cache_read_tokens=800,
        cache_creation_tokens=0,
        settings=prices,
    )
    assert result.cache_hit_pct == pytest.approx(80.0)


def test_zero_input_side_yields_zero_cache_pct(prices: FakeSettings) -> None:
    """Avoid division by zero when no input-side tokens at all."""
    result = calculate_cost(
        input_tokens=0,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        settings=prices,
    )
    assert result.cache_hit_pct == 0.0


def test_breakdown_is_immutable_dataclass(prices: FakeSettings) -> None:
    """CostBreakdown should be a frozen dataclass."""
    result = calculate_cost(
        input_tokens=100,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        settings=prices,
    )
    with pytest.raises(AttributeError):
        result.total_cost_usd = 999.0
