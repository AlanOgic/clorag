"""Pure pricing logic for Claude synthesis cost.

Reads per-MTok prices from settings_manager. Used by the search pipeline to
compute USD cost from Anthropic usage objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _SettingsLike(Protocol):
    def get_float(self, key: str) -> float: ...


@dataclass(frozen=True)
class CostBreakdown:
    """Per-call synthesis cost breakdown in USD."""

    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    total_cost_usd: float
    cache_hit_pct: float


def calculate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    settings: _SettingsLike,
) -> CostBreakdown:
    """Compute the USD cost for a single Anthropic call.

    Token counts follow the Anthropic SDK Usage object:
    - input_tokens: regular billed input (excludes cache-read and cache-write)
    - output_tokens: generated tokens
    - cache_read_tokens: input that hit the cache (cheap)
    - cache_creation_tokens: input that wrote to the cache (premium)

    cache_hit_pct is reported over the input side only.
    """
    input_price = settings.get_float("pricing.input_price_per_mtok")
    output_price = settings.get_float("pricing.output_price_per_mtok")
    cache_read_price = settings.get_float("pricing.cache_read_price_per_mtok")
    cache_write_price = settings.get_float("pricing.cache_write_price_per_mtok")

    mtok = 1_000_000
    input_cost = (input_tokens / mtok) * input_price
    output_cost = (output_tokens / mtok) * output_price
    cache_read_cost = (cache_read_tokens / mtok) * cache_read_price
    cache_write_cost = (cache_creation_tokens / mtok) * cache_write_price
    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    input_side_total = input_tokens + cache_read_tokens + cache_creation_tokens
    cache_hit_pct = (
        (cache_read_tokens / input_side_total) * 100.0
        if input_side_total > 0
        else 0.0
    )

    return CostBreakdown(
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        total_cost_usd=total,
        cache_hit_pct=cache_hit_pct,
    )
