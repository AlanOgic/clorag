"""Default RAG fine-tuning settings registry for CLORAG.

This module contains all hardcoded default settings for retrieval, reranking,
synthesis, caches, and prefetch parameters. These serve as fallbacks when
settings are not found in the database.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SettingDefinition:
    """Definition of a default RAG fine-tuning setting."""

    key: str
    name: str
    description: str
    category: str
    value_type: str  # "int", "float", "bool"
    default_value: str
    min_value: float | None
    max_value: float | None
    requires_restart: bool


# =============================================================================
# DEFAULT SETTINGS REGISTRY
# =============================================================================

DEFAULT_SETTINGS: list[SettingDefinition] = [
    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    SettingDefinition(
        key="retrieval.short_query_threshold",
        name="Short Query Threshold",
        description=(
            "After reranking, chunks scoring below this value are discarded"
            " for 1–2 word queries (e.g. 'RIO', 'tally'). Reranker scores"
            " range 0–1. At 0.15, most loosely related chunks survive."
            " Raise to tighten; lower to show more results for vague queries."
        ),
        category="retrieval",
        value_type="float",
        default_value="0.15",
        min_value=0.0,
        max_value=1.0,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.medium_query_threshold",
        name="Medium Query Threshold",
        description=(
            "Post-rerank score cutoff for 3–5 word queries (e.g. 'RIO"
            " firmware update'). Multi-word queries produce sharper reranker"
            " scores, so a higher cutoff filters noise without losing good"
            " matches. Applied only when reranking is enabled; skipped"
            " otherwise (RRF scores are uncalibrated)."
        ),
        category="retrieval",
        value_type="float",
        default_value="0.20",
        min_value=0.0,
        max_value=1.0,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.long_query_threshold",
        name="Long Query Threshold",
        description=(
            "Post-rerank score cutoff for 6+ word queries (e.g. 'how to"
            " connect a Sony FX6 via SDI to a RIO'). Long queries carry"
            " strong intent, so this is the strictest threshold — chunks"
            " must be highly relevant to survive. Still capped by"
            " Max Threshold Cap."
        ),
        category="retrieval",
        value_type="float",
        default_value="0.25",
        min_value=0.0,
        max_value=1.0,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.technical_term_bonus",
        name="Technical Term Bonus",
        description=(
            "Added to the base threshold when the query contains any of"
            " 30+ known technical terms (VISCA, SDI, HDMI, NDI, SRT, PTZ,"
            " tally, etc.). A query like 'VISCA over IP' gets threshold"
            " 0.20 + 0.05 = 0.25, discarding generic chunks that mention"
            " VISCA only in passing. Set to 0 to disable the bonus."
        ),
        category="retrieval",
        value_type="float",
        default_value="0.05",
        min_value=0.0,
        max_value=0.5,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.max_threshold_cap",
        name="Max Threshold Cap",
        description=(
            "Hard ceiling on the final threshold after base + technical"
            " bonus are summed. Prevents a long technical query from"
            " reaching e.g. 0.25 + 0.05 = 0.30+, which would over-filter"
            " and return too few chunks. The result is: min(base + bonus,"
            " this cap). Raise only if you see too many low-quality chunks."
        ),
        category="retrieval",
        value_type="float",
        default_value="0.30",
        min_value=0.1,
        max_value=1.0,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.min_guaranteed_results",
        name="Min Guaranteed Results",
        description=(
            "Safety net: if threshold filtering leaves fewer than this many"
            " chunks, the top N by reranker score are kept anyway. Ensures"
            " Claude always has context to work with, even for niche queries"
            " where all scores are low. Only applies when reranking is"
            " active. Set to 1 for strict filtering, higher for safety."
        ),
        category="retrieval",
        value_type="int",
        default_value="3",
        min_value=1,
        max_value=20,
        requires_restart=False,
    ),
    SettingDefinition(
        key="retrieval.overfetch_multiplier",
        name="Overfetch Multiplier",
        description=(
            "Multiplied by the requested result count to decide how many"
            " chunks to fetch from Qdrant before the Voyage reranker trims"
            " them. With limit=10 and multiplier=3, fetches 30 chunks,"
            " reranks all 30, keeps the best 10. Higher values give the"
            " reranker more candidates (better quality) but cost more"
            " Voyage API tokens and add ~50ms per extra 10 chunks."
        ),
        category="retrieval",
        value_type="int",
        default_value="3",
        min_value=1,
        max_value=10,
        requires_restart=False,
    ),
    # -------------------------------------------------------------------------
    # Reranking
    # -------------------------------------------------------------------------
    SettingDefinition(
        key="reranking.top_k",
        name="Rerank Top-K",
        description=(
            "After Voyage rerank-2.5 scores all overfetched chunks, only"
            " the top K are kept. These surviving chunks are then threshold-"
            "filtered and sent to Claude for synthesis. Increasing this"
            " gives Claude more context but increases API cost and response"
            " time. Interact with Overfetch Multiplier: overfetch feeds"
            " candidates in, top_k decides how many come out."
        ),
        category="reranking",
        value_type="int",
        default_value="5",
        min_value=1,
        max_value=50,
        requires_restart=False,
    ),
    SettingDefinition(
        key="reranking.source_diversity_threshold",
        name="Source Diversity Threshold",
        description=(
            "After RRF merge across the 3 collections (docs, cases, custom),"
            " if a collection has no result in the top set, its best chunk"
            " replaces the lowest-scoring result — but only if that chunk"
            " scores at least this fraction of the #1 result. At 0.5, a"
            " collection needs ≥50% of the top score to earn a slot."
            " Set to 0 to always force diversity; 1.0 to disable it."
        ),
        category="reranking",
        value_type="float",
        default_value="0.5",
        min_value=0.0,
        max_value=1.0,
        requires_restart=False,
    ),
    # -------------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------------
    SettingDefinition(
        key="synthesis.max_tokens",
        name="Max Tokens",
        description=(
            "Passed as max_tokens to the Claude Sonnet API call for both"
            " streaming and non-streaming synthesis. Controls the maximum"
            " length of the AI answer. ~750 words at 1500 tokens."
            " Increase for complex multi-step answers or comparison tables;"
            " decrease for snappier replies. Directly impacts API cost"
            " (output tokens are billed)."
        ),
        category="synthesis",
        value_type="int",
        default_value="1500",
        min_value=100,
        max_value=8000,
        requires_restart=False,
    ),
    SettingDefinition(
        key="synthesis.context_total_budget",
        name="Context Total Budget",
        description=(
            "Maximum total characters of retrieved text injected into"
            " Claude's prompt across all source groups combined. Once this"
            " budget is exhausted, remaining groups are truncated or"
            " dropped entirely. At 12000 chars (~3000 tokens), the context"
            " fits comfortably in the prompt with room for system"
            " instructions. Increase for thorough answers; decrease to"
            " reduce input token cost and latency."
        ),
        category="synthesis",
        value_type="int",
        default_value="12000",
        min_value=1000,
        max_value=50000,
        requires_restart=False,
    ),
    SettingDefinition(
        key="synthesis.context_group_budget",
        name="Context Group Budget",
        description=(
            "Maximum characters per source group (chunks from the same"
            " page or thread are merged into one group). If a single doc"
            " page has 6000 chars of matching chunks, only the first 4000"
            " are kept. This prevents one verbose source from consuming"
            " the entire Context Total Budget, leaving room for other"
            " sources. Should be ≤ total_budget / 3 for balanced coverage."
        ),
        category="synthesis",
        value_type="int",
        default_value="4000",
        min_value=500,
        max_value=20000,
        requires_restart=False,
    ),
    SettingDefinition(
        key="synthesis.max_chunks",
        name="Max Chunks",
        description=(
            "Hard limit on the number of text chunks selected for context"
            " building, applied before character budgets. Chunks are taken"
            " in reranker-score order, then grouped by source page/thread."
            " With 8 chunks, Claude typically sees 3–5 source groups."
            " Beyond 10, answers rarely improve but cost and latency grow."
            " Interacts with context budgets: whichever limit hits first"
            " wins."
        ),
        category="synthesis",
        value_type="int",
        default_value="8",
        min_value=1,
        max_value=30,
        requires_restart=False,
    ),
    # -------------------------------------------------------------------------
    # Caches (all require restart)
    # -------------------------------------------------------------------------
    SettingDefinition(
        key="caches.query_embedding_size",
        name="Query Embedding Cache Size",
        description=(
            "LRU cache slots for Voyage voyage-context-3 dense embeddings."
            " When a user repeats a query (or another user asks the same"
            " thing), the cached 1024-dim vector is reused instead of"
            " calling the Voyage API — saving ~200ms and one API call."
            " Each entry is ~4KB. At 200 slots = ~800KB memory. Read at"
            " startup; changes need a restart to take effect."
        ),
        category="caches",
        value_type="int",
        default_value="200",
        min_value=10,
        max_value=10000,
        requires_restart=True,
    ),
    SettingDefinition(
        key="caches.sparse_embedding_size",
        name="Sparse Embedding Cache Size",
        description=(
            "LRU cache slots for BM25 sparse vectors (the keyword-matching"
            " half of hybrid search). Computed locally — no API call — but"
            " caching still saves ~50ms of tokenization and TF-IDF math."
            " Keyed by exact query string. Same memory profile as dense"
            " cache. Read at startup; changes need a restart."
        ),
        category="caches",
        value_type="int",
        default_value="200",
        min_value=10,
        max_value=10000,
        requires_restart=True,
    ),
    SettingDefinition(
        key="caches.reranker_size",
        name="Reranker Cache Size",
        description=(
            "LRU cache slots for Voyage rerank-2.5 results. Keyed by"
            " hash(query + document IDs), so a cache hit requires the exact"
            " same query AND the same retrieved chunks. Saves ~300ms and"
            " one Voyage API call per hit. Lower hit rate than embedding"
            " caches because the key is more specific. Read at startup;"
            " changes need a restart."
        ),
        category="caches",
        value_type="int",
        default_value="100",
        min_value=10,
        max_value=5000,
        requires_restart=True,
    ),
    SettingDefinition(
        key="caches.camera_db_size",
        name="Camera DB Cache Size",
        description=(
            "LRU cache slots for camera SQLite query results (list, search,"
            " stats). Caches the full result set for each distinct query."
            " Speeds up /cameras page loads, camera search API, and"
            " comparison lookups. Invalidated when cameras are added,"
            " merged, or edited. Read at startup; changes need a restart."
        ),
        category="caches",
        value_type="int",
        default_value="200",
        min_value=10,
        max_value=5000,
        requires_restart=True,
    ),
    SettingDefinition(
        key="caches.camera_db_ttl",
        name="Camera DB Cache TTL",
        description=(
            "Time-to-live in seconds for camera DB cache entries. After"
            " this duration, the next request triggers a fresh SQLite"
            " query. At 300s (5 min), camera edits take up to 5 minutes"
            " to appear on the public /cameras page. Set lower during"
            " bulk camera imports; higher for stable production. Read at"
            " startup; changes need a restart."
        ),
        category="caches",
        value_type="int",
        default_value="300",
        min_value=10,
        max_value=86400,
        requires_restart=True,
    ),
    # -------------------------------------------------------------------------
    # Prefetch
    # -------------------------------------------------------------------------
    SettingDefinition(
        key="prefetch.multiplier",
        name="Prefetch Multiplier",
        description=(
            "Inside each Qdrant collection, both dense and sparse vectors"
            " independently fetch (limit × this multiplier) candidates"
            " before RRF merges them. With limit=10 and multiplier=3,"
            " each vector type retrieves 30 candidates per collection."
            " Higher values improve RRF fusion quality (more overlap"
            " between dense and sparse sets) but increase Qdrant query"
            " time. Capped by Prefetch Max Limit."
        ),
        category="prefetch",
        value_type="int",
        default_value="3",
        min_value=1,
        max_value=10,
        requires_restart=False,
    ),
    SettingDefinition(
        key="prefetch.max_limit",
        name="Prefetch Max Limit",
        description=(
            "Hard ceiling on prefetch candidates per vector type per"
            " collection: min(limit × multiplier, this cap). Prevents"
            " runaway retrieval when the requested limit is large (e.g."
            " limit=50 × multiplier=3 = 150 would be capped to 50)."
            " Controls peak Qdrant memory per query. 50 is safe for most"
            " deployments; raise only if you see RRF quality drop at"
            " high limits."
        ),
        category="prefetch",
        value_type="int",
        default_value="50",
        min_value=10,
        max_value=500,
        requires_restart=False,
    ),
]


def get_default_setting(key: str) -> SettingDefinition | None:
    """Get a default setting by key.

    Args:
        key: The setting key (e.g., "retrieval.short_query_threshold").

    Returns:
        SettingDefinition or None if not found.
    """
    for setting in DEFAULT_SETTINGS:
        if setting.key == key:
            return setting
    return None


def get_default_settings_by_category(category: str) -> list[SettingDefinition]:
    """Get all default settings in a category.

    Args:
        category: Category name (retrieval, reranking, synthesis, caches, prefetch).

    Returns:
        List of SettingDefinition objects.
    """
    return [s for s in DEFAULT_SETTINGS if s.category == category]


def get_all_setting_keys() -> list[str]:
    """Get all default setting keys.

    Returns:
        List of setting keys.
    """
    return [s.key for s in DEFAULT_SETTINGS]
