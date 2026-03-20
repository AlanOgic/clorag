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
        description="Score threshold for queries with ≤2 words",
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
        description="Score threshold for queries with 3-5 words",
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
        description="Score threshold for queries with >5 words",
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
        description="Score bonus added when query contains technical terms",
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
        description="Maximum allowed threshold value after bonuses",
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
        description="Minimum results returned even below threshold",
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
        description="Multiplier for over-fetching before reranking (e.g. 3 = fetch 3x limit)",
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
        description="Number of top results to return after reranking",
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
        description="Minimum score ratio (vs top result) for source diversity injection",
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
        description="Maximum tokens for Claude synthesis response",
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
        description="Maximum total characters across all context groups",
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
        description="Maximum characters per source group in context",
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
        description="Maximum chunks included in synthesis context",
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
        description="Max entries in query embedding LRU cache",
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
        description="Max entries in sparse (BM25) embedding LRU cache",
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
        description="Max entries in reranker result LRU cache",
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
        description="Max entries in camera database LRU cache",
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
        description="Time-to-live in seconds for camera database cache",
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
        description="Multiplier for Qdrant prefetch candidate count (limit × multiplier)",
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
        description="Maximum prefetch limit cap to avoid excessive retrieval",
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
