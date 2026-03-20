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
            "Minimum relevance score for short queries (1–2 words like"
            " 'RIO' or 'tally'). Lower = more results, higher = stricter."
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
            "Minimum relevance score for medium queries (3–5 words like"
            " 'RIO firmware update'). Higher because more words give"
            " better signal."
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
            "Minimum relevance score for detailed queries (6+ words)."
            " Strictest threshold — long queries should match precisely."
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
            "Extra score added to threshold when the query contains"
            " technical terms (VISCA, SDI, HDMI, NDI…). Makes filtering"
            " stricter for specific technical questions."
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
            "Upper limit for the threshold after all bonuses. Prevents"
            " over-filtering that would return too few results."
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
            "Always return at least this many results, even if scores"
            " fall below the threshold. Safety net so users never get"
            " an empty answer."
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
            "How many extra results to fetch before reranking trims them."
            " 3× means fetch 30 to pick the best 10. Higher = better"
            " quality but slower."
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
            "How many results to keep after the reranker scores them."
            " These are the chunks sent to Claude for answer synthesis."
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
            "Ensures results include docs, cases, and custom docs — not"
            " just one source. A result qualifies if its score is at"
            " least this fraction of the top result (0.5 = 50%)."
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
            "Maximum length of Claude's answer (~750 words at 1500"
            " tokens). Increase for complex answers, decrease for"
            " snappier replies."
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
            "Total character budget for all context sent to Claude."
            " Shared across docs, cases, and custom docs. Larger = more"
            " context but higher cost and latency."
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
            "Character budget per source type (docs, cases, custom)."
            " Prevents one source from dominating the context window."
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
            "Maximum text chunks passed to Claude. Each chunk is a"
            " section of a document or support case. Diminishing returns"
            " past 8–10."
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
            "Recent query embeddings kept in memory. Cache hits skip the"
            " Voyage API call, saving ~200ms per search. Increase for"
            " high traffic."
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
            "Recent BM25 sparse vectors kept in memory. Used alongside"
            " dense embeddings for hybrid search. Cache hits save ~50ms"
            " per search."
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
            "Recent reranker results kept in memory. Cache hits skip"
            " the Voyage rerank API, saving ~300ms. Same query + same"
            " docs = cache hit."
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
            "Camera database query results kept in memory. Speeds up"
            " the /cameras page and camera search API."
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
            "How long camera data stays cached before refresh (seconds)."
            " 300s = 5 min. Lower if camera data changes frequently."
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
            "Candidates Qdrant fetches per vector type before RRF merge."
            " 3× with limit 10 = 30 candidates per collection. Higher ="
            " better fusion quality."
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
            "Hard cap on prefetch candidates regardless of multiplier."
            " Prevents excessive memory usage. 50 is safe for most"
            " deployments."
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
