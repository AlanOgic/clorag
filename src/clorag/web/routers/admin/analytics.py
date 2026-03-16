"""Admin analytics and performance metrics endpoints.

Provides search analytics, cache statistics, and performance monitoring.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from clorag.core.metrics import get_metrics_collector
from clorag.web.auth import verify_admin
from clorag.web.dependencies import get_analytics_db
from clorag.web.search import get_sparse_embeddings

router = APIRouter()


@router.get("/search-stats", tags=["Analytics"])
async def api_search_stats(
    days: int = 30,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get search analytics statistics."""
    analytics = get_analytics_db()
    return {
        "stats": analytics.get_search_stats(days=days),
        "popular_queries": analytics.get_popular_queries(limit=10, days=days),
        "recent_searches": analytics.get_recent_searches(limit=20),
    }


@router.get("/search-stats/popular", tags=["Analytics"])
async def api_popular_queries(
    limit: int = 10,
    days: int = 30,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """Get popular search queries."""
    analytics = get_analytics_db()
    return analytics.get_popular_queries(limit=limit, days=days)


@router.get("/cache-stats", tags=["Performance"])
async def api_cache_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get embedding and rerank cache statistics for performance monitoring.

    Returns hit/miss rates for dense (Voyage AI), sparse (BM25), and rerank
    caches. Higher hit rates indicate better cache efficiency.
    """
    from clorag.core.embeddings import get_query_cache
    from clorag.core.reranker import get_rerank_cache

    sparse_emb = get_sparse_embeddings()
    dense_cache = get_query_cache()
    rerank_cache = get_rerank_cache()

    dense_stats = dense_cache.stats()
    sparse_stats = sparse_emb.cache_stats()
    rerank_stats = rerank_cache.stats()

    return {
        "dense_cache": dense_stats,
        "sparse_cache": sparse_stats,
        "rerank_cache": rerank_stats,
        "recommendations": _generate_cache_recommendations(
            dense_stats, sparse_stats, rerank_stats
        ),
    }


@router.get("/metrics", tags=["Performance"])
async def api_performance_metrics(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get comprehensive performance metrics for the RAG pipeline.

    Returns timing statistics for embedding generation, vector search,
    and total search latency with percentiles (p50, p90, p95, p99).

    Metrics are collected from a sliding window of the last 1000 operations.
    """
    metrics = get_metrics_collector()
    all_stats = metrics.get_all_stats()

    # Add target thresholds for comparison
    thresholds = {
        "embedding_generation": {"target_ms": 200, "warning_ms": 500},
        "vector_search": {"target_ms": 100, "warning_ms": 300},
        "total_search": {"target_ms": 500, "warning_ms": 1000},
        "llm_synthesis": {"target_ms": 2000, "warning_ms": 5000},
    }

    # Generate performance alerts
    alerts = []
    for metric_name, stats in all_stats.get("metrics", {}).items():
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            if stats.get("p95_ms", 0) > threshold["warning_ms"]:
                alerts.append(
                    {
                        "level": "warning",
                        "metric": metric_name,
                        "message": (
                            f"{metric_name} p95 ({stats['p95_ms']}ms) exceeds "
                            f"warning threshold ({threshold['warning_ms']}ms)"
                        ),
                    }
                )
            elif stats.get("p95_ms", 0) > threshold["target_ms"]:
                alerts.append(
                    {
                        "level": "info",
                        "metric": metric_name,
                        "message": (
                            f"{metric_name} p95 ({stats['p95_ms']}ms) "
                            f"exceeds target ({threshold['target_ms']}ms)"
                        ),
                    }
                )

    return {
        **all_stats,
        "thresholds": thresholds,
        "alerts": alerts,
    }


@router.get("/metrics/recent/{metric_name}", tags=["Performance"])
async def api_recent_metrics(
    metric_name: str,
    count: int = Query(default=10, ge=1, le=100),
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get recent measurements for a specific metric.

    Useful for debugging recent performance issues or viewing trends.
    """
    metrics = get_metrics_collector()
    recent = metrics.get_recent(metric_name, count)

    if not recent:
        raise HTTPException(
            status_code=404, detail=f"No data for metric: {metric_name}"
        )

    return {
        "metric": metric_name,
        "count": len(recent),
        "measurements": recent,
    }


@router.get("/search/{search_id}", tags=["Analytics"])
async def api_get_search(
    search_id: int,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a stored search by ID with full response and chunks data."""
    analytics = get_analytics_db()
    search = analytics.get_search_by_id(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")
    return search


@router.get("/search-quality", tags=["Analytics"])
async def api_search_quality(
    limit: int = 50,
    days: int = 30,
    max_avg_score: float = 0.3,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get low-quality searches for review.

    Returns searches with low average relevance scores, useful for
    identifying queries that return poor results and tuning thresholds.
    """
    analytics = get_analytics_db()
    low_quality = analytics.get_low_quality_searches(
        limit=limit, days=days, max_avg_score=max_avg_score
    )
    return {
        "count": len(low_quality),
        "max_avg_score_threshold": max_avg_score,
        "days": days,
        "searches": low_quality,
    }


@router.get("/conversations", tags=["Analytics"])
async def api_get_conversations(
    limit: int = 20,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """Get recent conversations grouped by session_id."""
    analytics = get_analytics_db()
    return analytics.get_recent_conversations(limit=limit)


def _generate_cache_recommendations(
    dense_stats: dict[str, int | float],
    sparse_stats: dict[str, int | float],
    rerank_stats: dict[str, int | float] | None = None,
) -> list[str]:
    """Generate actionable recommendations based on cache performance."""
    recommendations = []

    # Check dense cache hit rate
    if dense_stats.get("hit_rate_percent", 0) < 30:
        recommendations.append(
            "Dense cache hit rate is low (<30%). Consider increasing cache size "
            "or pre-warming with common queries."
        )

    # Check sparse cache hit rate
    if sparse_stats.get("hit_rate_percent", 0) < 30:
        recommendations.append(
            "Sparse cache hit rate is low (<30%). Users may be asking diverse queries."
        )

    # Check rerank cache hit rate
    if rerank_stats and rerank_stats.get("hit_rate_percent", 0) < 20:
        recommendations.append(
            "Rerank cache hit rate is low (<20%). This is normal for diverse queries."
        )

    # Check if caches are full
    if dense_stats.get("size", 0) >= 190:  # Near 200 limit
        recommendations.append(
            "Dense cache is near capacity. Consider increasing QUERY_CACHE_MAX_SIZE."
        )

    if sparse_stats.get("size", 0) >= 190:
        recommendations.append(
            "Sparse cache is near capacity. Consider increasing SPARSE_CACHE_MAX_SIZE."
        )

    if rerank_stats and rerank_stats.get("size", 0) >= 95:  # Near 100 limit
        recommendations.append(
            "Rerank cache is near capacity. Consider increasing RERANK_CACHE_MAX_SIZE."
        )

    if not recommendations:
        recommendations.append("Cache performance is healthy.")

    return recommendations
