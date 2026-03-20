"""Tests for RAG performance optimizations.

Updated to match modular architecture:
- Parallel embeddings are now handled by asyncio.gather in search pipeline
- Cache recommendations moved to web/routers/admin/analytics.py
- _perform_search removed; search pipeline is in web/search/pipeline.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSparseModelPreloading:
    """Test that sparse embedding model is pre-loaded at startup."""

    @pytest.mark.asyncio
    async def test_lifespan_preloads_sparse_model(self) -> None:
        """Test that lifespan() pre-loads the sparse embedding model."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.ensure_collections = AsyncMock()

        mock_sparse_client = MagicMock()
        mock_sparse_client._model_name = "Qdrant/bm25"

        with patch("clorag.web.app.get_vectorstore", return_value=mock_vectorstore), patch(
            "clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_client
        ) as mock_get_sparse, patch(
            "clorag.web.app.get_settings"
        ) as mock_settings:
            # Mock settings to disable draft polling
            mock_settings.return_value.draft_polling_enabled = False

            from clorag.web.app import lifespan

            mock_app = MagicMock()

            async with lifespan(mock_app):
                pass

            # Verify sparse embeddings client was retrieved (which loads the model)
            mock_get_sparse.assert_called_once()


class TestCacheStatisticsEndpoint:
    """Test the cache statistics helper (now in analytics router)."""

    def test_generate_cache_recommendations_healthy(self) -> None:
        """Test recommendations for healthy cache performance."""
        from clorag.web.routers.admin.analytics import _generate_cache_recommendations

        dense_stats = {"size": 50, "hits": 80, "misses": 20, "hit_rate_percent": 80.0}
        sparse_stats = {"size": 45, "hits": 75, "misses": 25, "hit_rate_percent": 75.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        assert len(recommendations) == 1
        assert "healthy" in recommendations[0].lower()

    def test_generate_cache_recommendations_low_hit_rate(self) -> None:
        """Test recommendations for low cache hit rate."""
        from clorag.web.routers.admin.analytics import _generate_cache_recommendations

        dense_stats = {"size": 50, "hits": 10, "misses": 90, "hit_rate_percent": 10.0}
        sparse_stats = {"size": 45, "hits": 15, "misses": 85, "hit_rate_percent": 15.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        assert len(recommendations) >= 2
        assert any("dense" in r.lower() for r in recommendations)
        assert any("sparse" in r.lower() for r in recommendations)

    def test_generate_cache_recommendations_near_capacity(self) -> None:
        """Test recommendations when caches are near capacity."""
        from clorag.web.routers.admin.analytics import _generate_cache_recommendations

        dense_stats = {"size": 195, "hits": 80, "misses": 20, "hit_rate_percent": 80.0}
        sparse_stats = {"size": 192, "hits": 75, "misses": 25, "hit_rate_percent": 75.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        assert any("capacity" in r.lower() for r in recommendations)
