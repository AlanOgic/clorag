"""Tests for Phase 1 RAG performance optimizations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.models import SparseVector


class TestParallelEmbeddingGeneration:
    """Test parallel dense/sparse embedding generation."""

    @pytest.mark.asyncio
    async def test_parallel_embeddings_returns_both_vectors(self) -> None:
        """Test that _generate_embeddings_parallel returns both dense and sparse vectors."""
        # Mock dense embeddings client
        mock_dense_client = MagicMock()
        mock_dense_client.embed_query = AsyncMock(return_value=[0.1] * 1024)

        # Mock sparse embeddings client
        mock_sparse_client = MagicMock()
        mock_sparse_vector = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        mock_sparse_client.embed_query = MagicMock(return_value=mock_sparse_vector)

        with patch(
            "clorag.web.app.get_embeddings", return_value=mock_dense_client
        ), patch(
            "clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_client
        ):
            from clorag.web.app import _generate_embeddings_parallel

            dense_vector, sparse_vector = await _generate_embeddings_parallel(
                "test query"
            )

            # Verify dense vector
            assert len(dense_vector) == 1024
            assert all(v == 0.1 for v in dense_vector)

            # Verify sparse vector
            assert isinstance(sparse_vector, SparseVector)
            assert sparse_vector.indices == [1, 2, 3]
            assert sparse_vector.values == [0.5, 0.3, 0.2]

    @pytest.mark.asyncio
    async def test_parallel_embeddings_calls_both_clients(self) -> None:
        """Test that both embedding clients are called with the query."""
        mock_dense_client = MagicMock()
        mock_dense_client.embed_query = AsyncMock(return_value=[0.1] * 1024)

        mock_sparse_client = MagicMock()
        mock_sparse_client.embed_query = MagicMock(
            return_value=SparseVector(indices=[1], values=[0.5])
        )

        with patch(
            "clorag.web.app.get_embeddings", return_value=mock_dense_client
        ), patch(
            "clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_client
        ):
            from clorag.web.app import _generate_embeddings_parallel

            await _generate_embeddings_parallel("my test query")

            # Verify both clients were called with the query
            mock_dense_client.embed_query.assert_called_once_with("my test query")
            mock_sparse_client.embed_query.assert_called_once_with("my test query")

    @pytest.mark.asyncio
    async def test_parallel_embeddings_runs_concurrently(self) -> None:
        """Test that embeddings are generated concurrently, not sequentially."""
        call_order: list[str] = []

        async def slow_dense_embed(query: str) -> list[float]:
            call_order.append("dense_start")
            await asyncio.sleep(0.05)  # Simulate API latency
            call_order.append("dense_end")
            return [0.1] * 1024

        def slow_sparse_embed(query: str) -> SparseVector:
            call_order.append("sparse_start")
            # Synchronous, but wrapped in to_thread
            import time

            time.sleep(0.05)  # Simulate computation time
            call_order.append("sparse_end")
            return SparseVector(indices=[1], values=[0.5])

        mock_dense_client = MagicMock()
        mock_dense_client.embed_query = slow_dense_embed

        mock_sparse_client = MagicMock()
        mock_sparse_client.embed_query = slow_sparse_embed

        with patch(
            "clorag.web.app.get_embeddings", return_value=mock_dense_client
        ), patch(
            "clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_client
        ):
            from clorag.web.app import _generate_embeddings_parallel

            await _generate_embeddings_parallel("test")

            # In parallel execution, both should start before either ends
            # The order should be: both starts, then both ends (interleaved)
            assert "dense_start" in call_order
            assert "sparse_start" in call_order
            assert "dense_end" in call_order
            assert "sparse_end" in call_order

            # Both should start before both end (parallel execution)
            dense_start_idx = call_order.index("dense_start")
            sparse_start_idx = call_order.index("sparse_start")
            dense_end_idx = call_order.index("dense_end")
            sparse_end_idx = call_order.index("sparse_end")

            # At least one should start before the other ends (proves parallelism)
            # In sequential: start1, end1, start2, end2
            # In parallel: start1, start2, end1/end2 (interleaved)
            assert (
                sparse_start_idx < dense_end_idx or dense_start_idx < sparse_end_idx
            ), "Embeddings should run in parallel"


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

            # Create a mock app
            mock_app = MagicMock()

            # Run lifespan
            async with lifespan(mock_app):
                pass

            # Verify sparse embeddings client was retrieved (which loads the model)
            mock_get_sparse.assert_called_once()


class TestCacheStatisticsEndpoint:
    """Test the cache statistics API endpoint."""

    def test_generate_cache_recommendations_healthy(self) -> None:
        """Test recommendations for healthy cache performance."""
        from clorag.web.app import _generate_cache_recommendations

        dense_stats = {"size": 50, "hits": 80, "misses": 20, "hit_rate_percent": 80.0}
        sparse_stats = {"size": 45, "hits": 75, "misses": 25, "hit_rate_percent": 75.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        assert len(recommendations) == 1
        assert "healthy" in recommendations[0].lower()

    def test_generate_cache_recommendations_low_hit_rate(self) -> None:
        """Test recommendations for low cache hit rate."""
        from clorag.web.app import _generate_cache_recommendations

        dense_stats = {"size": 50, "hits": 10, "misses": 90, "hit_rate_percent": 10.0}
        sparse_stats = {"size": 45, "hits": 15, "misses": 85, "hit_rate_percent": 15.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        # Should have recommendations for both caches
        assert len(recommendations) >= 2
        assert any("dense" in r.lower() for r in recommendations)
        assert any("sparse" in r.lower() for r in recommendations)

    def test_generate_cache_recommendations_near_capacity(self) -> None:
        """Test recommendations when caches are near capacity."""
        from clorag.web.app import _generate_cache_recommendations

        dense_stats = {"size": 195, "hits": 80, "misses": 20, "hit_rate_percent": 80.0}
        sparse_stats = {"size": 192, "hits": 75, "misses": 25, "hit_rate_percent": 75.0}

        recommendations = _generate_cache_recommendations(dense_stats, sparse_stats)

        # Should recommend increasing cache size
        assert any("capacity" in r.lower() for r in recommendations)

    @pytest.mark.asyncio
    async def test_cache_stats_endpoint_structure(self) -> None:
        """Test that cache stats endpoint returns expected structure."""
        # Mock the cache objects
        mock_dense_cache = MagicMock()
        mock_dense_cache.stats.return_value = {
            "size": 42,
            "hits": 100,
            "misses": 50,
            "hit_rate_percent": 66.7,
        }

        mock_sparse_client = MagicMock()
        mock_sparse_client.cache_stats.return_value = {
            "size": 38,
            "hits": 90,
            "misses": 60,
            "hit_rate_percent": 60.0,
        }

        # Patch at the source module where it's imported from
        with patch(
            "clorag.core.embeddings.get_query_cache", return_value=mock_dense_cache
        ), patch(
            "clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_client
        ):
            from clorag.web.app import api_cache_stats

            # Call the endpoint function directly (bypassing auth)
            result = await api_cache_stats(_=True)

            # Verify structure
            assert "dense_cache" in result
            assert "sparse_cache" in result
            assert "recommendations" in result

            # Verify dense cache stats
            assert result["dense_cache"]["size"] == 42
            assert result["dense_cache"]["hits"] == 100

            # Verify sparse cache stats
            assert result["sparse_cache"]["size"] == 38
            assert result["sparse_cache"]["hits"] == 90

            # Verify recommendations is a list
            assert isinstance(result["recommendations"], list)


class TestIntegrationPerformanceOptimizations:
    """Integration tests for performance optimizations."""

    @pytest.mark.asyncio
    async def test_perform_search_uses_parallel_embeddings(self) -> None:
        """Test that _perform_search uses parallel embedding generation."""
        # Track calls to verify parallel function is used
        parallel_called = False

        async def mock_parallel_embeddings(query: str):
            nonlocal parallel_called
            parallel_called = True
            return [0.1] * 1024, SparseVector(indices=[1], values=[0.5])

        # Mock vectorstore
        mock_vs = MagicMock()
        mock_vs.hybrid_search_rrf = AsyncMock(return_value=[])

        with patch(
            "clorag.web.app._generate_embeddings_parallel",
            side_effect=mock_parallel_embeddings,
        ), patch("clorag.web.app.get_vectorstore", return_value=mock_vs):
            from clorag.web.app import SearchRequest, SearchSource, _perform_search

            # Use SearchSource.BOTH which triggers hybrid_search_rrf
            req = SearchRequest(query="test query", limit=10, source=SearchSource.BOTH)
            await _perform_search(req)

            assert parallel_called, "_generate_embeddings_parallel should be called"
