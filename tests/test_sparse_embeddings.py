"""Tests for BM25 sparse embeddings client."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.http.models import SparseVector

from clorag.core.sparse_embeddings import SparseEmbeddingsClient


class TestSparseEmbeddingsClient:
    """Test SparseEmbeddingsClient with mocked FastEmbed."""

    def test_client_initialization(self) -> None:
        """Test that client initializes with default model."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding:
            mock_model = MagicMock()
            mock_embedding.return_value = mock_model

            client = SparseEmbeddingsClient()

            # Verify FastEmbed was initialized with correct model
            mock_embedding.assert_called_once_with(model_name="Qdrant/bm25")
            assert client._model_name == "Qdrant/bm25"

    def test_client_custom_model(self) -> None:
        """Test client initialization with custom model."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding:
            mock_model = MagicMock()
            mock_embedding.return_value = mock_model

            client = SparseEmbeddingsClient(model_name="custom-bm25")

            mock_embedding.assert_called_once_with(model_name="custom-bm25")
            assert client._model_name == "custom-bm25"

    def test_embed_texts(self) -> None:
        """Test embedding multiple texts."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock returns embeddings with numpy-like arrays
            mock_emb1 = MagicMock()
            mock_emb1.indices = MagicMock()
            mock_emb1.indices.tolist.return_value = [1, 2, 3]
            mock_emb1.values = MagicMock()
            mock_emb1.values.tolist.return_value = [0.5, 0.3, 0.2]

            mock_emb2 = MagicMock()
            mock_emb2.indices = MagicMock()
            mock_emb2.indices.tolist.return_value = [4, 5, 6]
            mock_emb2.values = MagicMock()
            mock_emb2.values.tolist.return_value = [0.6, 0.4, 0.1]

            mock_model.embed.return_value = iter([mock_emb1, mock_emb2])

            client = SparseEmbeddingsClient()
            texts = ["First text", "Second text"]
            result = client.embed_texts(texts)

            # Verify the result
            assert len(result) == 2
            assert isinstance(result[0], SparseVector)
            assert result[0].indices == [1, 2, 3]
            assert result[0].values == [0.5, 0.3, 0.2]
            assert result[1].indices == [4, 5, 6]

            # Verify API was called
            mock_model.embed.assert_called_once_with(texts)

    def test_embed_empty_list(self) -> None:
        """Test that empty list returns empty result."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            client = SparseEmbeddingsClient()
            result = client.embed_texts([])

            assert result == []
            mock_model.embed.assert_not_called()

    def test_embed_query(self) -> None:
        """Test embedding a single query."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock returns single embedding with numpy-like arrays
            mock_embedding = MagicMock()
            mock_embedding.indices = MagicMock()
            mock_embedding.indices.tolist.return_value = [5, 10, 15]
            mock_embedding.values = MagicMock()
            mock_embedding.values.tolist.return_value = [0.8, 0.6, 0.4]
            mock_model.embed.return_value = iter([mock_embedding])

            client = SparseEmbeddingsClient()
            query = "What is the IP configuration?"
            result = client.embed_query(query)

            # Verify the result
            assert isinstance(result, SparseVector)
            assert result.indices == [5, 10, 15]
            assert result.values == [0.8, 0.6, 0.4]

            # Verify embed_texts was called with list containing query
            mock_model.embed.assert_called_once_with([query])

    def test_embed_query_empty_result(self) -> None:
        """Test embed_query when result is empty."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock returns empty iterator
            mock_model.embed.return_value = iter([])

            client = SparseEmbeddingsClient()
            result = client.embed_query("test")

            # Should return empty sparse vector
            assert isinstance(result, SparseVector)
            assert result.indices == []
            assert result.values == []

    def test_embed_batch(self) -> None:
        """Test batch embedding."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock embeddings for batches with numpy-like arrays
            def create_mock_emb(i):
                mock_emb = MagicMock()
                mock_emb.indices = MagicMock()
                mock_emb.indices.tolist.return_value = [i, i + 1, i + 2]
                mock_emb.values = MagicMock()
                mock_emb.values.tolist.return_value = [0.5, 0.3, 0.2]
                return mock_emb

            # Return 5 embeddings per call
            call_count = [0]

            def embed_side_effect(texts):
                start = call_count[0]
                count = len(texts)
                call_count[0] += count
                return iter([create_mock_emb(start + i) for i in range(count)])

            mock_model.embed.side_effect = embed_side_effect

            client = SparseEmbeddingsClient()
            texts = [f"Text {i}" for i in range(10)]

            result = client.embed_batch(texts, batch_size=5)

            # Should have processed all texts
            assert len(result) == 10

            # Verify all are SparseVectors
            for vec in result:
                assert isinstance(vec, SparseVector)

    def test_embed_batch_empty(self) -> None:
        """Test batch embedding with empty list."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            client = SparseEmbeddingsClient()
            result = client.embed_batch([])

            assert result == []
            mock_model.embed.assert_not_called()


class TestSparseVectorConversion:
    """Test conversion from FastEmbed embeddings to Qdrant SparseVectors."""

    def test_sparse_vector_structure(self) -> None:
        """Test that SparseVector has correct structure."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock embedding with specific indices and values (numpy-like)
            mock_embedding = MagicMock()
            mock_embedding.indices = MagicMock()
            mock_embedding.indices.tolist.return_value = [10, 20, 30]
            mock_embedding.values = MagicMock()
            mock_embedding.values.tolist.return_value = [0.9, 0.5, 0.1]
            mock_model.embed.return_value = iter([mock_embedding])

            client = SparseEmbeddingsClient()
            result = client.embed_texts(["test"])

            sparse_vec = result[0]
            assert hasattr(sparse_vec, "indices")
            assert hasattr(sparse_vec, "values")
            assert sparse_vec.indices == [10, 20, 30]
            assert sparse_vec.values == [0.9, 0.5, 0.1]

    def test_numpy_to_list_conversion(self) -> None:
        """Test that numpy arrays are converted to Python lists."""
        with patch("clorag.core.sparse_embeddings.SparseTextEmbedding") as mock_embedding_cls:
            mock_model = MagicMock()
            mock_embedding_cls.return_value = mock_model

            # Mock embedding with numpy arrays (what FastEmbed actually returns)
            mock_embedding = MagicMock()
            mock_embedding.indices = np.array([1, 5, 10], dtype=np.int32)
            mock_embedding.values = np.array([0.8, 0.6, 0.4], dtype=np.float32)
            mock_model.embed.return_value = iter([mock_embedding])

            client = SparseEmbeddingsClient()
            result = client.embed_texts(["test"])

            # Should be converted to lists
            sparse_vec = result[0]
            assert isinstance(sparse_vec.indices, list)
            assert isinstance(sparse_vec.values, list)
            assert sparse_vec.indices == [1, 5, 10]
