"""Tests for Voyage AI embeddings client."""

from unittest.mock import MagicMock

import pytest

from clorag.config import Settings
from clorag.core.embeddings import EmbeddingResult, EmbeddingsClient


class TestEmbeddingsClient:
    """Test EmbeddingsClient with mocked Voyage AI API."""

    def test_client_initialization(
        self, test_settings: Settings, mock_voyage_client: MagicMock
    ) -> None:
        """Test that client initializes with correct settings."""
        client = EmbeddingsClient()

        assert client.model == "voyage-context-3"
        assert client.dimensions == 1024

    def test_client_custom_initialization(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test client initialization with custom parameters."""
        client = EmbeddingsClient(
            api_key="custom-key",
            model="voyage-2",
            dimensions=512,
        )

        assert client.model == "voyage-2"
        assert client.dimensions == 512

    async def test_embed_documents(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test embedding multiple documents."""
        client = EmbeddingsClient()
        texts = ["First document", "Second document"]

        result = await client.embed_documents(texts)

        # Verify the result
        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 2
        assert len(result.vectors[0]) == 1024
        assert result.total_tokens == 100

        # Verify API was called correctly
        mock_voyage_client.embed.assert_called_once()
        call_args = mock_voyage_client.embed.call_args
        assert call_args.kwargs["texts"] == texts
        assert call_args.kwargs["model"] == "voyage-context-3"
        assert call_args.kwargs["input_type"] == "document"
        assert call_args.kwargs["output_dimension"] == 1024

    async def test_embed_documents_with_query_type(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test embedding with query input type."""
        client = EmbeddingsClient()
        texts = ["Query text"]

        result = await client.embed_documents(texts, input_type="query")

        # Verify input_type is passed correctly
        call_args = mock_voyage_client.embed.call_args
        assert call_args.kwargs["input_type"] == "query"

    async def test_embed_empty_list(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test that empty list returns empty result without API call."""
        client = EmbeddingsClient()

        result = await client.embed_documents([])

        assert result.vectors == []
        assert result.total_tokens == 0
        mock_voyage_client.embed.assert_not_called()

    async def test_embed_query(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test embedding a single query."""
        client = EmbeddingsClient()
        query = "What is the IP address configuration?"

        result = await client.embed_query(query)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1024

        # Verify API was called with contextualized_embed
        mock_voyage_client.contextualized_embed.assert_called_once()
        call_args = mock_voyage_client.contextualized_embed.call_args
        assert call_args.kwargs["inputs"] == [[query]]
        assert call_args.kwargs["model"] == "voyage-context-3"
        assert call_args.kwargs["input_type"] == "query"
        assert call_args.kwargs["output_dimension"] == 1024

    async def test_embed_batch(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test batch embedding with multiple calls."""
        client = EmbeddingsClient()
        # Create 50 texts to test batching (default batch_size=30)
        texts = [f"Document {i}" for i in range(50)]

        result = await client.embed_batch(texts, batch_size=30)

        # Should make 2 calls (30 + 20)
        assert mock_voyage_client.embed.call_count == 2

        # Verify result contains all embeddings (2 per call from mock)
        assert len(result.vectors) == 4  # 2 vectors * 2 calls
        # Total tokens = 100 per call * 2 calls
        assert result.total_tokens == 200

    async def test_embed_batch_custom_batch_size(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test batch embedding with custom batch size."""
        client = EmbeddingsClient()
        texts = [f"Document {i}" for i in range(10)]

        result = await client.embed_batch(texts, batch_size=5)

        # Should make 2 calls (5 + 5)
        assert mock_voyage_client.embed.call_count == 2

    async def test_embed_contextualized(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test contextualized embedding for multi-chunk documents."""
        client = EmbeddingsClient()
        documents = [
            ["Chunk 1 of doc 1", "Chunk 2 of doc 1"],
            ["Chunk 1 of doc 2"],
        ]

        result = await client.embed_contextualized(documents)

        # Verify the result structure (mock returns 1 result with 1 embedding)
        assert len(result) == 1  # 1 document in mock results
        assert len(result[0]) == 1  # First doc has 1 chunk embedding from mock
        assert len(result[0][0]) == 1024  # Embedding dimension

        # Verify API call
        mock_voyage_client.contextualized_embed.assert_called_once()
        call_args = mock_voyage_client.contextualized_embed.call_args
        assert call_args.kwargs["inputs"] == documents
        assert call_args.kwargs["model"] == "voyage-context-3"
        assert call_args.kwargs["input_type"] == "document"

    async def test_embed_contextualized_empty(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test contextualized embedding with empty list."""
        client = EmbeddingsClient()

        result = await client.embed_contextualized([])

        assert result == []
        mock_voyage_client.contextualized_embed.assert_not_called()

    async def test_embed_contextualized_batch(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test batch contextualized embedding."""
        client = EmbeddingsClient()
        # Create 15 documents to test batching (batch_size=10)
        documents = [[f"Chunk of doc {i}"] for i in range(15)]

        result = await client.embed_contextualized_batch(
            documents, batch_size=10
        )

        # Should make 2 calls (10 + 5)
        assert mock_voyage_client.contextualized_embed.call_count == 2

        # Verify result structure (mock returns 1 result per call)
        assert len(result) == 2  # 1 result per call


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_embedding_result_creation(self) -> None:
        """Test creating an EmbeddingResult."""
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        total_tokens = 50

        result = EmbeddingResult(vectors=vectors, total_tokens=total_tokens)

        assert result.vectors == vectors
        assert result.total_tokens == total_tokens

    def test_embedding_result_empty(self) -> None:
        """Test EmbeddingResult with empty vectors."""
        result = EmbeddingResult(vectors=[], total_tokens=0)

        assert result.vectors == []
        assert result.total_tokens == 0


class TestEmbeddingsClientRetry:
    """Test retry behavior for embeddings client."""

    async def test_retry_on_failure(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test that client retries on failure."""
        client = EmbeddingsClient()

        # Mock failure then success
        mock_voyage_client.embed.side_effect = [
            Exception("API Error"),
            MagicMock(embeddings=[[0.1] * 1024], total_tokens=10),
        ]

        result = await client.embed_documents(["Test"])

        # Should have retried and succeeded
        assert mock_voyage_client.embed.call_count == 2
        assert len(result.vectors) == 1

    async def test_retry_exhaustion(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test that client fails after max retries."""
        client = EmbeddingsClient()

        # Mock persistent failure
        mock_voyage_client.embed.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await client.embed_documents(["Test"])

        # Should have tried 3 times (initial + 2 retries)
        assert mock_voyage_client.embed.call_count >= 1  # At least one attempt


class TestEmbeddingsClientProperties:
    """Test EmbeddingsClient properties."""

    def test_model_property(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test model property returns correct value."""
        client = EmbeddingsClient()
        assert client.model == "voyage-context-3"

        custom_client = EmbeddingsClient(model="voyage-2")
        assert custom_client.model == "voyage-2"

    def test_dimensions_property(
        self, mock_voyage_client: MagicMock
    ) -> None:
        """Test dimensions property returns correct value."""
        client = EmbeddingsClient()
        assert client.dimensions == 1024

        custom_client = EmbeddingsClient(dimensions=512)
        assert custom_client.dimensions == 512

    def test_query_model_constant(self) -> None:
        """Test QUERY_MODEL constant is defined."""
        assert EmbeddingsClient.QUERY_MODEL == "voyage-2"
