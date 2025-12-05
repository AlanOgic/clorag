"""Shared pytest fixtures for CLORAG tests."""

import os
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from clorag.config import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with mocked values.

    This fixture provides a Settings instance with all required fields
    populated with test values, eliminating the need for actual API keys.
    """
    return Settings(
        anthropic_api_key=SecretStr("test-anthropic-key"),
        voyage_api_key=SecretStr("test-voyage-key"),
        qdrant_url="http://localhost:6333",
        qdrant_api_key=SecretStr("test-qdrant-key"),
        qdrant_docs_collection="test_docs",
        qdrant_cases_collection="test_cases",
        voyage_model="voyage-context-3",
        voyage_dimensions=1024,
        docusaurus_url="https://test.example.com",
        gmail_label="test-label",
        google_credentials_path=Path("test_credentials.json"),
        google_token_path=Path("test_token.json"),
        claude_model="claude-sonnet-4-20250514",
        haiku_model="claude-haiku-4-5-20251001",
        sonnet_model="claude-sonnet-4-5-20250929",
        max_turns=50,
        database_path="test_clorag.db",
        analytics_database_path="test_analytics.db",
        admin_password=SecretStr("test-admin-password"),
        searxng_url="https://search.example.com",
    )


@pytest.fixture
def mock_settings(test_settings: Settings) -> Generator[Settings, None, None]:
    """Fixture to mock get_settings() when needed.

    Use this fixture explicitly in tests that need mocked settings
    but don't need to test environment variable loading.
    """
    with patch("clorag.config.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def mock_voyage_client() -> Generator[MagicMock, None, None]:
    """Mock Voyage AI client for embeddings tests."""
    mock_client = MagicMock()

    # Mock embed() response
    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_embed_response.total_tokens = 100
    mock_client.embed.return_value = mock_embed_response

    # Mock contextualized_embed() response
    mock_contextualized_response = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [[0.3] * 1024]
    mock_contextualized_response.results = [mock_result]
    mock_client.contextualized_embed.return_value = mock_contextualized_response

    with patch("voyageai.Client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_qdrant_client() -> Generator[AsyncMock, None, None]:
    """Mock Qdrant async client for vector store tests."""
    mock_client = AsyncMock()

    # Mock collection operations
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = MagicMock(vectors_count=100)

    # Mock search results
    mock_search_result = MagicMock()
    mock_search_result.id = "test-id-1"
    mock_search_result.score = 0.95
    mock_search_result.payload = {
        "text": "Test document content",
        "url": "https://test.example.com/doc",
    }
    mock_client.search.return_value = [mock_search_result]
    mock_client.query_points.return_value = MagicMock(points=[mock_search_result])

    with patch("clorag.core.vectorstore.AsyncQdrantClient", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_anthropic_client() -> Generator[AsyncMock, None, None]:
    """Mock Anthropic async client for Claude API tests."""
    mock_client = AsyncMock()

    # Mock streaming response
    async def mock_stream():
        """Async generator for streaming responses."""
        yield MagicMock(
            type="content_block_start",
            content_block=MagicMock(type="text", text=""),
        )
        yield MagicMock(
            type="content_block_delta",
            delta=MagicMock(type="text_delta", text="Test response"),
        )
        yield MagicMock(
            type="content_block_stop",
            index=0,
        )
        yield MagicMock(
            type="message_stop",
        )

    mock_stream_response = AsyncMock()
    mock_stream_response.__aenter__.return_value = mock_stream()
    mock_client.messages.stream.return_value = mock_stream_response

    # Mock non-streaming response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_fastembed() -> Generator[MagicMock, None, None]:
    """Mock FastEmbed SparseTextEmbedding for BM25 tests."""
    mock_model = MagicMock()

    # Mock embedding result
    mock_embedding = MagicMock()
    mock_embedding.indices = [1, 2, 3]
    mock_embedding.values = [0.5, 0.3, 0.2]
    mock_model.embed.return_value = iter([mock_embedding])

    with patch("fastembed.SparseTextEmbedding", return_value=mock_model):
        yield mock_model


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create TestClient for FastAPI sync tests."""
    # Import here to avoid circular imports
    from clorag.web.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create AsyncClient for FastAPI async tests.

    Uses ASGI transport for proper async handling and lifespan events.
    """
    # Import here to avoid circular imports
    from clorag.web.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def temp_database(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary database file for testing."""
    db_path = tmp_path / "test.db"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_embedding_vector() -> list[float]:
    """Sample 1024-dimensional embedding vector for tests."""
    return [0.1] * 1024


@pytest.fixture
def sample_sparse_vector() -> dict[str, list[int | float]]:
    """Sample sparse vector (BM25) for tests."""
    return {
        "indices": [1, 5, 10, 25, 100],
        "values": [0.8, 0.6, 0.5, 0.3, 0.2],
    }


@pytest.fixture
def cleanup_settings_cache() -> Generator[None, None, None]:
    """Clean up the settings cache before and after tests.

    The get_settings() function uses @lru_cache, which can cause
    stale settings to persist across tests. This fixture clears
    the cache to ensure test isolation.
    """
    from clorag.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
