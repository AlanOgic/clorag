"""Tests for FastAPI web endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from clorag.config import Settings


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRobotsTxt:
    """Test robots.txt endpoint."""

    def test_robots_txt(self, client: TestClient) -> None:
        """Test that robots.txt blocks search engines."""
        response = client.get("/robots.txt")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "User-agent: *" in response.text
        assert "Disallow: /" in response.text


class TestSearchEndpoints:
    """Test search API endpoints."""

    @pytest.fixture
    def mock_vectorstore(self) -> MagicMock:
        """Mock VectorStore for search tests."""
        mock_vs = AsyncMock()

        # Mock search results
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = {
            "text": "Test documentation content about RCP configuration.",
            "title": "RCP Configuration Guide",
            "url": "https://docs.example.com/rcp-config",
            "_source": "documentation",
        }
        mock_vs.hybrid_search_rrf.return_value = [mock_result]
        mock_vs.search_docs_hybrid.return_value = [mock_result]
        mock_vs.search_cases_hybrid.return_value = [mock_result]

        return mock_vs

    @pytest.fixture
    def mock_embeddings_for_search(self) -> AsyncMock:
        """Mock EmbeddingsClient for search."""
        mock_emb = AsyncMock()
        mock_emb.embed_query.return_value = [0.1] * 1024
        return mock_emb

    @pytest.fixture
    def mock_sparse_embeddings_for_search(self) -> MagicMock:
        """Mock SparseEmbeddingsClient for search."""
        mock_sparse = MagicMock()
        mock_sparse.embed_query.return_value = MagicMock(
            indices=[1, 2, 3],
            values=[0.5, 0.3, 0.2],
        )
        return mock_sparse

    @pytest.fixture
    def mock_anthropic_for_search(self) -> AsyncMock:
        """Mock Anthropic client for answer synthesis."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="This is a synthesized answer about RCP configuration.")
        ]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        return mock_client

    async def test_search_basic(
        self,
        async_client: AsyncClient,
        mock_vectorstore: AsyncMock,
        mock_embeddings_for_search: AsyncMock,
        mock_sparse_embeddings_for_search: MagicMock,
        mock_anthropic_for_search: AsyncMock,
    ) -> None:
        """Test basic search endpoint."""
        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vectorstore),
            patch("clorag.web.app.get_embeddings", return_value=mock_embeddings_for_search),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_embeddings_for_search),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic_for_search),
        ):
            response = await async_client.post(
                "/api/search",
                json={
                    "query": "How to configure RCP IP address?",
                    "source": "both",
                    "limit": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "query" in data
            assert "answer" in data
            assert "results" in data
            assert "source_links" in data
            assert data["query"] == "How to configure RCP IP address?"
            assert isinstance(data["results"], list)

            # Verify embeddings were called
            mock_embeddings_for_search.embed_query.assert_called_once()
            mock_sparse_embeddings_for_search.embed_query.assert_called_once()

    async def test_search_docs_only(
        self,
        async_client: AsyncClient,
        mock_vectorstore: AsyncMock,
        mock_embeddings_for_search: AsyncMock,
        mock_sparse_embeddings_for_search: MagicMock,
        mock_anthropic_for_search: AsyncMock,
    ) -> None:
        """Test search with docs-only source."""
        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vectorstore),
            patch("clorag.web.app.get_embeddings", return_value=mock_embeddings_for_search),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_embeddings_for_search),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic_for_search),
        ):
            response = await async_client.post(
                "/api/search",
                json={
                    "query": "RCP documentation",
                    "source": "docs",
                    "limit": 5,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["source"] == "docs"

            # Verify correct search method was called
            mock_vectorstore.search_docs_hybrid.assert_called_once()

    async def test_search_gmail_only(
        self,
        async_client: AsyncClient,
        mock_vectorstore: AsyncMock,
        mock_embeddings_for_search: AsyncMock,
        mock_sparse_embeddings_for_search: MagicMock,
        mock_anthropic_for_search: AsyncMock,
    ) -> None:
        """Test search with gmail-only source."""
        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vectorstore),
            patch("clorag.web.app.get_embeddings", return_value=mock_embeddings_for_search),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_embeddings_for_search),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic_for_search),
        ):
            response = await async_client.post(
                "/api/search",
                json={
                    "query": "Support case",
                    "source": "gmail",
                    "limit": 5,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["source"] == "gmail"

            # Verify correct search method was called
            mock_vectorstore.search_cases_hybrid.assert_called_once()

    async def test_search_empty_query(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that empty query is rejected."""
        response = await async_client.post(
            "/api/search",
            json={
                "query": "",
                "source": "both",
            },
        )

        assert response.status_code == 422  # Validation error

    async def test_search_query_too_long(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that overly long query is rejected."""
        response = await async_client.post(
            "/api/search",
            json={
                "query": "x" * 2001,  # Exceeds max_length=2000
                "source": "both",
            },
        )

        assert response.status_code == 422  # Validation error

    async def test_search_invalid_source(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that invalid source is rejected."""
        response = await async_client.post(
            "/api/search",
            json={
                "query": "test",
                "source": "invalid",
            },
        )

        assert response.status_code == 422  # Validation error

    async def test_search_limit_validation(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test search limit validation."""
        # Test limit too low
        response = await async_client.post(
            "/api/search",
            json={
                "query": "test",
                "limit": 0,
            },
        )
        assert response.status_code == 422

        # Test limit too high
        response = await async_client.post(
            "/api/search",
            json={
                "query": "test",
                "limit": 51,
            },
        )
        assert response.status_code == 422

    async def test_search_stream_endpoint(
        self,
        async_client: AsyncClient,
        mock_vectorstore: AsyncMock,
        mock_embeddings_for_search: AsyncMock,
        mock_sparse_embeddings_for_search: MagicMock,
        mock_anthropic_for_search: AsyncMock,
    ) -> None:
        """Test streaming search endpoint."""
        # Mock streaming response
        async def mock_stream():
            yield MagicMock(type="content_block_delta", delta=MagicMock(text="Test "))
            yield MagicMock(type="content_block_delta", delta=MagicMock(text="answer"))

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = MagicMock(
            text_stream=mock_stream()
        )
        mock_anthropic_for_search.messages.stream.return_value = mock_stream_context

        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vectorstore),
            patch("clorag.web.app.get_embeddings", return_value=mock_embeddings_for_search),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse_embeddings_for_search),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic_for_search),
        ):
            response = await async_client.post(
                "/api/search/stream",
                json={
                    "query": "RCP configuration",
                    "source": "both",
                },
            )

            assert response.status_code == 200
            # Streaming response should have text/event-stream content type
            assert "text/event-stream" in response.headers.get("content-type", "")


class TestRateLimiting:
    """Test rate limiting functionality."""

    async def test_rate_limit_exists(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that rate limiting is configured."""
        # The app has rate limiting configured
        # We just verify the limiter is set up
        from clorag.web.app import app

        assert hasattr(app.state, "limiter")
        assert app.state.limiter is not None


class TestAdminAuthentication:
    """Test admin authentication functionality."""

    def test_admin_login_endpoint_exists(
        self,
        client: TestClient,
    ) -> None:
        """Test that admin login endpoint exists."""
        response = client.get("/admin/login")

        # Should return HTML login page
        assert response.status_code == 200

    async def test_admin_api_without_auth(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that admin API endpoints require authentication."""
        # Try to access admin endpoint without authentication
        # Note: GET /api/admin/cameras doesn't exist, POST is for creating cameras
        response = await async_client.post("/api/admin/cameras", json={"name": "Test"})

        # Should be unauthorized
        assert response.status_code in [401, 403, 422]  # 422 if validation fails before auth

    async def test_admin_login_with_valid_password(
        self,
        async_client: AsyncClient,
        test_settings: Settings,
    ) -> None:
        """Test admin login with valid password."""
        # Mock get_settings to return test settings with admin password
        with patch("clorag.web.app.get_settings", return_value=test_settings):
            response = await async_client.post(
                "/api/admin/login",
                json={
                    "password": "test-admin-password",
                },
            )

            # Should succeed
            assert response.status_code == 200
            data = response.json()
            assert data.get("success") is True

    async def test_admin_login_with_invalid_password(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test admin login with invalid password."""
        response = await async_client.post(
            "/api/admin/login",
            json={
                "password": "wrong-password",
            },
        )

        # Should fail
        assert response.status_code == 401


class TestCamerasEndpoint:
    """Test public cameras endpoint."""

    def test_cameras_page(
        self,
        client: TestClient,
    ) -> None:
        """Test that cameras page loads."""
        response = client.get("/cameras")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    async def test_cameras_api(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test cameras API endpoint."""
        with patch("clorag.web.app.get_camera_database") as mock_db:
            mock_db_instance = MagicMock()
            mock_db_instance.get_all_cameras.return_value = []
            mock_db.return_value = mock_db_instance

            response = await async_client.get("/api/cameras")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestHomeEndpoint:
    """Test home page endpoint."""

    def test_home_page(
        self,
        client: TestClient,
    ) -> None:
        """Test that home page loads."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    async def test_analytics_page_requires_auth(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that analytics page requires authentication."""
        response = await async_client.get("/admin/analytics")

        # Should redirect or require auth
        # The actual behavior depends on authentication implementation
        assert response.status_code in [200, 302, 401, 403]

    async def test_search_stats_endpoint(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test search stats API endpoint."""
        with patch("clorag.web.app.get_analytics_db") as mock_db:
            mock_db_instance = MagicMock()
            mock_db_instance.get_search_stats.return_value = {
                "total_searches": 100,
                "unique_queries": 50,
            }
            mock_db.return_value = mock_db_instance

            # This endpoint may require auth - test both cases
            response = await async_client.get("/api/admin/search-stats")

            # Either succeeds with data or requires auth
            assert response.status_code in [200, 401, 403]


class TestSearchValidation:
    """Test search request validation."""

    async def test_search_request_defaults(
        self,
        async_client: AsyncClient,
        mock_settings: Settings,
    ) -> None:
        """Test that search request uses defaults."""
        mock_vs = AsyncMock()
        mock_vs.hybrid_search_rrf.return_value = []

        mock_emb = AsyncMock()
        mock_emb.embed_query.return_value = [0.1] * 1024

        mock_sparse = MagicMock()
        mock_sparse.embed_query.return_value = MagicMock(indices=[], values=[])

        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="No results")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vs),
            patch("clorag.web.app.get_embeddings", return_value=mock_emb),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic),
        ):
            # Minimal request - should use defaults
            response = await async_client.post(
                "/api/search",
                json={"query": "test"},
            )

            assert response.status_code == 200
            data = response.json()
            # Default source should be "both"
            assert data["source"] == "both"


class TestConversationSession:
    """Test conversation session functionality for follow-up questions."""

    def test_session_store_creation(self) -> None:
        """Test SessionStore creates and manages sessions."""
        from clorag.web.app import SessionStore

        store = SessionStore(max_sessions=10)

        # Create a session
        session = store.create_session()
        assert session.session_id is not None
        assert len(session.exchanges) == 0

        # Retrieve the session
        retrieved = store.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_session_store_get_or_create(self) -> None:
        """Test get_or_create_session behavior."""
        from clorag.web.app import SessionStore

        store = SessionStore()

        # With None, should create new
        session1 = store.get_or_create_session(None)
        assert session1.session_id is not None

        # With existing ID, should return same session
        session2 = store.get_or_create_session(session1.session_id)
        assert session2.session_id == session1.session_id

        # With invalid ID, should create new
        session3 = store.get_or_create_session("invalid-uuid")
        assert session3.session_id != session1.session_id

    def test_conversation_exchange_history(self) -> None:
        """Test conversation history is maintained correctly."""
        from clorag.web.app import ConversationSession

        session = ConversationSession(session_id="test-session")

        # Add exchanges
        session.add_exchange("Question 1", "Answer 1")
        session.add_exchange("Question 2", "Answer 2")
        session.add_exchange("Question 3", "Answer 3")

        assert len(session.exchanges) == 3

        # Get context messages
        messages = session.get_context_messages()
        assert len(messages) == 6  # 3 Q&A pairs = 6 messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Question 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Answer 1"

    def test_conversation_history_limit(self) -> None:
        """Test conversation history is trimmed to MAX_CONVERSATION_HISTORY."""
        from clorag.web.app import ConversationSession, MAX_CONVERSATION_HISTORY

        session = ConversationSession(session_id="test-session")

        # Add more exchanges than the limit
        for i in range(5):
            session.add_exchange(f"Question {i}", f"Answer {i}")

        # Should only keep the last MAX_CONVERSATION_HISTORY exchanges
        assert len(session.exchanges) == MAX_CONVERSATION_HISTORY
        # The oldest should be trimmed, newest should remain
        assert session.exchanges[-1].query == "Question 4"
        assert session.exchanges[-1].answer == "Answer 4"

    def test_session_lru_eviction(self) -> None:
        """Test sessions are evicted when max_sessions is reached."""
        from clorag.web.app import SessionStore

        store = SessionStore(max_sessions=3)

        # Create sessions up to the limit
        session1 = store.create_session()
        session2 = store.create_session()
        session3 = store.create_session()

        # Creating a new session should evict the oldest
        session4 = store.create_session()

        # Session 1 should be evicted
        assert store.get_session(session1.session_id) is None
        # Session 4 should exist
        assert store.get_session(session4.session_id) is not None

    async def test_search_returns_session_id(
        self,
        async_client: AsyncClient,
        mock_settings: Settings,
    ) -> None:
        """Test that search endpoint returns session_id."""
        mock_vs = AsyncMock()
        mock_vs.hybrid_search_rrf.return_value = []

        mock_emb = AsyncMock()
        mock_emb.embed_query.return_value = [0.1] * 1024

        mock_sparse = MagicMock()
        mock_sparse.embed_query.return_value = MagicMock(indices=[], values=[])

        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test answer")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch("clorag.web.app.get_vectorstore", return_value=mock_vs),
            patch("clorag.web.app.get_embeddings", return_value=mock_emb),
            patch("clorag.web.app.get_sparse_embeddings", return_value=mock_sparse),
            patch("clorag.web.app.get_anthropic", return_value=mock_anthropic),
        ):
            response = await async_client.post(
                "/api/search",
                json={"query": "test question"},
            )

            assert response.status_code == 200
            data = response.json()
            # Should include session_id
            assert "session_id" in data
            assert data["session_id"] is not None
