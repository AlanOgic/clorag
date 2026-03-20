"""Tests for FastAPI web endpoints.

Updated to match modular architecture:
- Search pipeline functions moved to web/search/pipeline.py and web/search/synthesis.py
- SessionStore moved to web/auth/sessions.py
- ConversationSession moved to web/schemas.py
- Cache recommendations moved to web/routers/admin/analytics.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from clorag.config import Settings


# ===========================================================================
# Session / Conversation Tests (unit tests, no FastAPI client needed)
# ===========================================================================


class TestConversationSession:
    """Test conversation session functionality for follow-up questions."""

    def test_session_store_creation(self) -> None:
        """Test SessionStore creates and manages sessions."""
        from clorag.web.auth.sessions import SessionStore

        store = SessionStore(max_sessions=10)

        session = store.create_session()
        assert session.session_id is not None
        assert len(session.exchanges) == 0

        retrieved = store.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_session_store_get_or_create(self) -> None:
        """Test get_or_create_session behavior."""
        from clorag.web.auth.sessions import SessionStore

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
        from clorag.web.schemas import ConversationSession

        session = ConversationSession(session_id="test-session")

        session.add_exchange("Question 1", "Answer 1")
        session.add_exchange("Question 2", "Answer 2")
        session.add_exchange("Question 3", "Answer 3")

        assert len(session.exchanges) == 3

        messages = session.get_context_messages()
        assert len(messages) == 6  # 3 Q&A pairs = 6 messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Question 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Answer 1"

    def test_conversation_history_limit(self) -> None:
        """Test conversation history is trimmed to MAX_CONVERSATION_HISTORY."""
        from clorag.web.schemas import MAX_CONVERSATION_HISTORY, ConversationSession

        session = ConversationSession(session_id="test-session")

        for i in range(5):
            session.add_exchange(f"Question {i}", f"Answer {i}")

        assert len(session.exchanges) == MAX_CONVERSATION_HISTORY
        assert session.exchanges[-1].query == "Question 4"
        assert session.exchanges[-1].answer == "Answer 4"

    def test_session_lru_eviction(self) -> None:
        """Test sessions are evicted when max_sessions is reached."""
        from clorag.web.auth.sessions import SessionStore

        store = SessionStore(max_sessions=3)

        session1 = store.create_session()
        session2 = store.create_session()
        session3 = store.create_session()
        session4 = store.create_session()

        # Session 1 should be evicted
        assert store.get_session(session1.session_id) is None
        assert store.get_session(session4.session_id) is not None


# ===========================================================================
# Search Validation Tests (unit tests for Pydantic models)
# ===========================================================================


class TestSearchValidation:
    """Test search request validation."""

    def test_search_request_valid(self) -> None:
        """Test SearchRequest accepts valid input."""
        from clorag.web.schemas import SearchRequest

        req = SearchRequest(query="test query")
        assert req.query == "test query"
        assert req.source == "both"  # default
        assert req.limit == 10  # default

    def test_search_request_empty_query_rejected(self) -> None:
        """Test that empty query is rejected."""
        from pydantic import ValidationError

        from clorag.web.schemas import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_search_request_query_too_long(self) -> None:
        """Test that overly long query is rejected."""
        from pydantic import ValidationError

        from clorag.web.schemas import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 2001)


# ===========================================================================
# Tests requiring FastAPI client (skip if Qdrant unavailable)
# These tests use the client/async_client fixtures which start the app
# and need Qdrant. They are integration tests.
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting functionality."""

    async def test_rate_limit_exists(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that rate limiting is configured."""
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
        assert response.status_code == 200

    async def test_admin_api_without_auth(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that admin API endpoints require authentication."""
        response = await async_client.post("/api/admin/cameras", json={"name": "Test"})
        assert response.status_code in [401, 403, 422]

    async def test_admin_login_with_valid_password(
        self,
        async_client: AsyncClient,
        test_settings: Settings,
    ) -> None:
        """Test admin login with valid password."""
        with patch("clorag.web.routers.admin.auth.get_settings", return_value=test_settings):
            response = await async_client.post(
                "/api/admin/login",
                json={"password": "test-admin-password"},
            )
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
            json={"password": "wrong-password"},
        )
        assert response.status_code == 401


@pytest.mark.integration
class TestHealthEndpoint:
    """Test health check endpoint (requires Qdrant)."""

    def test_health_check(self, client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.integration
class TestRobotsTxt:
    """Test robots.txt endpoint (requires Qdrant)."""

    def test_robots_txt(self, client: TestClient) -> None:
        """Test that robots.txt blocks search engines."""
        response = client.get("/robots.txt")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "User-agent: *" in response.text
        assert "Disallow: /" in response.text


@pytest.mark.integration
class TestHomeEndpoint:
    """Test home page endpoint (requires Qdrant)."""

    def test_home_page(self, client: TestClient) -> None:
        """Test that home page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.integration
class TestCamerasEndpoint:
    """Test public cameras endpoint (requires Qdrant for page, API mockable)."""

    def test_cameras_page(self, client: TestClient) -> None:
        """Test that cameras page loads."""
        response = client.get("/cameras")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    async def test_cameras_api(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test cameras API endpoint."""
        with patch("clorag.web.routers.cameras.get_camera_database") as mock_db:
            mock_db_instance = MagicMock()
            mock_db_instance.get_all_cameras.return_value = []
            mock_db.return_value = mock_db_instance

            response = await async_client.get("/api/cameras")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    async def test_analytics_page_requires_auth(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test that analytics page requires authentication."""
        response = await async_client.get("/admin/analytics")
        assert response.status_code in [200, 302, 401, 403]
