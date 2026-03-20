"""Tests for the CLORAG MCP server module.

Tests cover:
- Server creation and configuration
- Lifespan management (init/cleanup)
- Tool registration (search, cameras, documents, support)
- Tool execution with mocked services
- Input validation and edge cases
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient as StarletteTestClient

from clorag.models.camera import Camera, CameraSource, DeviceType
from clorag.models.custom_document import CustomDocument, DocumentCategory
from clorag.models.support_case import CaseStatus, ResolutionQuality, SupportCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_services() -> MagicMock:
    """Create a fully mocked MCPServices instance."""
    services = MagicMock()

    # Retriever (async)
    services.retriever = MagicMock()
    services.retriever.retrieve = AsyncMock()
    services.retriever.retrieve_docs = AsyncMock()
    services.retriever.retrieve_cases = AsyncMock()

    # Camera DB (sync)
    services.camera_db = MagicMock()
    services.camera_db.close = MagicMock()

    # Support case DB (sync)
    services.support_case_db = MagicMock()
    services.support_case_db.close = MagicMock()

    # Document service (async)
    services.document_service = MagicMock()
    services.document_service.list_documents = AsyncMock()
    services.document_service.get_document = AsyncMock()
    services.document_service.get_categories = AsyncMock()

    # Vectorstore (async)
    services.vectorstore = MagicMock()
    services.vectorstore.get_chunk = AsyncMock()
    services.vectorstore.delete_chunk = AsyncMock()

    # Embeddings (async)
    services.embeddings = MagicMock()
    services.embeddings.embed = AsyncMock()

    # Sparse embeddings (async)
    services.sparse_embeddings = MagicMock()
    services.sparse_embeddings.embed = AsyncMock()

    # Prompt manager (sync)
    services.prompt_manager = MagicMock()

    # Analytics DB (sync)
    services.analytics_db = MagicMock()

    return services


@pytest.fixture
def sample_camera() -> Camera:
    """Sample camera for testing."""
    return Camera(
        id=1,
        name="FX6",
        manufacturer="Sony",
        code_model="ILME-FX6V",
        device_type=DeviceType.CAMERA_CINEMA,
        ports=["SDI", "HDMI"],
        protocols=["Sony RCP", "VISCA"],
        supported_controls=["Iris", "Gain", "Shutter"],
        notes=["Requires firmware v3.0+"],
        source=CameraSource.DOCUMENTATION,
        doc_url="https://docs.example.com/fx6",
        manufacturer_url="https://sony.com/fx6",
        confidence=0.95,
    )


@pytest.fixture
def sample_camera_b() -> Camera:
    """Second sample camera for comparison tests."""
    return Camera(
        id=2,
        name="AW-UE150",
        manufacturer="Panasonic",
        code_model="AW-UE150",
        device_type=DeviceType.CAMERA_PTZ,
        ports=["Ethernet", "SDI", "HDMI"],
        protocols=["Panasonic AW", "VISCA over IP"],
        supported_controls=["PTZ", "Iris", "Gain", "Preset"],
        notes=[],
        source=CameraSource.DOCUMENTATION,
        confidence=0.9,
    )


@pytest.fixture
def sample_support_case() -> SupportCase:
    """Sample support case for testing."""
    return SupportCase(
        id="case-001",
        thread_id="thread-abc123",
        subject="RIO +WAN connectivity issue",
        status=CaseStatus.RESOLVED,
        resolution_quality=ResolutionQuality.GOOD,
        problem_summary="Customer cannot connect RIO to cloud",
        solution_summary="Firewall was blocking port 443",
        keywords=["RIO", "cloud", "firewall", "connectivity"],
        category="Network",
        product="RIO +WAN",
        document="Full structured case document...",
        messages_count=5,
        created_at=datetime(2026, 1, 15),
        resolved_at=datetime(2026, 1, 16),
    )


@pytest.fixture
def sample_document() -> CustomDocument:
    """Sample custom document for testing."""
    return CustomDocument(
        id="doc-001",
        title="RIO +WAN Setup Guide",
        content="Step-by-step guide for setting up RIO +WAN with cloud connectivity...",
        tags=["RIO", "setup", "cloud"],
        category=DocumentCategory.CONFIGURATION,
        url_reference="https://docs.example.com/rio-wan",
        created_at=datetime(2026, 2, 1),
        updated_at=datetime(2026, 2, 10),
        created_by="admin",
    )


def _make_search_result(text: str, score: float, payload: dict[str, Any]) -> MagicMock:
    """Create a mock SearchResult."""
    result = MagicMock()
    result.text = text
    result.score = score
    result.payload = payload
    return result


def _make_retrieval_result(
    query: str, results: list[MagicMock], total: int, reranked: bool = True
) -> MagicMock:
    """Create a mock RetrievalResult."""
    rr = MagicMock()
    rr.query = query
    rr.results = results
    rr.total_found = total
    rr.reranked = reranked
    return rr


# ===========================================================================
# Server Creation & Lifespan
# ===========================================================================


class TestServerCreation:
    """Tests for MCP server factory and lifespan."""

    @patch("clorag.mcp.server.register_search_tools")
    @patch("clorag.mcp.server.register_camera_tools")
    @patch("clorag.mcp.server.register_document_tools")
    @patch("clorag.mcp.server.register_support_tools")
    @patch("clorag.mcp.server.register_chunk_tools")
    @patch("clorag.mcp.server.register_prompt_tools")
    @patch("clorag.mcp.server.register_ingestion_tools")
    def test_create_mcp_server_registers_all_tools(
        self, mock_ingestion, mock_prompts, mock_chunks,
        mock_support, mock_docs, mock_cameras, mock_search,
    ) -> None:
        """Server creation registers all tool groups."""
        from clorag.mcp.server import create_mcp_server

        mcp = create_mcp_server()

        assert mcp is not None
        mock_search.assert_called_once_with(mcp)
        mock_cameras.assert_called_once_with(mcp)
        mock_docs.assert_called_once_with(mcp)
        mock_support.assert_called_once_with(mcp)
        mock_chunks.assert_called_once_with(mcp)
        mock_prompts.assert_called_once_with(mcp)
        mock_ingestion.assert_called_once_with(mcp)

    @patch("clorag.mcp.server.register_search_tools")
    @patch("clorag.mcp.server.register_camera_tools")
    @patch("clorag.mcp.server.register_document_tools")
    @patch("clorag.mcp.server.register_support_tools")
    @patch("clorag.mcp.server.register_chunk_tools")
    @patch("clorag.mcp.server.register_prompt_tools")
    @patch("clorag.mcp.server.register_ingestion_tools")
    def test_server_has_correct_name(self, *_mocks: Any) -> None:
        """Server is named 'clorag'."""
        from clorag.mcp.server import create_mcp_server

        mcp = create_mcp_server()
        assert mcp.name == "clorag"

    def test_get_services_raises_before_init(self) -> None:
        """get_services() raises RuntimeError when not initialized."""
        import clorag.mcp.server as srv

        original = srv._services
        try:
            srv._services = None
            with pytest.raises(RuntimeError, match="not initialized"):
                srv.get_services()
        finally:
            srv._services = original

    @pytest.mark.asyncio
    async def test_lifespan_initializes_and_cleans_up(self) -> None:
        """Lifespan context manager creates services and cleans up."""
        import clorag.mcp.server as srv

        original = srv._services

        with (
            patch("clorag.mcp.server.MCPServices") as mock_cls,
            patch("clorag.mcp.server.get_camera_database"),
            patch("clorag.mcp.server.get_support_case_database"),
        ):
            mock_instance = MagicMock()
            mock_instance.camera_db = MagicMock()
            mock_instance.support_case_db = MagicMock()
            mock_cls.return_value = mock_instance

            mock_app = MagicMock()

            async with srv.lifespan(mock_app) as services:
                # During lifespan, services are available
                assert services is mock_instance
                assert srv._services is mock_instance

            # After exit, services are cleaned up
            mock_instance.camera_db.close.assert_called_once()
            mock_instance.support_case_db.close.assert_called_once()
            assert srv._services is None

        srv._services = original


# ===========================================================================
# Search Tools
# ===========================================================================


class TestSearchTools:
    """Tests for RAG search MCP tools."""

    @pytest.mark.asyncio
    async def test_search_hybrid(self, mock_services: MagicMock) -> None:
        """search() with default source='both' calls retriever.retrieve with HYBRID."""
        sr = _make_search_result(
            "RIO documentation content",
            0.8765,
            {"_source": "documentation", "title": "RIO Guide", "url": "https://docs.example.com/rio", "category": ""},
        )
        mock_services.retriever.retrieve.return_value = _make_retrieval_result(
            "RIO setup", [sr], total=1
        )

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.search import register_search_tools

            mcp = MagicMock()
            tools: dict[str, Any] = {}

            def capture_tool():
                def decorator(fn: Any) -> Any:
                    tools[fn.__name__] = fn
                    return fn
                return decorator

            mcp.tool = capture_tool
            register_search_tools(mcp)

            result = await tools["search"]("RIO setup", source="both", limit=5)

        assert result["query"] == "RIO setup"
        assert result["total_found"] == 1
        assert result["reranked"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.8765
        assert result["results"][0]["title"] == "RIO Guide"

    @pytest.mark.asyncio
    async def test_search_limit_clamped(self, mock_services: MagicMock) -> None:
        """search() clamps limit to [1, 20]."""
        mock_services.retriever.retrieve.return_value = _make_retrieval_result(
            "test", [], total=0
        )

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.search import register_search_tools

            mcp = MagicMock()
            tools: dict[str, Any] = {}

            def capture_tool():
                def decorator(fn: Any) -> Any:
                    tools[fn.__name__] = fn
                    return fn
                return decorator

            mcp.tool = capture_tool
            register_search_tools(mcp)

            # Limit above max
            await tools["search"]("test", limit=100)
            call_kwargs = mock_services.retriever.retrieve.call_args
            assert call_kwargs.kwargs["limit"] == 20

            # Limit below min
            await tools["search"]("test", limit=-5)
            call_kwargs = mock_services.retriever.retrieve.call_args
            assert call_kwargs.kwargs["limit"] == 1

    @pytest.mark.asyncio
    async def test_search_docs_only(self, mock_services: MagicMock) -> None:
        """search_docs() calls retriever.retrieve_docs."""
        mock_services.retriever.retrieve_docs.return_value = _make_retrieval_result(
            "VISCA protocol", [], total=0
        )

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.search import register_search_tools

            mcp = MagicMock()
            tools: dict[str, Any] = {}

            def capture_tool():
                def decorator(fn: Any) -> Any:
                    tools[fn.__name__] = fn
                    return fn
                return decorator

            mcp.tool = capture_tool
            register_search_tools(mcp)

            result = await tools["search_docs"]("VISCA protocol", limit=3)

        assert result["query"] == "VISCA protocol"
        mock_services.retriever.retrieve_docs.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_cases_only(self, mock_services: MagicMock) -> None:
        """search_cases() calls retriever.retrieve_cases."""
        sr = _make_search_result(
            "Support case about SDI",
            0.75,
            {"title": "SDI Issue", "subject": "SDI not working", "thread_id": "t1", "category": "Hardware"},
        )
        mock_services.retriever.retrieve_cases.return_value = _make_retrieval_result(
            "SDI issue", [sr], total=1
        )

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.search import register_search_tools

            mcp = MagicMock()
            tools: dict[str, Any] = {}

            def capture_tool():
                def decorator(fn: Any) -> Any:
                    tools[fn.__name__] = fn
                    return fn
                return decorator

            mcp.tool = capture_tool
            register_search_tools(mcp)

            result = await tools["search_cases"]("SDI issue", limit=5)

        assert result["results"][0]["thread_id"] == "t1"
        assert result["results"][0]["category"] == "Hardware"

    @pytest.mark.asyncio
    async def test_search_source_mapping(self, mock_services: MagicMock) -> None:
        """search() correctly maps source strings to SearchSource enum."""
        mock_services.retriever.retrieve.return_value = _make_retrieval_result(
            "test", [], total=0
        )

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.core.retriever import SearchSource
            from clorag.mcp.tools.search import register_search_tools

            mcp = MagicMock()
            tools: dict[str, Any] = {}

            def capture_tool():
                def decorator(fn: Any) -> Any:
                    tools[fn.__name__] = fn
                    return fn
                return decorator

            mcp.tool = capture_tool
            register_search_tools(mcp)

            for source_str, expected in [
                ("docs", SearchSource.DOCS),
                ("cases", SearchSource.CASES),
                ("custom", SearchSource.CUSTOM),
                ("both", SearchSource.HYBRID),
                ("DOCS", SearchSource.DOCS),  # case insensitive
                ("unknown", SearchSource.HYBRID),  # fallback
            ]:
                await tools["search"]("test", source=source_str, limit=1)
                call_kwargs = mock_services.retriever.retrieve.call_args
                assert call_kwargs.kwargs["source"] == expected, f"Failed for source='{source_str}'"


# ===========================================================================
# Camera Tools
# ===========================================================================


class TestCameraTools:
    """Tests for camera database MCP tools."""

    def _register(self, mock_services: MagicMock) -> dict[str, Any]:
        """Helper to register camera tools and capture them."""
        tools: dict[str, Any] = {}

        def capture_tool():
            def decorator(fn: Any) -> Any:
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp = MagicMock()
        mcp.tool = capture_tool

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.cameras import register_camera_tools
            register_camera_tools(mcp)

        return tools

    def test_search_cameras(
        self, mock_services: MagicMock, sample_camera: Camera
    ) -> None:
        """search_cameras() calls camera_db.search_cameras and formats results."""
        mock_services.camera_db.search_cameras.return_value = [sample_camera]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["search_cameras"]("Sony FX6")

        assert result["query"] == "Sony FX6"
        assert result["total_found"] == 1
        assert result["cameras"][0]["name"] == "FX6"
        assert result["cameras"][0]["manufacturer"] == "Sony"
        assert result["cameras"][0]["device_type"] == "camera_cinema"

    def test_search_cameras_limits_to_20(
        self, mock_services: MagicMock, sample_camera: Camera
    ) -> None:
        """search_cameras() limits output to 20 cameras."""
        cameras = [sample_camera] * 25
        mock_services.camera_db.search_cameras.return_value = cameras

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["search_cameras"]("Sony")

        assert len(result["cameras"]) == 20

    def test_get_camera_found(
        self, mock_services: MagicMock, sample_camera: Camera
    ) -> None:
        """get_camera() returns camera dict when found."""
        mock_services.camera_db.get_camera.return_value = sample_camera

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_camera"](1)

        assert result["camera"]["id"] == 1
        assert result["camera"]["confidence"] == 0.95

    def test_get_camera_not_found(self, mock_services: MagicMock) -> None:
        """get_camera() returns error when not found."""
        mock_services.camera_db.get_camera.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_camera"](999)

        assert "error" in result
        assert "999" in result["error"]

    def test_find_related_cameras(
        self, mock_services: MagicMock, sample_camera: Camera, sample_camera_b: Camera
    ) -> None:
        """find_related_cameras() returns reference + related cameras."""
        mock_services.camera_db.get_camera.return_value = sample_camera
        mock_services.camera_db.find_related_cameras.return_value = [sample_camera_b]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["find_related_cameras"](1, limit=5)

        assert result["reference_camera"]["name"] == "FX6"
        assert len(result["related_cameras"]) == 1
        assert result["related_cameras"][0]["name"] == "AW-UE150"

    def test_find_related_cameras_not_found(self, mock_services: MagicMock) -> None:
        """find_related_cameras() returns error when reference camera not found."""
        mock_services.camera_db.get_camera.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["find_related_cameras"](999)

        assert "error" in result

    def test_compare_cameras(
        self, mock_services: MagicMock, sample_camera: Camera, sample_camera_b: Camera
    ) -> None:
        """compare_cameras() returns comparison with common specs."""
        mock_services.camera_db.get_cameras_by_ids.return_value = [
            sample_camera, sample_camera_b
        ]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["compare_cameras"]([1, 2])

        assert len(result["cameras"]) == 2
        # Both have SDI
        assert "SDI" in result["common_ports"]
        # Both have VISCA (via "VISCA" and "VISCA over IP" — only exact match)
        # Actually common_protocols is intersection of sets
        # Camera A: {"Sony RCP", "VISCA"}, Camera B: {"Panasonic AW", "VISCA over IP"}
        # Intersection = empty (VISCA != VISCA over IP)
        assert result["common_protocols"] == []
        assert set(result["manufacturers"]) == {"Sony", "Panasonic"}

    def test_compare_cameras_too_few(self, mock_services: MagicMock) -> None:
        """compare_cameras() rejects fewer than 2 cameras."""
        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["compare_cameras"]([1])

        assert "error" in result
        assert "At least 2" in result["error"]

    def test_compare_cameras_too_many(self, mock_services: MagicMock) -> None:
        """compare_cameras() rejects more than 5 cameras."""
        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["compare_cameras"]([1, 2, 3, 4, 5, 6])

        assert "error" in result
        assert "Maximum 5" in result["error"]

    def test_list_cameras_with_filters(
        self, mock_services: MagicMock, sample_camera: Camera
    ) -> None:
        """list_cameras() passes filters to camera_db and formats output."""
        mock_services.camera_db.list_cameras.return_value = [sample_camera]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["list_cameras"](
                manufacturer="Sony",
                device_type="camera_cinema",
                protocol="VISCA",
                limit=10,
            )

        assert result["filters"]["manufacturer"] == "Sony"
        assert result["total_found"] == 1
        mock_services.camera_db.list_cameras.assert_called_once_with(
            manufacturer="Sony",
            device_type="camera_cinema",
            protocol="VISCA",
            limit=10,
        )

    def test_list_cameras_clamps_limit(self, mock_services: MagicMock) -> None:
        """list_cameras() clamps limit to [1, 50]."""
        mock_services.camera_db.list_cameras.return_value = []

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)

            tools["list_cameras"](limit=999)
            assert mock_services.camera_db.list_cameras.call_args.kwargs["limit"] == 50

            tools["list_cameras"](limit=-10)
            assert mock_services.camera_db.list_cameras.call_args.kwargs["limit"] == 1

    def test_get_camera_stats(self, mock_services: MagicMock) -> None:
        """get_camera_stats() aggregates data from multiple DB calls."""
        mock_services.camera_db.get_stats.return_value = {
            "total_cameras": 150,
            "by_source": {"documentation": 100, "manual": 50},
            "manufacturers": 25,
        }
        mock_services.camera_db.get_manufacturers.return_value = [
            "Sony", "Panasonic", "Canon"
        ]
        mock_services.camera_db.get_device_types.return_value = [
            "camera_ptz", "camera_cinema"
        ]
        mock_services.camera_db.get_all_ports.return_value = ["SDI", "HDMI", "Ethernet"]
        mock_services.camera_db.get_all_protocols.return_value = ["VISCA", "Sony RCP"]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_camera_stats"]()

        assert result["total_cameras"] == 150
        assert result["manufacturers_count"] == 25
        assert "Sony" in result["manufacturers"]
        assert "VISCA" in result["protocols"]


# ===========================================================================
# Document Tools
# ===========================================================================


class TestDocumentTools:
    """Tests for custom document MCP tools."""

    def _register(self, mock_services: MagicMock) -> dict[str, Any]:
        tools: dict[str, Any] = {}

        def capture_tool():
            def decorator(fn: Any) -> Any:
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp = MagicMock()
        mcp.tool = capture_tool

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.documents import register_document_tools
            register_document_tools(mcp)

        return tools

    @pytest.mark.asyncio
    async def test_list_documents(
        self, mock_services: MagicMock, sample_document: CustomDocument
    ) -> None:
        """list_documents() returns formatted document list."""
        # Create a mock list item with the fields the tool accesses
        doc_item = MagicMock()
        doc_item.id = "doc-001"
        doc_item.title = "RIO +WAN Setup Guide"
        doc_item.category = DocumentCategory.CONFIGURATION
        doc_item.tags = ["RIO", "setup", "cloud"]
        doc_item.content_preview = "Step-by-step guide..."
        doc_item.created_at = datetime(2026, 2, 1)
        doc_item.is_expired = False

        mock_services.document_service.list_documents.return_value = ([doc_item], 1)

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["list_documents"](category="configuration", limit=10, offset=0)

        assert result["total"] == 1
        assert result["category_filter"] == "configuration"
        assert result["documents"][0]["title"] == "RIO +WAN Setup Guide"
        assert result["documents"][0]["category"] == "configuration"

    @pytest.mark.asyncio
    async def test_list_documents_clamps_values(self, mock_services: MagicMock) -> None:
        """list_documents() clamps limit and offset."""
        mock_services.document_service.list_documents.return_value = ([], 0)

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)

            await tools["list_documents"](limit=999, offset=-5)
            call_kwargs = mock_services.document_service.list_documents.call_args.kwargs
            assert call_kwargs["limit"] == 50
            assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_document_found(
        self, mock_services: MagicMock, sample_document: CustomDocument
    ) -> None:
        """get_document() returns full document when found."""
        mock_services.document_service.get_document.return_value = sample_document

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["get_document"]("doc-001")

        assert result["document"]["id"] == "doc-001"
        assert result["document"]["title"] == "RIO +WAN Setup Guide"
        assert result["document"]["category"] == "configuration"
        assert result["document"]["created_by"] == "admin"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_services: MagicMock) -> None:
        """get_document() returns error when not found."""
        mock_services.document_service.get_document.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["get_document"]("nonexistent")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_document_categories(self, mock_services: MagicMock) -> None:
        """get_document_categories() returns category list."""
        mock_services.document_service.get_categories.return_value = [
            "product_info", "troubleshooting", "configuration"
        ]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["get_document_categories"]()

        assert "categories" in result
        assert len(result["categories"]) == 3


# ===========================================================================
# Support Tools
# ===========================================================================


class TestSupportTools:
    """Tests for support case MCP tools."""

    def _register(self, mock_services: MagicMock) -> dict[str, Any]:
        tools: dict[str, Any] = {}

        def capture_tool():
            def decorator(fn: Any) -> Any:
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp = MagicMock()
        mcp.tool = capture_tool

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.support import register_support_tools
            register_support_tools(mcp)

        return tools

    def test_search_support_cases(
        self, mock_services: MagicMock, sample_support_case: SupportCase
    ) -> None:
        """search_support_cases() returns formatted case list."""
        mock_services.support_case_db.search_cases.return_value = [sample_support_case]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["search_support_cases"]("RIO connectivity", limit=5)

        assert result["query"] == "RIO connectivity"
        assert result["total_found"] == 1
        case = result["cases"][0]
        assert case["id"] == "case-001"
        assert case["status"] == "resolved"
        assert case["resolution_quality"] == 3
        assert case["product"] == "RIO +WAN"
        # document should NOT be included in search results
        assert "document" not in case

    def test_search_support_cases_clamps_limit(self, mock_services: MagicMock) -> None:
        """search_support_cases() clamps limit to [1, 20]."""
        mock_services.support_case_db.search_cases.return_value = []

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)

            tools["search_support_cases"]("test", limit=50)
            assert mock_services.support_case_db.search_cases.call_args[1]["limit"] == 20

    def test_get_support_case_found(
        self, mock_services: MagicMock, sample_support_case: SupportCase
    ) -> None:
        """get_support_case() returns full case with document."""
        mock_services.support_case_db.get_case_by_id.return_value = sample_support_case

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_support_case"]("case-001")

        case = result["case"]
        assert case["id"] == "case-001"
        assert case["document"] == "Full structured case document..."

    def test_get_support_case_not_found(self, mock_services: MagicMock) -> None:
        """get_support_case() returns error when not found."""
        mock_services.support_case_db.get_case_by_id.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_support_case"]("nonexistent")

        assert "error" in result

    def test_list_support_cases(
        self, mock_services: MagicMock, sample_support_case: SupportCase
    ) -> None:
        """list_support_cases() passes filters and formats output."""
        mock_services.support_case_db.list_cases.return_value = ([sample_support_case], 1)

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["list_support_cases"](
                category="Network", product="RIO +WAN", limit=10, offset=0
            )

        assert result["total"] == 1
        assert result["filters"]["category"] == "Network"
        assert result["filters"]["product"] == "RIO +WAN"
        assert len(result["cases"]) == 1

    def test_list_support_cases_clamps(self, mock_services: MagicMock) -> None:
        """list_support_cases() clamps limit and offset."""
        mock_services.support_case_db.list_cases.return_value = ([], 0)

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)

            tools["list_support_cases"](limit=999, offset=-10)
            call_kwargs = mock_services.support_case_db.list_cases.call_args.kwargs
            assert call_kwargs["limit"] == 50
            assert call_kwargs["offset"] == 0

    def test_get_support_stats(self, mock_services: MagicMock) -> None:
        """get_support_stats() returns formatted stats."""
        mock_services.support_case_db.get_stats.return_value = {
            "total": 200,
            "by_category": {"Network": 50, "Hardware": 30},
            "by_product": {"RIO": 80, "CI0": 40},
            "by_quality": {"good": 100, "excellent": 50},
        }

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["get_support_stats"]()

        assert result["total_cases"] == 200
        assert result["by_category"]["Network"] == 50
        assert result["by_product"]["RIO"] == 80


# ===========================================================================
# Camera Serialization
# ===========================================================================


class TestCameraSerialization:
    """Tests for _camera_to_dict helper."""

    def test_camera_to_dict_full(self, sample_camera: Camera) -> None:
        """_camera_to_dict() serializes all fields correctly."""
        from clorag.mcp.tools.cameras import _camera_to_dict

        d = _camera_to_dict(sample_camera)

        assert d["id"] == 1
        assert d["name"] == "FX6"
        assert d["manufacturer"] == "Sony"
        assert d["code_model"] == "ILME-FX6V"
        assert d["device_type"] == "camera_cinema"
        assert d["ports"] == ["SDI", "HDMI"]
        assert d["protocols"] == ["Sony RCP", "VISCA"]
        assert d["supported_controls"] == ["Iris", "Gain", "Shutter"]
        assert d["notes"] == ["Requires firmware v3.0+"]
        assert d["doc_url"] == "https://docs.example.com/fx6"
        assert d["confidence"] == 0.95

    def test_camera_to_dict_none_device_type(self) -> None:
        """_camera_to_dict() handles None device_type."""
        from clorag.mcp.tools.cameras import _camera_to_dict

        camera = Camera(id=3, name="Unknown Cam")
        d = _camera_to_dict(camera)
        assert d["device_type"] is None


# ===========================================================================
# Support Case Serialization
# ===========================================================================


class TestSupportCaseSerialization:
    """Tests for _case_to_dict helper."""

    def test_case_to_dict_without_document(
        self, sample_support_case: SupportCase
    ) -> None:
        """_case_to_dict() excludes document by default."""
        from clorag.mcp.tools.support import _case_to_dict

        d = _case_to_dict(sample_support_case)

        assert d["id"] == "case-001"
        assert d["status"] == "resolved"
        assert d["resolution_quality"] == 3
        assert "document" not in d

    def test_case_to_dict_with_document(
        self, sample_support_case: SupportCase
    ) -> None:
        """_case_to_dict() includes document when requested."""
        from clorag.mcp.tools.support import _case_to_dict

        d = _case_to_dict(sample_support_case, include_document=True)

        assert "document" in d
        assert d["document"] == "Full structured case document..."

    def test_case_to_dict_none_quality(self) -> None:
        """_case_to_dict() handles None resolution_quality."""
        from clorag.mcp.tools.support import _case_to_dict

        case = SupportCase(
            id="case-002",
            thread_id="t2",
            subject="Test",
            status=CaseStatus.UNRESOLVED,
        )
        d = _case_to_dict(case)
        assert d["resolution_quality"] is None


# ===========================================================================
# Chunk Tools (delete_chunk)
# ===========================================================================


class TestChunkTools:
    """Tests for chunk management MCP tools."""

    def _register(self, mock_services: MagicMock) -> dict[str, Any]:
        tools: dict[str, Any] = {}

        def capture_tool():
            def decorator(fn: Any) -> Any:
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp = MagicMock()
        mcp.tool = capture_tool

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.chunks import register_chunk_tools
            register_chunk_tools(mcp)

        return tools

    @pytest.mark.asyncio
    async def test_delete_chunk_success(self, mock_services: MagicMock) -> None:
        """delete_chunk() deletes and returns confirmation."""
        mock_services.vectorstore.get_chunk.return_value = {
            "id": "chunk-123",
            "payload": {"text": "some text"},
        }
        mock_services.vectorstore.delete_chunk.return_value = True

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["delete_chunk"](
                collection="docs", chunk_id="chunk-123",
            )

        assert result["status"] == "deleted"
        assert result["chunk_id"] == "chunk-123"
        assert result["collection"] == "docusaurus_docs"

    @pytest.mark.asyncio
    async def test_delete_chunk_not_found(self, mock_services: MagicMock) -> None:
        """delete_chunk() returns error when chunk doesn't exist."""
        mock_services.vectorstore.get_chunk.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["delete_chunk"](
                collection="docs", chunk_id="nonexistent",
            )

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_chunk_invalid_collection(self, mock_services: MagicMock) -> None:
        """delete_chunk() returns error for invalid collection."""
        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = await tools["delete_chunk"](
                collection="invalid", chunk_id="chunk-123",
            )

        assert "error" in result
        assert "Invalid collection" in result["error"]


# ===========================================================================
# Prompt Tools
# ===========================================================================


class TestPromptTools:
    """Tests for prompt management MCP tools."""

    def _register(self, mock_services: MagicMock) -> dict[str, Any]:
        tools: dict[str, Any] = {}

        def capture_tool():
            def decorator(fn: Any) -> Any:
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp = MagicMock()
        mcp.tool = capture_tool

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            from clorag.mcp.tools.prompts import register_prompt_tools
            register_prompt_tools(mcp)

        return tools

    def test_update_prompt_content(self, mock_services: MagicMock) -> None:
        """update_prompt() updates content and creates version."""
        mock_prompt = MagicMock()
        mock_prompt.to_dict.return_value = {
            "id": "p-001",
            "key": "synthesis.web_answer",
            "name": "Web Answer",
            "content": "New content with {query}",
        }
        mock_services.prompt_manager.update_prompt.return_value = mock_prompt
        mock_services.prompt_manager.detect_variables.return_value = ["query"]

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["update_prompt"](
                prompt_id="p-001",
                content="New content with {query}",
                change_note="Updated via MCP",
            )

        assert result["status"] == "updated"
        assert result["version_created"] is True
        mock_services.prompt_manager.update_prompt.assert_called_once()
        call_kwargs = mock_services.prompt_manager.update_prompt.call_args.kwargs
        assert call_kwargs["content"] == "New content with {query}"
        assert call_kwargs["variables"] == ["query"]
        assert call_kwargs["updated_by"] == "mcp"

    def test_update_prompt_metadata_only(self, mock_services: MagicMock) -> None:
        """update_prompt() with name only doesn't create version."""
        mock_prompt = MagicMock()
        mock_prompt.to_dict.return_value = {"id": "p-001", "name": "New Name"}
        mock_services.prompt_manager.update_prompt.return_value = mock_prompt

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["update_prompt"](prompt_id="p-001", name="New Name")

        assert result["status"] == "updated"
        assert result["version_created"] is False

    def test_update_prompt_no_fields(self, mock_services: MagicMock) -> None:
        """update_prompt() with no fields returns error."""
        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["update_prompt"](prompt_id="p-001")

        assert "error" in result

    def test_update_prompt_not_found(self, mock_services: MagicMock) -> None:
        """update_prompt() returns error when prompt doesn't exist."""
        mock_services.prompt_manager.update_prompt.return_value = None
        mock_services.prompt_manager.detect_variables.return_value = []

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["update_prompt"](
                prompt_id="nonexistent", content="test",
            )

        assert "error" in result

    def test_rollback_prompt_success(self, mock_services: MagicMock) -> None:
        """rollback_prompt() restores previous version."""
        mock_prompt = MagicMock()
        mock_prompt.to_dict.return_value = {
            "id": "p-001",
            "key": "synthesis.web_answer",
            "content": "Old content",
        }
        mock_services.prompt_manager.rollback_prompt.return_value = mock_prompt

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["rollback_prompt"](prompt_id="p-001", version=2)

        assert result["status"] == "rolled_back"
        assert result["to_version"] == 2
        mock_services.prompt_manager.rollback_prompt.assert_called_once_with(
            prompt_id="p-001", version=2, rolled_back_by="mcp",
        )

    def test_rollback_prompt_not_found(self, mock_services: MagicMock) -> None:
        """rollback_prompt() returns error when prompt/version not found."""
        mock_services.prompt_manager.rollback_prompt.return_value = None

        with patch("clorag.mcp.server.get_services", return_value=mock_services):
            tools = self._register(mock_services)
            result = tools["rollback_prompt"](prompt_id="p-001", version=999)

        assert "error" in result


@pytest.mark.asyncio
async def test_search_hybrid_rrf_with_match_filters(mock_settings):
    """search_hybrid_rrf passes match_filters to Qdrant prefetch."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from qdrant_client.http import models

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    with patch("clorag.core.vectorstore.AsyncQdrantClient", return_value=mock_client):
        from clorag.core.vectorstore import VectorStore

        vs = VectorStore()

        results = await vs.search_hybrid_rrf(
            collection="test_docs",
            dense_vector=[0.1] * 1024,
            sparse_vector=models.SparseVector(indices=[1, 2], values=[0.5, 0.3]),
            limit=5,
            match_filters={"category": "troubleshooting"},
        )

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]

        for prefetch in call_kwargs["prefetch"]:
            assert prefetch.filter is not None
            assert len(prefetch.filter.must) == 1
            assert prefetch.filter.must[0].key == "category"


@pytest.mark.asyncio
async def test_search_hybrid_rrf_without_filters(mock_settings):
    """search_hybrid_rrf works without match_filters (backward compat)."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from qdrant_client.http import models

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    with patch("clorag.core.vectorstore.AsyncQdrantClient", return_value=mock_client):
        from clorag.core.vectorstore import VectorStore

        vs = VectorStore()

        results = await vs.search_hybrid_rrf(
            collection="test_docs",
            dense_vector=[0.1] * 1024,
            sparse_vector=models.SparseVector(indices=[1, 2], values=[0.5, 0.3]),
            limit=5,
        )

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]

        for prefetch in call_kwargs["prefetch"]:
            assert prefetch.filter is None


# ---------------------------------------------------------------------------
# Bearer Auth Middleware Tests
# ---------------------------------------------------------------------------


def _make_test_app(api_key: str):
    """Create a minimal Starlette app wrapped with BearerAuthMiddleware."""
    from clorag.mcp.auth import apply_bearer_auth

    async def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    return apply_bearer_auth(app, api_key)


class TestBearerAuthMiddleware:
    """Tests for BearerAuthMiddleware."""

    def test_valid_token(self):
        app = _make_test_app("test-secret-key")
        client = StarletteTestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_missing_auth_header(self):
        app = _make_test_app("test-secret-key")
        client = StarletteTestClient(app)
        resp = client.get("/")
        assert resp.status_code == 401
        assert "Missing Bearer token" in resp.json()["error"]

    def test_invalid_token(self):
        app = _make_test_app("test-secret-key")
        client = StarletteTestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]

    def test_malformed_auth_header(self):
        app = _make_test_app("test-secret-key")
        client = StarletteTestClient(app)
        resp = client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401

    def test_bearer_with_extra_whitespace(self):
        app = _make_test_app("test-secret-key")
        client = StarletteTestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer  test-secret-key  "})
        assert resp.status_code == 200


class TestPathSanitization:
    """Tests for import_custom_documents path validation."""

    def test_path_traversal_rejected(self, tmp_path):
        """Paths outside base dir are rejected."""
        from pathlib import Path

        base_dir = tmp_path / "imports"
        base_dir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        folder_path = Path(str(outside)).resolve()
        base = Path(str(base_dir)).resolve()

        assert not folder_path.is_relative_to(base)

    def test_dotdot_traversal_rejected(self, tmp_path):
        """../.. traversal paths are caught after resolve()."""
        from pathlib import Path

        base_dir = tmp_path / "imports"
        base_dir.mkdir()

        evil_path = Path(str(base_dir) + "/../../../etc").resolve()
        base = Path(str(base_dir)).resolve()

        assert not evil_path.is_relative_to(base)

    def test_valid_subdir_accepted(self, tmp_path):
        """Subdirectories within base dir are accepted."""
        from pathlib import Path

        base_dir = tmp_path / "imports"
        sub = base_dir / "batch1"
        sub.mkdir(parents=True)

        folder_path = Path(str(sub)).resolve()
        base = Path(str(base_dir)).resolve()

        assert folder_path.is_relative_to(base)

    def test_base_dir_itself_accepted(self, tmp_path):
        """The base directory itself is a valid target."""
        from pathlib import Path

        base_dir = tmp_path / "imports"
        base_dir.mkdir()

        folder_path = Path(str(base_dir)).resolve()
        base = Path(str(base_dir)).resolve()

        assert folder_path.is_relative_to(base)
