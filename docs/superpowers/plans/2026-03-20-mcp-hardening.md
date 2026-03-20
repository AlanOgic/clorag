# MCP Server Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the CLORAG MCP server with HTTP auth, path sanitization, asyncio fixes, public API usage, and enriched responses.

**Architecture:** 6 surgical changes to existing files + 1 new file (`mcp/auth.py`). Changes are ordered by dependency: asyncio fix → filter param → search refactor → (auth, path, responses in parallel). All backward-compatible.

**Tech Stack:** Python 3.10+, FastMCP, Starlette ASGI, pydantic-settings, asyncio, Qdrant client

**Spec:** `docs/superpowers/specs/2026-03-20-mcp-hardening-design.md`

---

### Task 1: Replace deprecated `asyncio.get_event_loop()` in chunks.py

**Files:**
- Modify: `src/clorag/mcp/tools/chunks.py:244-259` (edit_chunk)
- Modify: `src/clorag/mcp/tools/chunks.py:382-395` (search_chunks)

- [ ] **Step 1: Fix `edit_chunk` — replace `get_event_loop().run_in_executor()`**

In `src/clorag/mcp/tools/chunks.py`, find the `edit_chunk` function (~line 244). Replace:

```python
            import asyncio

            metadata_updates["text"] = text

            dense_task = services.embeddings.embed_documents(
                [text],
            )
            sparse_task = asyncio.get_event_loop().run_in_executor(
                None, services.sparse_embeddings.embed_texts, [text],
            )

            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task,
            )
```

With:

```python
            import asyncio

            metadata_updates["text"] = text

            dense_task = services.embeddings.embed_documents(
                [text],
            )
            sparse_task = asyncio.to_thread(
                services.sparse_embeddings.embed_texts, [text],
            )

            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task,
            )
```

- [ ] **Step 2: Fix `search_chunks` — replace `get_event_loop().run_in_executor()`**

In the same file, find `search_chunks` (~line 382). Replace:

```python
        import asyncio

        from qdrant_client import models as qmodels

        from clorag.core.vectorstore import SearchResult

        dense_task = services.embeddings.embed_query(query)
        sparse_task = asyncio.get_event_loop().run_in_executor(
            None, services.sparse_embeddings.embed_query, query,
        )

        dense_vector, sparse_vector = await asyncio.gather(
            dense_task, sparse_task,
        )
```

With:

```python
        import asyncio

        from qdrant_client import models as qmodels

        from clorag.core.vectorstore import SearchResult

        dense_task = services.embeddings.embed_query(query)
        sparse_task = asyncio.to_thread(
            services.sparse_embeddings.embed_query, query,
        )

        dense_vector, sparse_vector = await asyncio.gather(
            dense_task, sparse_task,
        )
```

- [ ] **Step 3: Run linter to verify no regressions**

Run: `uv run ruff check src/clorag/mcp/tools/chunks.py`
Expected: No errors (or only pre-existing ones).

- [ ] **Step 4: Commit**

```bash
git add src/clorag/mcp/tools/chunks.py
git commit -m "fix: replace deprecated asyncio.get_event_loop() with to_thread() in MCP chunks"
```

---

### Task 2: Add `match_filters` parameter to `search_hybrid_rrf()`

**Files:**
- Modify: `src/clorag/core/vectorstore.py:412-470` (search_hybrid_rrf method)
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp.py`:

```python
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

        # Verify query_points was called
        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]

        # Both prefetch items should have filter set
        for prefetch in call_kwargs["prefetch"]:
            assert prefetch.filter is not None
            assert len(prefetch.filter.must) == 1
            assert prefetch.filter.must[0].key == "category"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp.py::test_search_hybrid_rrf_with_match_filters -v`
Expected: FAIL — `search_hybrid_rrf()` does not accept `match_filters` parameter.

- [ ] **Step 3: Write the implementation**

In `src/clorag/core/vectorstore.py`, find `search_hybrid_rrf` (~line 412). Change the signature:

```python
    async def search_hybrid_rrf(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
        match_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
```

Add filter construction before the `query_points` call (after `prefetch_limit = ...`):

```python
        # Build Qdrant filter from exact-match dict
        query_filter = None
        if match_filters:
            must_conditions = [
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in match_filters.items()
            ]
            query_filter = models.Filter(must=must_conditions)
```

Then pass `filter=query_filter` to both Prefetch blocks:

```python
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp.py::test_search_hybrid_rrf_with_match_filters -v`
Expected: PASS

- [ ] **Step 5: Also write a test for the None case (backward compat)**

Add to `tests/test_mcp.py`:

```python
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

        # Prefetch items should have no filter
        for prefetch in call_kwargs["prefetch"]:
            assert prefetch.filter is None
```

- [ ] **Step 6: Run both tests**

Run: `uv run pytest tests/test_mcp.py::test_search_hybrid_rrf_with_match_filters tests/test_mcp.py::test_search_hybrid_rrf_without_filters -v`
Expected: Both PASS

- [ ] **Step 7: Run type checker**

Run: `uv run mypy src/clorag/core/vectorstore.py --strict 2>&1 | head -20`
Expected: No new errors from the change.

**Note:** The tests use `mock_settings` which patches `clorag.config.get_settings`. Inside `search_hybrid_rrf`, calls to `clorag.services.settings_manager.get_setting()` will hit the `try/except` fallback (lines 436-441 in vectorstore.py) and use hardcoded defaults. This is acceptable — the fallback exists precisely for this case.

- [ ] **Step 8: Commit**

```bash
git add src/clorag/core/vectorstore.py tests/test_mcp.py
git commit -m "feat: add match_filters param to search_hybrid_rrf() for filtered hybrid search"
```

---

### Task 3: Refactor `search_chunks` to use public VectorStore API

**Files:**
- Modify: `src/clorag/mcp/tools/chunks.py:380-467` (search_chunks function)

- [ ] **Step 1: Replace the manual Qdrant query block in `search_chunks`**

In `src/clorag/mcp/tools/chunks.py`, find `search_chunks` (~line 380). Replace the entire block from `import asyncio` through the results list comprehension (lines ~382-445) with:

```python
        import asyncio

        dense_task = services.embeddings.embed_query(query)
        sparse_task = asyncio.to_thread(
            services.sparse_embeddings.embed_query, query,
        )

        dense_vector, sparse_vector = await asyncio.gather(
            dense_task, sparse_task,
        )

        over_fetch = min(limit * 3, 50)
        filter_dict = {field: value} if field and value else None

        results = await services.vectorstore.search_hybrid_rrf(
            collection=coll,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=over_fetch,
            match_filters=filter_dict,
        )
```

Keep the existing reranking block (lines ~447-467) unchanged. It uses `results[item.index]` which still works since `search_hybrid_rrf` returns the same `SearchResult` objects.

Remove the now-unused imports that were inside the function:
- `from qdrant_client import models as qmodels`
- `from clorag.core.vectorstore import SearchResult`

**Note:** Check if `SearchResult` is referenced elsewhere in the function. If the reranking block references it by name in a type annotation, keep the import. Otherwise remove it.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/clorag/mcp/tools/chunks.py`
Expected: No errors.

- [ ] **Step 3: Run type checker**

Run: `uv run mypy src/clorag/mcp/tools/chunks.py --strict 2>&1 | head -20`
Expected: No new errors. If `SearchResult` is needed for mypy, add it back as a `TYPE_CHECKING` import.

- [ ] **Step 4: Commit**

```bash
git add src/clorag/mcp/tools/chunks.py
git commit -m "refactor: search_chunks uses public VectorStore API instead of _client"
```

---

### Task 4: HTTP Bearer auth via `MCP_API_KEY`

**Files:**
- Modify: `src/clorag/config.py` (add field)
- Create: `src/clorag/mcp/auth.py`
- Modify: `src/clorag/mcp/server.py` (rewrite main_http)
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write tests for the auth middleware**

Add to `tests/test_mcp.py`:

```python
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _make_test_app(api_key: str) -> Starlette:
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
        client = TestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_missing_auth_header(self):
        app = _make_test_app("test-secret-key")
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 401
        assert "Missing Bearer token" in resp.json()["error"]

    def test_invalid_token(self):
        app = _make_test_app("test-secret-key")
        client = TestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]

    def test_malformed_auth_header(self):
        app = _make_test_app("test-secret-key")
        client = TestClient(app)
        resp = client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401

    def test_bearer_with_extra_whitespace(self):
        app = _make_test_app("test-secret-key")
        client = TestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer  test-secret-key  "})
        assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp.py::TestBearerAuthMiddleware -v`
Expected: FAIL — `clorag.mcp.auth` does not exist yet.

- [ ] **Step 3: Add `mcp_api_key` field to config.py**

In `src/clorag/config.py`, find the `# Security Settings` section (~line 111). Add after `secure_cookies`:

```python
    mcp_api_key: SecretStr | None = Field(
        default=None,
        description="API key for MCP HTTP transport (Bearer token). Not required for stdio.",
    )
```

- [ ] **Step 4: Create `src/clorag/mcp/auth.py`**

```python
"""Bearer token auth wrapper for MCP HTTP transport."""

import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """ASGI middleware that requires a valid Bearer token on HTTP requests.

    Only checks HTTP requests. Non-HTTP scopes (lifespan, etc.) pass through.
    Uses secrets.compare_digest for timing-safe comparison.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self._key = api_key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            request = Request(scope)
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                resp = JSONResponse(
                    status_code=401,
                    content={"error": "Missing Bearer token"},
                )
                await resp(scope, receive, send)
                return
            token = auth.removeprefix("Bearer ").strip().encode()
            if not secrets.compare_digest(token, self._key):
                resp = JSONResponse(
                    status_code=401,
                    content={"error": "Invalid API key"},
                )
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def apply_bearer_auth(app: ASGIApp, api_key: str) -> ASGIApp:
    """Wrap an ASGI app with Bearer token authentication."""
    return BearerAuthMiddleware(app, api_key)
```

- [ ] **Step 5: Run auth tests**

Run: `uv run pytest tests/test_mcp.py::TestBearerAuthMiddleware -v`
Expected: All 5 PASS.

- [ ] **Step 6: Rewrite `main_http()` in server.py**

In `src/clorag/mcp/server.py`, replace the `main_http` function:

```python
def main_http() -> None:
    """Entry point for the MCP server (streamable-http transport for Docker/remote)."""
    import os

    import anyio
    import structlog
    import uvicorn

    from clorag.config import get_settings

    settings = get_settings()
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))
    mcp = create_mcp_server(host=host, port=port)

    # Get the Starlette app, wrap with auth if configured
    app = mcp.streamable_http_app()

    if settings.mcp_api_key:
        from clorag.mcp.auth import apply_bearer_auth

        app = apply_bearer_auth(app, settings.mcp_api_key.get_secret_value())
        structlog.get_logger().info("mcp_http_auth_enabled")
    else:
        structlog.get_logger().warning(
            "mcp_http_no_auth",
            msg="MCP HTTP running without authentication. Set MCP_API_KEY.",
        )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)
```

- [ ] **Step 7: Run linter on all changed files**

Run: `uv run ruff check src/clorag/mcp/auth.py src/clorag/mcp/server.py src/clorag/config.py`
Expected: No errors.

- [ ] **Step 8: Smoke test — verify `main_http` starts with lifespan**

The rewrite switches from `mcp.run()` to manual `uvicorn.Server` + `mcp.streamable_http_app()`. Verify the lifespan still fires (MCPServices init). Run briefly with `MCP_API_KEY` unset:

Run: `timeout 5 uv run clorag-mcp-http 2>&1 || true`
Expected: Server starts, logs show "mcp_http_no_auth" warning. No `RuntimeError("MCP services not initialized")`.

If it fails with a lifespan error, the `streamable_http_app()` approach needs adjustment.

- [ ] **Step 9: Commit**

```bash
git add src/clorag/config.py src/clorag/mcp/auth.py src/clorag/mcp/server.py tests/test_mcp.py
git commit -m "feat: add Bearer token auth for MCP HTTP transport via MCP_API_KEY"
```

---

### Task 5: Path sanitization with `MCP_IMPORT_BASE_DIR`

**Files:**
- Modify: `src/clorag/config.py` (add field)
- Modify: `src/clorag/mcp/tools/ingestion.py:135-144` (import_custom_documents)
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write tests for path sanitization**

Add to `tests/test_mcp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass (pure logic tests)**

Run: `uv run pytest tests/test_mcp.py::TestPathSanitization -v`
Expected: All 4 PASS (these test `pathlib` behavior, not our code yet).

- [ ] **Step 3: Add `mcp_import_base_dir` field to config.py**

In `src/clorag/config.py`, find the `mcp_api_key` field just added (Task 4). Add after it:

```python
    mcp_import_base_dir: str = Field(
        default="data/imports",
        description="Base directory for MCP document imports. Paths must resolve under this directory.",
    )
```

- [ ] **Step 4: Add path validation to `import_custom_documents`**

In `src/clorag/mcp/tools/ingestion.py`, find `import_custom_documents` (~line 113). Replace the path validation block:

```python
        from pathlib import Path

        from clorag.models.custom_document import DocumentCategory
        from clorag.scripts.import_documents import import_documents

        folder_path = Path(folder)
        if not folder_path.exists():
            return {"status": "error", "error": f"Folder not found: {folder}"}
        if not folder_path.is_dir():
            return {"status": "error", "error": f"Not a directory: {folder}"}
```

With:

```python
        from pathlib import Path

        from clorag.config import get_settings
        from clorag.models.custom_document import DocumentCategory
        from clorag.scripts.import_documents import import_documents

        settings = get_settings()
        base_dir = Path(settings.mcp_import_base_dir).resolve()
        folder_path = Path(folder).resolve()

        if not folder_path.is_relative_to(base_dir):
            return {
                "status": "error",
                "error": f"Path must be under {base_dir}. Got: {folder_path}",
            }
        if not folder_path.exists():
            return {"status": "error", "error": f"Folder not found: {folder_path}"}
        if not folder_path.is_dir():
            return {"status": "error", "error": f"Not a directory: {folder_path}"}
```

- [ ] **Step 5: Run linter**

Run: `uv run ruff check src/clorag/mcp/tools/ingestion.py src/clorag/config.py`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/clorag/config.py src/clorag/mcp/tools/ingestion.py tests/test_mcp.py
git commit -m "feat: add path sanitization for MCP import_custom_documents via MCP_IMPORT_BASE_DIR"
```

---

### Task 6: Enriched ingestion responses with timing and summaries

**Files:**
- Modify: `src/clorag/mcp/tools/ingestion.py` (all 7 tool functions)

- [ ] **Step 1: Add timing + summary to `ingest_docs`**

In `src/clorag/mcp/tools/ingestion.py`, find `ingest_docs` (~line 24). Replace:

```python
        from clorag.scripts.ingest_docs import run_ingestion

        try:
            count = await run_ingestion(
                url=None,
                fresh=fresh,
                extract_cameras=extract_cameras,
            )
            return {
                "status": "success",
                "documents_ingested": count,
                "fresh": fresh,
                "cameras_extracted": extract_cameras,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

With:

```python
        import time

        from clorag.scripts.ingest_docs import run_ingestion

        start = time.monotonic()
        try:
            count = await run_ingestion(
                url=None,
                fresh=fresh,
                extract_cameras=extract_cameras,
            )
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "documents_ingested": count,
                "duration_seconds": duration,
                "fresh": fresh,
                "cameras_extracted": extract_cameras,
                "summary": f"Ingested {count} documentation pages in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}
```

- [ ] **Step 2: Add timing + summary to `ingest_curated`**

Same pattern. Add `import time` and `start = time.monotonic()` before the `try` block in `ingest_curated` (~line 88). Replace the try/except block:

Success return becomes:
```python
            duration = round(time.monotonic() - start, 1)
            summary = f"Ingested {count} support cases in {duration}s"
            if offset:
                summary += f" (offset {offset})"
            if since_days:
                summary += f" (last {since_days} days)"
            summary += "."
            return {
                "status": "success",
                "cases_ingested": count,
                "duration_seconds": duration,
                "max_threads": max_threads,
                "offset": offset,
                "min_confidence": min_confidence,
                "fresh": fresh,
                "since_days": since_days,
                "summary": summary,
            }
```

Error return becomes:
```python
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}
```

- [ ] **Step 3: Add timing + summary to `import_custom_documents`**

In the try/except block of `import_custom_documents` (~line 154). Success return becomes:

```python
            duration = round(time.monotonic() - start, 1)
            prefix = "[DRY RUN] Would import" if dry_run else "Imported"
            return {
                "status": "success",
                "imported": imported,
                "skipped": skipped,
                "duration_seconds": duration,
                "folder": str(folder_path),
                "category": category,
                "tags": tag_list,
                "dry_run": dry_run,
                "summary": f"{prefix} {imported} documents, skipped {skipped} in {duration}s.",
            }
```

- [ ] **Step 4: Add timing + summary to `enrich_cameras`**

In `enrich_cameras` (~line 200). Success return:

```python
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "cameras_enriched": count,
                "duration_seconds": duration,
                "manufacturer": manufacturer,
                "limit": limit,
                "dry_run": dry_run,
                "summary": f"Enriched {count} cameras in {duration}s.",
            }
```

- [ ] **Step 5: Add timing + summary to `populate_graph`**

In `populate_graph` (~line 251). Success return:

```python
            duration = round(time.monotonic() - start, 1)
            formatted = ", ".join(f"{k}: {v}" for k, v in counts.items()) if isinstance(counts, dict) else str(counts)
            return {
                "status": "success",
                "collections": target_collections,
                "entity_counts": counts,
                "duration_seconds": duration,
                "summary": f"Extracted entities from {len(target_collections)} collections: {formatted} in {duration}s.",
            }
```

- [ ] **Step 6: Add timing + summary to `fix_rio_preview`**

In `fix_rio_preview` (~line 311). Success return:

```python
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "fixes_found": len(fixes),
                "fixes_saved": saved,
                "duration_seconds": duration,
                "max_chunks": max_chunks,
                "summary": f"Found {len(fixes)} fixes, saved {saved} suggestions in {duration}s.",
                "message": (
                    "Fixes saved as suggestions. Review and"
                    " approve via /admin/terminology-fixes"
                    " before applying."
                ),
            }
```

- [ ] **Step 7: Add timing + summary to `fix_rio_apply`**

In `fix_rio_apply` (~line 371). Success return:

```python
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "fixes_applied": count,
                "duration_seconds": duration,
                "summary": f"Applied {count} terminology fixes in {duration}s.",
            }
```

- [ ] **Step 8: Also add timing + summary to `rebuild_fts_index` and `init_prompts_db`**

These are sync tools but benefit from the same pattern. Add `import time` and wrap with `time.monotonic()`.

`rebuild_fts_index`:
```python
        import time

        start = time.monotonic()
        try:
            db = get_camera_database()
            count = db.rebuild_fts_index()
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "cameras_indexed": count,
                "duration_seconds": duration,
                "summary": f"Rebuilt FTS5 index for {count} cameras in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}
```

`init_prompts_db`:
```python
        import time

        start = time.monotonic()
        try:
            pm = get_prompt_manager()
            result = pm.initialize_defaults(force=force)
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "force": force,
                "duration_seconds": duration,
                "summary": f"Initialized prompts in {duration}s.",
                **result,
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}
```

- [ ] **Step 9: Run linter**

Run: `uv run ruff check src/clorag/mcp/tools/ingestion.py`
Expected: No errors.

- [ ] **Step 10: Commit**

```bash
git add src/clorag/mcp/tools/ingestion.py
git commit -m "feat: add timing and summary to all MCP ingestion tool responses"
```

---

### Task 7: Update .env.example and CLAUDE.md

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add MCP section to .env.example**

Add after the Security Settings section in `.env.example`:

```
# =============================================================================
# MCP Server Configuration
# =============================================================================

# API key for MCP HTTP transport (Bearer token)
# Not required for stdio transport (local use)
# MCP_API_KEY=your_mcp_api_key_here

# Base directory for MCP document imports
# Paths passed to import_custom_documents must resolve under this directory
# Default: data/imports (relative to CWD). Use absolute path in production.
# MCP_IMPORT_BASE_DIR=data/imports
```

- [ ] **Step 2: Update CLAUDE.md Configuration section**

In `CLAUDE.md`, find the Configuration section (under "Environment variables"). Add these lines to the bullet list:

```
- `MCP_API_KEY` - Bearer token for MCP HTTP transport auth (optional, stdio needs no auth)
- `MCP_IMPORT_BASE_DIR` (default: `data/imports`) - Base directory for MCP document imports (path containment)
```

- [ ] **Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: add MCP_API_KEY and MCP_IMPORT_BASE_DIR to env example and CLAUDE.md"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full linter**

Run: `uv run ruff check src/clorag/mcp/ src/clorag/core/vectorstore.py src/clorag/config.py`
Expected: No errors.

- [ ] **Step 2: Run full type checker**

Run: `uv run mypy src/clorag/mcp/ src/clorag/core/vectorstore.py --strict 2>&1 | tail -5`
Expected: No new errors from our changes.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass, including new ones.

- [ ] **Step 4: Verify git log**

Run: `git log --oneline -8`
Expected: 7 clean commits (Tasks 1-7).
