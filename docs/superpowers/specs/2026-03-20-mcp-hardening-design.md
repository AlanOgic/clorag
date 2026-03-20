# MCP Server Hardening — Design Spec

**Date:** 2026-03-20
**Scope:** 6 improvements to `clorag-mcp` covering security, correctness, and quality-of-life.
**Approach:** Incremental — 6 independent changes, ordered by dependency.

## Overview

| # | Change | Files | Lines |
|---|---|---|---|
| 1 | Replace deprecated `get_event_loop()` with `asyncio.to_thread()` | `mcp/tools/chunks.py` | ~4 |
| 2 | Add `match_filters` param to `search_hybrid_rrf()` | `core/vectorstore.py` | ~10 |
| 3 | `search_chunks` uses public VectorStore API | `mcp/tools/chunks.py` | ~30 replaced |
| 4 | HTTP Bearer auth via `MCP_API_KEY` | `config.py`, `mcp/server.py`, new `mcp/auth.py` | ~40 new |
| 5 | Path sanitization with `MCP_IMPORT_BASE_DIR` | `config.py`, `mcp/tools/ingestion.py` | ~10 |
| 6 | Enriched ingestion responses with timing and summaries | `mcp/tools/ingestion.py` | ~35 |

**Execution order:** 1 → 2 → 3 (depends on 2) → 4, 5, 6 (independent).
**Breaking changes:** None. All existing callers unaffected. Stdio transport unchanged.

---

## 1. Replace `asyncio.get_event_loop()` with `asyncio.to_thread()`

**Problem:** `chunks.py` uses `asyncio.get_event_loop().run_in_executor()` at two locations (lines 251, 389). This is deprecated in Python 3.10+, raises `DeprecationWarning` in 3.12, and will error in a future version. The web pipeline already uses `asyncio.to_thread()` correctly.

**Change:** Replace both occurrences. Note the two calls have different signatures:

In `edit_chunk` (line 251) — batch embed for document text:
```python
# Before
sparse_task = asyncio.get_event_loop().run_in_executor(
    None, services.sparse_embeddings.embed_texts, [text],
)
# After
sparse_task = asyncio.to_thread(services.sparse_embeddings.embed_texts, [text])
```

In `search_chunks` (line 389) — single query embed:
```python
# Before
sparse_task = asyncio.get_event_loop().run_in_executor(
    None, services.sparse_embeddings.embed_query, query,
)
# After
sparse_task = asyncio.to_thread(services.sparse_embeddings.embed_query, query)
```

**Files:** `src/clorag/mcp/tools/chunks.py`

---

## 2. Add filter parameter to `search_hybrid_rrf()`

**Problem:** `search_chunks` in `chunks.py` accesses `services.vectorstore._client.query_points()` directly because the public `search_hybrid_rrf()` method doesn't support passing a Qdrant filter. This bypasses abstractions (no metrics, no cache, no logging).

**Change:** Add an optional `match_filters` parameter to `search_hybrid_rrf()` in `vectorstore.py`. Named `match_filters` (not `match_filters`) to clearly indicate it supports exact-match filters only — not range, list, or complex conditions:

```python
async def search_hybrid_rrf(
    self,
    collection: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    limit: int = 10,
    match_filters: dict[str, Any] | None = None,  # NEW — exact match only
) -> list[SearchResult]:
```

Inside the method, build a Qdrant filter from the dict (reusing the same pattern as `scroll_chunks`):

```python
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

Pass `filter=query_filter` to both `Prefetch` blocks (dense and sparse). Defaults to `None` — fully backward-compatible, no changes to existing callers.

**Files:** `src/clorag/core/vectorstore.py`

---

## 3. `search_chunks` uses public VectorStore API

**Problem:** `search_chunks` manually constructs Qdrant prefetch/fusion queries by accessing the private `_client` attribute. This duplicates logic already in `VectorStore`, skips metrics and logging, and couples the MCP layer to Qdrant internals.

**Change:** Replace the ~50-line manual query block with:

```python
# Generate embeddings in parallel
dense_task = services.embeddings.embed_query(query)
sparse_task = asyncio.to_thread(
    services.sparse_embeddings.embed_query, query,
)
dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)

# Build filter dict from field/value
filter_dict = {field: value} if field and value else None

# Use public API (enabled by change #2)
over_fetch = min(limit * 3, 50)
results = await services.vectorstore.search_hybrid_rrf(
    collection=coll,
    dense_vector=dense_vector,
    sparse_vector=sparse_vector,
    limit=over_fetch,
    match_filters=filter_dict,
)
```

This removes:
- Direct `_client` access
- Manual `qmodels.Prefetch` / `qmodels.FusionQuery` construction
- `from qdrant_client import models as qmodels` import

**Note:** The `from clorag.core.vectorstore import SearchResult` import is no longer needed at runtime (duck typing), but keep it if `mypy --strict` requires the type. Verify during implementation.

The existing reranking logic (lines 447-467) stays unchanged — it operates on the results list regardless of how they were fetched.

**Files:** `src/clorag/mcp/tools/chunks.py`
**Depends on:** Change #2

---

## 4. HTTP Bearer auth via `MCP_API_KEY`

**Problem:** The HTTP transport (`main_http`) binds `0.0.0.0:8080` with zero authentication. Anyone who can reach the port has full access: read all data, modify prompts, delete chunks, trigger ingestion.

### config.py

Add one field to `Settings`:

```python
mcp_api_key: SecretStr | None = Field(
    default=None,
    description="API key for MCP HTTP transport (Bearer token). Not required for stdio.",
)
```

### New file: `src/clorag/mcp/auth.py`

Pure ASGI middleware wrapping the Starlette app — no monkey-patching, no private attributes:

```python
"""Bearer token auth wrapper for MCP HTTP transport."""
import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """ASGI middleware that requires a valid Bearer token on HTTP requests."""

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

### server.py

Replace `main_http()`:

```python
def main_http() -> None:
    """Entry point for the MCP server (streamable-http transport for Docker/remote)."""
    import os

    import anyio
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
    else:
        import structlog

        structlog.get_logger().warning(
            "mcp_http_no_auth",
            msg="MCP HTTP running without authentication. Set MCP_API_KEY.",
        )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)
```

**Lifespan safety:** Verified that `streamable_http_app()` returns a Starlette app with `lifespan=lambda app: self.session_manager.run()` wired in (FastMCP server.py line 1019). Wrapping with ASGI middleware preserves the lifespan chain — `MCPServices` init/cleanup still runs correctly.

**Transport note:** Streamable-http is pure HTTP (POST + SSE). No WebSocket. The middleware only checks `scope["type"] == "http"` which covers all MCP traffic.

**Behavior matrix:**

| `MCP_API_KEY` | Transport | Auth enforced? |
|---|---|---|
| Not set | stdio | No (local, expected) |
| Not set | HTTP | No, but warning logged |
| Set | stdio | No (ignored) |
| Set | HTTP | Yes, Bearer token required |

**Files:** `src/clorag/config.py`, `src/clorag/mcp/server.py`, new `src/clorag/mcp/auth.py`

---

## 5. Path sanitization with `MCP_IMPORT_BASE_DIR`

**Problem:** `import_custom_documents` accepts any absolute path. A malicious or confused client could pass `/etc`, `/root/.ssh`, or use `..` traversal to read arbitrary filesystem content into the RAG.

### config.py

Add one field:

```python
mcp_import_base_dir: str = Field(
    default="data/imports",
    description="Base directory for MCP document imports. Paths must resolve under this directory.",
)
```

### ingestion.py

Replace the current path validation in `import_custom_documents`:

```python
from clorag.config import get_settings

settings = get_settings()
base_dir = Path(settings.mcp_import_base_dir).resolve()
folder_path = Path(folder).resolve()

# Path traversal check — before existence check
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

Key details:
- `resolve()` collapses `..`, symlinks, and relative paths to absolute
- `is_relative_to()` (Python 3.9+) is the standard containment check
- Validation before `exists()` — even non-existent paths outside the base dir are rejected
- Default `data/imports/` is relative — resolves from CWD at runtime. In Docker this is `/opt/clorag/data/imports`. For production, set an absolute path via `MCP_IMPORT_BASE_DIR`

**Files:** `src/clorag/config.py`, `src/clorag/mcp/tools/ingestion.py`

---

## 6. Enriched ingestion responses

**Problem:** Ingestion tools return bare counts (`{"status": "success", "documents_ingested": 42}`). The LLM client has no context on duration, what happened during the run, or a human-readable summary to relay.

**Change:** Wrap each ingestion tool's execution with `time.monotonic()` and add `duration_seconds` + `summary` fields to the response. Apply to all 7 ingestion/maintenance tools:

- `ingest_docs`
- `ingest_curated`
- `import_custom_documents`
- `enrich_cameras`
- `populate_graph`
- `fix_rio_preview`
- `fix_rio_apply`

**Pattern (using `ingest_docs` as example):**

```python
import time

start = time.monotonic()
try:
    count = await run_ingestion(...)
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

**Error responses also get `duration_seconds`** — useful for diagnosing timeouts ("error after 1800s" = ran 30 minutes before failing).

**The `summary` field** gives the LLM client a ready-to-display message without interpreting raw numbers.

**Summary templates per tool:**

| Tool | Summary template |
|---|---|
| `ingest_docs` | `"Ingested {count} documentation pages in {duration}s."` |
| `ingest_curated` | `"Ingested {count} support cases in {duration}s (offset {offset})."` |
| `import_custom_documents` | `"Imported {imported} documents, skipped {skipped} in {duration}s."` (prefixed with `[DRY RUN]` if applicable) |
| `enrich_cameras` | `"Enriched {count} cameras in {duration}s."` |
| `populate_graph` | `"Extracted entities from {collections}: {formatted_counts} in {duration}s."` (format entity_counts dict as `"Camera: 12, Protocol: 8, ..."`) |
| `fix_rio_preview` | `"Found {found} fixes, saved {saved} suggestions in {duration}s."` |
| `fix_rio_apply` | `"Applied {count} terminology fixes in {duration}s."` |

**Files:** `src/clorag/mcp/tools/ingestion.py`

---

## Environment Variables Summary

| Variable | Default | Required | Description |
|---|---|---|---|
| `MCP_API_KEY` | None | No | Bearer token for HTTP transport auth |
| `MCP_IMPORT_BASE_DIR` | `data/imports` | No | Allowed base directory for document imports |
| `MCP_HOST` | `0.0.0.0` | No | HTTP transport bind address (existing) |
| `MCP_PORT` | `8080` | No | HTTP transport port (existing) |

## Files Changed

| File | Action | Changes |
|---|---|---|
| `src/clorag/config.py` | Modified | Add `mcp_api_key`, `mcp_import_base_dir` fields |
| `src/clorag/core/vectorstore.py` | Modified | Add `match_filters` param to `search_hybrid_rrf()` |
| `src/clorag/mcp/auth.py` | **New** | Bearer token ASGI middleware (~30 lines) |
| `src/clorag/mcp/server.py` | Modified | Rewrite `main_http()` with auth wrapping |
| `src/clorag/mcp/tools/chunks.py` | Modified | Fix `asyncio.to_thread()`, use public API for search |
| `src/clorag/mcp/tools/ingestion.py` | Modified | Path sanitization, enriched responses |
| `.env.example` | Modified | Add `MCP_API_KEY`, `MCP_IMPORT_BASE_DIR` |
| `CLAUDE.md` | Modified | Document new env vars in Configuration section |
