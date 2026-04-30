# CLAUDE.md

## Project Overview

CLORAG is a Multi-RAG agent for Cyanview support combining:
- **Docusaurus documentation** from the support site
- **Gmail support threads** (curated and anonymized)
- **Custom knowledge documents** (admin-managed)

Uses hybrid search (dense voyage-context-3 + sparse BM25 vectors with RRF fusion) + **Voyage rerank-2.5** cross-encoder for refined relevance across four Qdrant collections (main docs, gmail cases, custom docs, legacy docs). Claude Sonnet synthesizes responses with automatic Excalidraw diagrams (hand-drawn style) for integration scenarios.

**Version**: 0.11.0 | **Python**: 3.10-3.13

## Commands

```bash
uv sync                                    # Install dependencies
uv run rag-web                             # Web server (port 8080)
uv run clorag "query"                      # CLI agent

# Ingestion
uv run ingest-docs                         # Docusaurus documentation
uv run ingest-curated --max-threads 300    # Gmail threads (--offset N for incremental)
uv run import-docs ./folder --category pre_sales  # Bulk import custom docs

# Maintenance
uv run extract-cameras --from-sqlite       # Extract cameras from support cases
uv run extract-cameras --docs-only         # Extract cameras from Qdrant docs
uv run enrich-cameras                      # Enrich cameras with model codes + URLs
uv run verify-cameras                      # Verify/correct cameras via web (dry-run)
uv run verify-cameras --apply              # Apply corrections
uv run populate-graph                      # Build Neo4j knowledge graph
uv run draft-support                       # Auto-reply drafts (--preview)
uv run rebuild-fts                         # Rebuild camera FTS5 index
uv run fix-rio-terminology --preview       # Scan for RIO terminology issues
uv run fix-rio-terminology --apply         # Apply approved fixes
uv run init-prompts                        # Initialize prompt database with defaults
uv run init-prompts --force                # Reset all to defaults (auto-backups first)
uv run init-prompts --list                 # List all prompts (* = customized)
uv run init-prompts --stats                # Show prompt database stats
uv run init-prompts --backup               # Backup customized prompts to JSON
uv run init-prompts --backup -o file.json  # Backup to specific path
uv run init-prompts --restore FILE         # Restore customizations from backup
uv run init-settings                       # Initialize RAG settings with defaults
uv run init-settings --list                # List all settings by category
uv run init-settings --stats               # Show settings database stats
uv run archive-collection                  # Archive a Qdrant collection
uv run ingest-legacy-docs [path] --fresh   # Legacy docs from local markdown

# MCP Server
uv run clorag-mcp                          # MCP server (stdio transport)
uv run clorag-mcp-http                     # MCP server (HTTP transport, port 8080)

# Quality
uv run ruff check src/ && uv run mypy src/clorag --strict
uv run pytest                              # Full suite
uv run pytest tests/path/to/test_file.py::test_name -v   # Single test
uv run pytest --cov=src/clorag --cov-report=term-missing # With coverage
```

Tests live in `tests/`. Web routes use FastAPI's `TestClient`; database tests
use temp SQLite paths via fixtures. CLI entry points are defined in
`pyproject.toml` `[project.scripts]` — when adding a new script, register it
there and rerun `uv sync`.

## Architecture

```
Query → Voyage AI embeddings → Qdrant (hybrid RRF) → Reranker → Neo4j enrichment → Claude synthesis
              ↓                        ↓                  ↓
        BM25 sparse vectors      Over-fetch 3x      Voyage rerank-2.5
```

### Source Layout

**Core** (`core/`): `vectorstore.py` (AsyncQdrantClient, RRF fusion, dynamic prefetch, document-context operations via `get_chunks_by_field()`), `embeddings.py` (voyage-context-3 with contextualized_embed API), `sparse_embeddings.py` (BM25 with cache), `reranker.py` (Voyage rerank-2.5 cross-encoder), `metrics.py` (performance instrumentation), `retriever.py` (MultiSourceRetriever with reranking), `graph_store.py` (Neo4j), `entity_extractor.py` (Sonnet), `database.py` (camera SQLite with connection pool), `analytics_db.py`, `support_case_db.py` (support cases SQLite with FTS5 and connection pool), `prompt_db.py` (LLM prompts SQLite with version history), `settings_db.py` (RAG settings SQLite with version history), `ingestion_db.py` (ingestion job history SQLite), `terminology_db.py` (RIO terminology fixes SQLite storage), `cache.py` (generic thread-safe LRU cache with TTL)

**Ingestion** (`ingestion/`): `curated_gmail.py` (7-step: Fetch→Anonymize→Sonnet→Filter→Sonnet QC→Embed→Store), `docusaurus.py` (sitemap crawler with Jina Reader + BeautifulSoup fallback), `chunker.py`, `base.py`

**Analysis** (`analysis/`): `thread_analyzer.py` (Sonnet classification), `quality_controller.py` (Sonnet QC), `camera_extractor.py`, `rio_analyzer.py` (RIO terminology context analysis)

**Agent** (`agent/`): `tools.py` (Claude Agent SDK MCP tools), `prompts.py`

**MCP** (`mcp/`): Standalone MCP server for Claude Desktop and remote clients. `server.py` (FastMCP server with lifespan for stdio, manual init for HTTP), `tools/` (search, cameras, documents, support) exposing full RAG capabilities via stdio and StreamableHTTP transports.

**Graph** (`graph/`): `schema.py` (Camera, Product, Protocol, Issue, Solution entities), `enrichment.py`

**Services** (`services/`): `custom_docs.py` (CustomDocumentService CRUD), `prompt_manager.py` (LLM prompt management with caching), `default_prompts.py` (hardcoded prompt registry), `settings_manager.py` (RAG settings management with caching), `default_settings.py` (hardcoded settings registry)

**Drafts** (`drafts/`): `gmail_service.py`, `draft_generator.py`, `draft_pipeline.py`

**Web** (`web/`): FastAPI application with modular router architecture
- `app.py` - Middleware, lifespan, app initialization
- `routers/` - API routes by domain: `cameras.py`, `pages.py`, `search.py`, `legacy.py`, `admin/` (14 routers: analytics, auth, cameras, chunks, debug, documents, drafts, graph, messages, prompts, settings, support, terminology)
- `auth/` - Authentication: `admin.py`, `csrf.py`, `sessions.py`
- `schemas.py` - Request/response Pydantic models
- `search/` - Search pipeline: `pipeline.py`, `synthesis.py`, `utils.py`
- `dependencies.py` - FastAPI dependency injection
- `templates/` - Jinja2 templates including `/admin/docs` (doc pages), legacy (`legacy*.html`), and the `ai_lexicon.html` standalone page
- Admin UI at `/admin/{cameras,cameras-list,knowledge,analytics,metrics,drafts,chunks,graph,support-cases,prompts,settings,ingestion,terminology-fixes,messages}`, REST APIs at `/api/`
- Legacy UI at `/legacy`, `/legacy/manage` (admin-protected), `/legacy/help`

**Models** (`models/`): `camera.py`, `custom_document.py` (10 categories), `support_case.py`

**Utils** (`utils/`): `token_encryption.py` (Fernet/PBKDF2), `anonymizer.py`, `logger.py`, `tokenizer.py` (tiktoken token counting), `text_transforms.py` (RIO product name transformations)

### Vector Collections

Four Qdrant collections each with `dense` (1024-dim voyage-context-3) + `sparse` (BM25) vectors:
- `docusaurus_docs` - Main documentation (support.cyanview.cloud)
- `gmail_cases` - Curated support threads
- `custom_docs` - Admin-managed knowledge
- `docusaurus_docs_legacy` - Legacy docs (support.cyanview.com), independent from main search

## Configuration

Environment variables (see `.env.example`):
- `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` - Required
- `QDRANT_URL`, `DOCUSAURUS_URL`, `GMAIL_LABEL`
- `DATABASE_PATH` (default: `data/clorag.db`), `ANALYTICS_DATABASE_PATH` (default: `data/analytics.db`)
- `ADMIN_PASSWORD` - Admin auth + OAuth token encryption key
- `SEARXNG_URL` (default: `https://search.sapti.me`)
- `SECURE_COOKIES` - Set `false` for local dev
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` - Optional GraphRAG
- `RERANK_ENABLED` (default: `true`) - Enable/disable reranking
- `VOYAGE_RERANK_MODEL` (default: `rerank-2.5`) - Reranker model
- `RERANK_TOP_K` (default: `5`) - Results after reranking
- `CHUNK_USE_TOKENS` (default: `true`) - Token-based chunking (vs character-based)
- `CHUNK_SIZE_DOCS` (default: `450`) - Chunk size for documentation (tokens)
- `CHUNK_SIZE_CASES` (default: `350`) - Chunk size for support cases (tokens)
- `CHUNK_SIZE_DEFAULT` (default: `400`) - Default chunk size (tokens)
- `CHUNK_OVERLAP` (default: `50`) - Chunk overlap (~12.5%)
- `CHUNK_ADAPTIVE_THRESHOLD` (default: `200`) - Single-chunk threshold (tokens)
- `OPENAI_COMPAT_API_KEY` - Optional, enables `/v1/chat/completions` OpenAI-compatible API
- `MCP_API_KEY` - Bearer token for MCP HTTP transport auth (required for HTTP, not needed for stdio). Docker secret: `/run/secrets/mcp_api_key`
- `MCP_IMPORT_BASE_DIR` (default: `data/imports`) - Base directory for MCP document imports (path containment)
- `QDRANT_LEGACY_DOCS_COLLECTION` (default: `docusaurus_docs_legacy`) - Separate collection for legacy site
Settings via `clorag.config.get_settings()` (cached singleton).

## Key Patterns

### Search & Retrieval
- **Hybrid search**: Dense + sparse vectors combined via RRF across all collections
- **Reranking**: Voyage `rerank-2.5` cross-encoder refines top results (+15-40% relevance improvement)
- **Over-fetch strategy**: Retrieves N×limit (configurable, default 3x), reranks, returns top-K for optimal quality
- **Threshold after reranking**: Dynamic thresholds applied AFTER reranking (calibrated 0-1 scores). RRF scores are uncalibrated and skip threshold filtering without reranking.
- **Unified threshold logic**: Single `calculate_dynamic_threshold()` in `core/retriever.py` shared by CLI and web pipelines. 30+ technical terms (visca, sdi, hdmi, ndi, srt, ptz, etc.)
- **Dynamic thresholds**: ≤2 words: 0.15, 3-5: 0.20, >5: 0.25; technical terms +0.05; minimum 3 results. All values configurable via `/admin/settings`
- **Source diversity**: Post-merge interleaving ensures at least 1 result from each collection with relevant hits (score ≥ configurable % of top result, default 50%)
- **Query cache**: LRU caches for embeddings, sparse vectors, and reranking (sizes configurable via admin settings, require restart)
- **Search quality logging**: Scores + source types logged per query; `/api/admin/search-quality` for low-score review
- **Contextualized embeddings**: `voyage-context-3` uses `/v1/contextualizedembeddings` API (not `/v1/embeddings`) to encode chunks with full document context
- **Synthesis grounding**: Prompt instructs "I don't know" on insufficient context; prefers docs over cases on conflicts
- **User feedback**: Thumbs up/down per answer with optional comment on downvote. Stored in `search_feedback` table (upsert per search_id). Admin sees satisfaction rate + recent feedback at `/admin/analytics`
- **Conversation grounding**: Follow-up questions use history for intent only (e.g., "and the FX6?" → "connect the FX6"); facts from previous answers are never mixed into current response. Enforced via message-level separator + `<conversation_grounding>` prompt rule

### Chunking
- **Token-based sizing**: Uses `tiktoken` (cl100k_base) for 15-20% more consistent chunk sizes vs character-based
- **Content-type specific**: Documentation (450 tokens), Support cases (350 tokens), Default (400 tokens)
- **Factory method**: `SemanticChunker.from_settings(ContentType.DOCUMENTATION)` for config-driven creation
- **Atomic blocks**: Code blocks, tables, and markdown headings preserved as units
- **Adaptive threshold**: Short content (< 200 tokens) stays as single chunk
- **Backward compatible**: Set `CHUNK_USE_TOKENS=false` for character-based mode

### Docusaurus Ingestion
- **Jina Reader primary**: Uses `r.jina.ai` JSON API (`Accept: application/json`) for structured title + content extraction
- **Jina noise reduction**: `X-Retain-Images: none`, `X-Target-Selector` for Docusaurus content, `X-Remove-Selector` for nav/sidebar/ToC
- **BeautifulSoup fallback**: Automatic fallback on Jina 429/503 errors with retry logic (3 attempts)
- **Table preservation**: HTML tables converted to markdown format before text extraction (BeautifulSoup fallback)
- **Keyword extraction**: Sonnet extracts 5-10 technical keywords per page (parallel, 10 concurrent). Samples first 2000 + last 2000 chars for full page coverage. Stored on all chunks
- **RIO terminology fixes**: High-confidence fixes auto-applied during ingestion before embedding
- **Camera extraction**: Sonnet extracts camera compatibility info post-ingestion

### Data Protection
- **Anonymization**: Gmail threads use placeholders (`[SERIAL:XXX-N]`, `[EMAIL-N]`) before LLM processing
- **Human edits preserved**: Admin-modified chunks/cameras are NOT overwritten by automated ingestion
- **Camera extraction**: Sonnet extracts camera info during ingestion; upsert merges multiple sources

### Camera Database
- **FTS5 search**: Full-text search with BM25 ranking and Porter stemming via SQLite FTS5 virtual table
- **TTL cache**: Thread-safe LRU cache (100 entries, 5-min TTL) for list/search/stats queries
- **Connection pool**: 5-connection pool with WAL mode, 64MB cache, automatic corruption recovery
- **Comparison**: Side-by-side comparison of up to 5 cameras with highlighted common specs
- **Related cameras**: Similarity scoring based on manufacturer, device_type, ports, protocols
- **CSV import/export**: Bulk data management with upsert logic (name + manufacturer)
- **Duplicate detection**: `find_duplicate_candidates()` normalizes names (Mark II→mk2, strips hyphens/spaces), cross-references code_model↔name, groups with union-find
- **Camera merge**: `merge_cameras()` unions array fields, keeps primary scalars with fallback, max confidence. Neo4j `camera_db_id` reassigned via `reassign_camera_db_id()`

### Support Cases Database
- **SQLite storage**: Full document storage with problem/solution summaries, keywords, categories
- **FTS5 search**: Full-text search across subject, problem, solution, document, keywords
- **Thread cleaning**: `clean_thread_quotes()` removes quoted replies, headers, signatures (EN/FR/DE)
- **Connection pool**: Reuses `ConnectionPool` from camera database for concurrent access
- **Admin UI**: Browse/search cases at `/admin/support-cases` with detail modal (Summary/Document/Raw tabs)

### Performance Monitoring
- **Metrics collection**: `MetricsCollector` with sliding window (1000 entries), percentiles (p50/p90/p95/p99)
- **Pipeline instrumentation**: Timing for embedding_generation, vector_search, total_search, llm_synthesis
- **Parallel embeddings**: Dense + sparse generated concurrently via `asyncio.gather()`
- **Sparse model preload**: BM25 model loaded at startup to eliminate cold start latency
- **Async Voyage client**: Non-blocking API calls with `voyageai.AsyncClient`
- **Dynamic prefetch**: Scales with limit (configurable multiplier and max, default 3x/50) for better RRF fusion quality
- **Cache stats endpoint**: `/api/admin/cache-stats` for hit/miss rates with recommendations
- **Metrics endpoint**: `/api/admin/metrics` with thresholds, alerts, and percentile breakdowns

### Security
- Session-based admin auth with brute force protection (5 attempts → 5min lockout per IP)
- XSS: DOMPurify with SVG allowlist for Excalidraw diagrams
- OAuth tokens: Fernet encryption with PBKDF2 (480K iterations)
- Secure cookies in production (configurable via `SECURE_COOKIES`)
- **CSP Policy**: Differentiated per page type:
  - Public pages: Strict nonce-only (`script-src 'nonce-...'`)
  - Admin pages: Allows inline handlers (`unsafe-inline`) while templates migrate to `data-action`
- **Event Delegation**: `AdminActions` in `admin.js` handles `data-action` attributes globally
- **Dark Mode**: `ThemeToggle` in `admin.js` with localStorage persistence, `prefers-color-scheme` detection, 30+ CSS variables
- **Animations**: `AdminAnimations` in `admin.js` — stagger entrance, count-up, animated alerts/modals, `prefers-reduced-motion` respected

### GraphRAG
Optional Neo4j knowledge graph with entities (Camera, Product, Protocol, Port, Issue, Solution, Firmware) and relationships (COMPATIBLE_WITH, USES_PROTOCOL, AFFECTS, RESOLVED_BY, etc.). Gracefully disabled if `NEO4J_PASSWORD` not set.

**Local dev**: SSH tunnel to production Neo4j:
```bash
ssh -L 7687:localhost:7687 root@cyanview.cloud -N -f
```

### OpenAI-Compatible API
- **Endpoint**: `POST /v1/chat/completions` — accepts standard OpenAI `ChatCompletion` request format
- **Auth**: Bearer token via `OPENAI_COMPAT_API_KEY` env var
- **Models endpoint**: `GET /v1/models` — returns `clorag` model
- **Query routing**: Last user message → RAG search query, prior messages → conversation history
- **Streaming**: `stream: true` returns SSE in OpenAI chunk format (`chat.completion.chunk`)
- **Sources**: Appended as markdown links at the end of the response
- **Usage**: Any OpenAI SDK client can connect with `base_url="https://cyanview.cloud"` and `api_key="<OPENAI_COMPAT_API_KEY>"`

### Custom Documents
10 categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, pre_sales, internal, other. Supports .txt/.md/.pdf upload, full metadata, chunked and embedded into RAG search. Sonnet-generated keywords auto-enriched alongside user-provided tags.

### Messages
Admin-managed announcements displayed on the public index page. Collapsible "Latest Updates" panel between hero and quick-action tiles. 4 message types: info (blue), warning (orange), feature (green), fix (red). Stored in `messages` table in analytics SQLite DB. Auto-hides expired messages. Hidden entirely when 0 active messages. Admin CRUD at `/admin/messages`, public API at `GET /api/messages`.

### Prompt Management
- **Admin-editable prompts**: 13 LLM prompts stored in SQLite, editable via `/admin/prompts` without code changes
- **Composable prompt architecture**: 3 base blocks composed at call time via `get_composed_prompt()`:
  - `base.identity` — Cyanview team voice, response rules, formatting, language (product-free)
  - `base.product_reference` — Product ecosystem, connection rules, decision points
  - Interface layer: `synthesis.web_layer` (web) or `agent.tools_layer` (CLI)
- **Composition per pipeline**:
  - Main web/CLI: `base.identity` + `base.product_reference` + layer (full product knowledge)
  - Legacy search: `base.identity` + `synthesis.web_layer` (no product knowledge — answers from docs only)
  - Analysis/drafts: use `{product_reference}` variable injection from `base.product_reference`
- **Shared product knowledge**: `base.product_reference` is the single source of truth for all product facts. Injected via `{product_reference}` variable into 6 prompts: `thread_analyzer`, `quality_controller`, `camera_extractor`, `rio_terminology`, `entity_extractor`, `email_generator`. Edit once at `/admin/prompts` to update everywhere.
- **Version history**: Every content change creates a new version for audit and rollback
- **Fallback to defaults**: If DB prompt not found, falls back to hardcoded defaults in `default_prompts.py`
- **Backup/restore**: `init-prompts --backup` exports customized prompts to JSON; `--restore FILE` re-applies them. `--force` auto-backups before resetting
- **Customization detection**: Compares DB content against hardcoded defaults to identify truly customized prompts
- **Caching**: In-memory cache with TTL (default: 300s) for performance, hot reload via API
- **Variable substitution**: `{variable}` placeholders auto-detected and substituted at runtime
- **Categories**: agent, analysis, base, synthesis, drafts, graph, scripts
- **Configuration**: `PROMPTS_CACHE_TTL` (default: 300 seconds)
- **API usage**: `pm = get_prompt_manager(); prompt = pm.get_prompt("analysis.thread_analyzer", thread_content="...")`
- **Composed prompts**: `get_composed_prompt("base.identity", "base.product_reference", "synthesis.web_layer")`

### RAG Settings
- **Admin-editable settings**: 20 RAG tuning parameters stored in SQLite, editable via `/admin/settings` without code changes
- **Version history**: Every value change creates a new version for audit and rollback
- **Fallback to defaults**: If DB setting not found, falls back to hardcoded defaults in `default_settings.py`
- **Caching**: In-memory cache with TTL (default: 300s) for performance, hot reload via API
- **Type-safe getters**: `get_setting(key)` returns typed values (int/float/bool) with validation
- **Categories**: retrieval (7), reranking (2), synthesis (4), caches (5), prefetch (2)
- **Restart-required settings**: Cache sizes (query embedding, sparse, reranker, camera DB) read at init time, marked with badge in UI
- **API usage**: `from clorag.services.settings_manager import get_setting; threshold = get_setting("retrieval.short_query_threshold")`
- **Integration**: Wired into `retriever.py` (thresholds, overfetch, min results), `vectorstore.py` (prefetch, source diversity), `synthesis.py` (max_tokens), `utils.py` (context budgets), `pipeline.py` (overfetch), cache modules (sizes at init)

### Legacy Docs System
Completely independent RAG for `support.cyanview.com` (the existing production site), separate from the main CLORAG search engine. Used by David to search, review, and update the legacy support documentation.

- **Separate collection**: `docusaurus_docs_legacy` — no interaction with `docusaurus_docs`, `gmail_cases`, or `custom_docs`
- **Public search**: `/legacy` with streaming AI answers from Claude Sonnet, source links to `support.cyanview.com` (no URL rewriting)
- **Manage page**: `/legacy/manage` (admin auth required) — scan sitemap for new pages, ingest single page by URL, full site re-crawl
- **Help page**: `/legacy/help` — usage documentation
- **Local markdown ingestion**: `uv run ingest-legacy-docs` reads `.md` files from Docusaurus sources, reconstructs URLs from file paths, preserves original text (no RIO terminology transforms)
- **Live site ingestion**: "Full Re-ingest" and "Ingest Page" use the standard `DocusaurusIngestionPipeline` (Jina Reader + keyword extraction)
- **Timeout exemption**: `/api/legacy/*` endpoints exempt from 60s middleware timeout (ingestion can take minutes)
- **Router**: `web/routers/legacy.py` — scan, ingest-new, ingest-page, reingest-full, stats endpoints
- **Templates**: `legacy.html`, `legacy_manage.html`, `legacy_help.html` — brown theme to distinguish from main blue UI

### RIO Product Terminology
- **RIO**: Generic RIO hardware reference. Use when license is NOT relevant: physical dimensions, ports, grounding, power, wiring, mounting, weight
- **RIO +WAN**: Full license. LAN & WAN connectivity, Cyanview cloud access, REMI mode, uses Internet, 1-128 cameras
- **RIO +LAN**: Formerly "RIO-Live". LAN only, single camera companion (max 2). LAN production robustness. No WAN/cloud/REMI
- Legacy terms ("RIO-Live", "RIO Live", "RIOLive", "RIO +WAN Live") all map to "RIO +LAN"
- Context rules: cloud/REMI/Internet/WAN/remote/multi-camera(>2) → **RIO +WAN** | LAN-only/local/1-2 cameras → **RIO +LAN** | physical/hardware → generic **RIO**
- **Query normalization**: Legacy terms in search queries are auto-normalized before embedding (via `apply_product_name_transforms`)
- **Pre-ingestion fixes**: High-confidence fixes (≥0.85) auto-applied during ingestion before embedding
- **Configuration**: `RIO_FIX_ON_INGEST` (default: true), `RIO_FIX_MIN_CONFIDENCE` (default: 0.85)
- Admin UI at `/admin/terminology-fixes` for reviewing and applying corrections to existing chunks
- **Document-context re-embedding**: When fixes are applied post-ingestion, ALL sibling chunks re-embedded together
- **Grouping fields**: `url` (docusaurus_docs), `thread_id` (gmail_cases), `parent_doc_id` (custom_docs)

## Deployment

```bash
rsync -avz \
  --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='data' --exclude='node_modules' \
  --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
  --exclude='.env' --exclude='secrets' --exclude='logs' \
  --exclude='.DS_Store' --exclude='token*.json' \
  /Users/alanogic/dev/clorag/ root@cyanview.cloud:/opt/clorag/

ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

CRITICAL excludes — without these, rsync clobbers production:
- `.env` — overwrites prod secrets and may contain malformed lines that crash `pydantic-settings`
- `secrets/` — rsync `-a` preserves local file modes; the `clorag` container user (different uid) needs **644** on all secret files (`/run/secrets/<name>`). A 600 file → `PermissionError` at startup.
- `logs/`, `.DS_Store`, `token*.json` — pollute the prod tree.

Verify post-deploy:
```bash
curl -sI https://cyanview.cloud/                                  # 200
ssh root@cyanview.cloud "cd /opt/clorag && docker compose ps"     # all healthy
ssh root@cyanview.cloud "cd /opt/clorag && docker compose logs clorag-web --tail=30"
```

Production:
- Web: https://cyanview.cloud/ (Docker maps 8085→8080)
- MCP HTTP: https://mcp.cyanview.cloud/ (Docker maps 8086→8080, Bearer auth required)

## Change History

For release notes, refactors, and feature history, use `git log` and commit messages
— the project follows Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.).
Don't add release-note sections here; they rot. If a change introduces a non-obvious
constraint or invariant that future Claude sessions need (e.g., "RRF scores are
uncalibrated, never threshold them pre-rerank"), add it under the relevant **Key
Patterns** subsection above.
