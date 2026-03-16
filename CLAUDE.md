# CLAUDE.md

## Project Overview

CLORAG is a Multi-RAG agent for Cyanview support combining:
- **Docusaurus documentation** from the support site
- **Gmail support threads** (curated and anonymized)
- **Custom knowledge documents** (admin-managed)

Uses hybrid search (dense voyage-context-3 + sparse BM25 vectors with RRF fusion) + **Voyage rerank-2.5** cross-encoder for refined relevance across three Qdrant collections. Claude Sonnet synthesizes responses with automatic Excalidraw diagrams (hand-drawn style) for integration scenarios.

**Version**: 0.8.0 | **Python**: 3.10-3.13

## Commands

```bash
uv sync                                    # Install dependencies
uv run rag-web                             # Web server (port 8080)
uv run clorag "query"                      # CLI agent
uv run clorag-mcp                          # MCP server for Claude Desktop

# Ingestion
uv run ingest-docs                         # Docusaurus documentation
uv run ingest-curated --max-threads 300    # Gmail threads (--offset N for incremental)
uv run import-docs ./folder --category pre_sales  # Bulk import custom docs

# Maintenance
uv run enrich-cameras                      # Extract camera info from docs
uv run populate-graph                      # Build Neo4j knowledge graph
uv run draft-support                       # Auto-reply drafts (--preview)
uv run rebuild-fts                         # Rebuild camera FTS5 index
uv run fix-rio-terminology --preview       # Scan for RIO terminology issues
uv run fix-rio-terminology --apply         # Apply approved fixes
uv run init-prompts                        # Initialize prompt database with defaults
uv run init-prompts --list                 # List all prompts
uv run init-prompts --stats                # Show prompt database stats

# Quality
uv run ruff check src/ && uv run mypy src/clorag --strict
uv run pytest
```

## Architecture

```
Query → Voyage AI embeddings → Qdrant (hybrid RRF) → Reranker → Neo4j enrichment → Claude synthesis
              ↓                        ↓                  ↓
        BM25 sparse vectors      Over-fetch 3x      Voyage rerank-2.5
```

### Source Layout

**Core** (`core/`): `vectorstore.py` (AsyncQdrantClient, RRF fusion, dynamic prefetch, document-context operations via `get_chunks_by_field()`), `embeddings.py` (voyage-context-3 with contextualized_embed API), `sparse_embeddings.py` (BM25 with cache), `reranker.py` (Voyage rerank-2.5 cross-encoder), `metrics.py` (performance instrumentation), `retriever.py` (MultiSourceRetriever with reranking), `graph_store.py` (Neo4j), `entity_extractor.py` (Sonnet), `database.py` (camera SQLite with connection pool), `analytics_db.py`, `support_case_db.py` (support cases SQLite with FTS5 and connection pool), `prompt_db.py` (LLM prompts SQLite with version history), `terminology_db.py` (RIO terminology fixes SQLite storage), `cache.py` (generic thread-safe LRU cache with TTL)

**Ingestion** (`ingestion/`): `curated_gmail.py` (7-step: Fetch→Anonymize→Sonnet→Filter→Sonnet QC→Embed→Store), `docusaurus.py` (sitemap crawler with Jina Reader + BeautifulSoup fallback), `chunker.py`, `base.py`

**Analysis** (`analysis/`): `thread_analyzer.py` (Sonnet classification), `quality_controller.py` (Sonnet QC), `camera_extractor.py`, `rio_analyzer.py` (RIO terminology context analysis)

**Agent** (`agent/`): `tools.py` (Claude Agent SDK MCP tools), `prompts.py`

**MCP** (`mcp/`): Standalone MCP server for Claude Desktop. `server.py` (FastMCP server with lifespan), `tools/` (search, cameras, documents, support). 16+ tools exposing full RAG capabilities via stdio transport.

**Graph** (`graph/`): `schema.py` (Camera, Product, Protocol, Issue, Solution entities), `enrichment.py`

**Services** (`services/`): `custom_docs.py` (CustomDocumentService CRUD), `prompt_manager.py` (LLM prompt management with caching), `default_prompts.py` (hardcoded prompt registry)

**Drafts** (`drafts/`): `gmail_service.py`, `draft_generator.py`, `draft_pipeline.py`

**Web** (`web/`): FastAPI application with modular router architecture
- `app.py` - Middleware, lifespan, app initialization
- `routers/` - API routes by domain: `cameras.py`, `pages.py`, `search.py`, `admin/` (12 routers: analytics, auth, cameras, chunks, debug, documents, drafts, graph, prompts, support, terminology)
- `auth/` - Authentication: `admin.py`, `csrf.py`, `sessions.py`
- `schemas.py` - Request/response Pydantic models
- `search/` - Search pipeline: `pipeline.py`, `synthesis.py`, `utils.py`
- `dependencies.py` - FastAPI dependency injection
- `templates/` - 30 Jinja2 templates including `/admin/docs` (10 doc pages)
- Admin UI at `/admin/{cameras,cameras-list,knowledge,analytics,drafts,chunks,graph,support-cases,prompts,terminology-fixes}`, REST APIs at `/api/`

**Models** (`models/`): `camera.py`, `custom_document.py` (10 categories), `support_case.py`

**Utils** (`utils/`): `token_encryption.py` (Fernet/PBKDF2), `anonymizer.py`, `logger.py`, `tokenizer.py` (tiktoken token counting), `text_transforms.py` (RIO product name transformations)

### Vector Collections

Three Qdrant collections (`docusaurus_docs`, `gmail_cases`, `custom_docs`) each with:
- `dense` - 1024-dim voyage-context-3 embeddings
- `sparse` - BM25 vectors

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
Settings via `clorag.config.get_settings()` (cached singleton).

## Key Patterns

### Search & Retrieval
- **Hybrid search**: Dense + sparse vectors combined via RRF across all collections
- **Reranking**: Voyage `rerank-2.5` cross-encoder refines top results (+15-40% relevance improvement)
- **Over-fetch strategy**: Retrieves 3x limit, reranks, returns top-K for optimal quality
- **Threshold after reranking**: Dynamic thresholds applied AFTER reranking (calibrated 0-1 scores). RRF scores are uncalibrated and skip threshold filtering without reranking.
- **Unified threshold logic**: Single `calculate_dynamic_threshold()` in `core/retriever.py` shared by CLI and web pipelines. 30+ technical terms (visca, sdi, hdmi, ndi, srt, ptz, etc.)
- **Dynamic thresholds**: ≤2 words: 0.15, 3-5: 0.20, >5: 0.25; technical terms +0.05; minimum 3 results
- **Source diversity**: Post-merge interleaving ensures at least 1 result from each collection with relevant hits (score ≥50% of top result)
- **Query cache**: LRU caches (200 entries) for embeddings + (100 entries) for reranking
- **Search quality logging**: Scores + source types logged per query; `/api/admin/search-quality` for low-score review
- **Contextualized embeddings**: `voyage-context-3` uses `/v1/contextualizedembeddings` API (not `/v1/embeddings`) to encode chunks with full document context
- **Synthesis grounding**: Prompt instructs "I don't know" on insufficient context; prefers docs over cases on conflicts

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
- **Dynamic prefetch**: Scales with limit (3x, max 50) for better RRF fusion quality
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

### GraphRAG
Optional Neo4j knowledge graph with entities (Camera, Product, Protocol, Port, Issue, Solution, Firmware) and relationships (COMPATIBLE_WITH, USES_PROTOCOL, AFFECTS, RESOLVED_BY, etc.). Gracefully disabled if `NEO4J_PASSWORD` not set.

**Local dev**: SSH tunnel to production Neo4j:
```bash
ssh -L 7687:localhost:7687 root@cyanview.cloud -N -f
```

### Custom Documents
10 categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, pre_sales, internal, other. Supports .txt/.md/.pdf upload, full metadata, chunked and embedded into RAG search. Sonnet-generated keywords auto-enriched alongside user-provided tags.

### Prompt Management
- **Admin-editable prompts**: 11 LLM prompts stored in SQLite, editable via `/admin/prompts` without code changes
- **Version history**: Every content change creates a new version for audit and rollback
- **Fallback to defaults**: If DB prompt not found, falls back to hardcoded defaults in `default_prompts.py`
- **Caching**: In-memory cache with TTL (default: 300s) for performance, hot reload via API
- **Variable substitution**: `{variable}` placeholders auto-detected and substituted at runtime
- **Categories**: agent, analysis, synthesis, drafts, graph, scripts
- **Configuration**: `PROMPTS_CACHE_TTL` (default: 300 seconds)
- **API usage**: `pm = get_prompt_manager(); prompt = pm.get_prompt("analysis.thread_analyzer", thread_content="...")`

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
rsync -avz --exclude '.venv' ... root@cyanview.cloud:/opt/clorag/
ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

Production: https://cyanview.cloud/ (Docker maps 8085→8080)

## Recent Updates (2026-03-16)

### v0.8.0: Retrieval Accuracy Improvements

- **Fix RRF threshold scaling** (BUG): Removed arbitrary `threshold * 0.5` pre-rerank filtering. RRF scores are uncalibrated and the old scaling caused unpredictable filtering masked by the "minimum 3 results" fallback
- **Fix filtering order** (BUG): Dynamic threshold now applied AFTER reranking (calibrated 0-1 scores), not before. Documents previously filtered out by bad RRF thresholds can now be recovered by the reranker
- **Unified threshold logic** (BUG): Web pipeline had only 9 technical terms vs CLI's 30+. Both now import shared `calculate_dynamic_threshold()` from `core/retriever.py`
- **Source diversity**: Post-merge interleaving in `hybrid_search_rrf()` ensures representation from each collection (docs, cases, custom) in top results
- **Synthesis grounding**: Added explicit "I don't know" instruction + source conflict handling (prefer docs over cases) to synthesis prompt
- **Keyword extraction**: Now samples first 2000 + last 2000 chars instead of first 4000, covering full page content
- **Smarter context truncation**: Groups merged before truncation (4K per group, 12K total budget) instead of per-chunk truncation
- **Search quality logging**: Scores + source types per result logged to analytics DB. New `/api/admin/search-quality` endpoint for reviewing low-scoring queries
- **Reranker-based confidence**: Draft generator confidence uses top-3 reranker scores instead of chunk-count heuristic
- **Thread analyzer**: Increased `max_tokens` from 1024 to 2048 for complex multi-reply threads
- **Eval scripts**: `scripts/generate_eval_dataset.py` (synthetic Q&A pairs from docs) and `scripts/eval_retrieval.py` (Recall@5, MRR, NDCG measurement)

### v0.7.0: Product Knowledge in Prompts + Unified Sonnet Model

- **Product ecosystem knowledge**: Injected Cyanview product context (RCP, RIO, CI0, VP4, NIO, RSBM, connection rules, licensing) into 4 LLM prompts: `synthesis.web_answer`, `analysis.thread_analyzer`, `analysis.quality_controller`, `drafts.email_generator`
- **Unified Sonnet model**: Replaced all Haiku usage with Sonnet (`claude-sonnet-4-6`) across the entire pipeline — analysis, keyword extraction, camera extraction, entity extraction, and RIO terminology analysis
- **Removed `haiku_model` config**: Single `sonnet_model` setting for all LLM tasks
- **Updated model IDs**: All model references use `claude-sonnet-4-6` (no date suffix)
- **CSP fix**: Added `api.fontshare.com` / `cdn.fontshare.com` to Content Security Policy
- **Thread analyzer categories**: Expanded to RCP, RIO, CI0, VP4, Network, Firmware, Configuration, Installation, REMI, Tally, Other
- **Product field**: Now supports RCP, RCP-J, RIO, RIO +WAN, RIO +LAN, CI0, CI0BM, VP4, NIO, RSBM

### Previous Updates (2026-01-23)

### Major Refactoring: Modular Web Architecture

The monolithic `app.py` (2700+ lines) has been refactored into a clean modular structure:

**New Web Structure:**

```text
web/
├── app.py              # Now ~200 lines: middleware, lifespan, app init
├── schemas.py          # Pydantic request/response models
├── dependencies.py     # FastAPI DI: limiter, templates, DB singletons
├── auth/               # Authentication module
│   ├── admin.py        # verify_admin, brute force protection, rate limits
│   ├── csrf.py         # CSRF token generation and validation
│   └── sessions.py     # Cookie-based session management
├── search/             # Search pipeline module
│   ├── pipeline.py     # Main search orchestration
│   ├── synthesis.py    # Claude LLM synthesis with streaming
│   └── utils.py        # Helpers: score thresholds, source formatting
└── routers/            # API routes by domain
    ├── cameras.py      # Public camera API (/api/cameras)
    ├── pages.py        # Page routes (/, /cameras, /help, /admin/*)
    ├── search.py       # Search API (/api/search, /api/search/stream)
    └── admin/          # 11 admin routers under /api/admin
        ├── analytics.py, auth.py, cameras.py, chunks.py
        ├── debug.py, documents.py, drafts.py, graph.py
        ├── prompts.py, support.py, terminology.py
```

### New Core Module: Generic Cache

Added `core/cache.py` - thread-safe LRU cache with optional TTL:

- `LRUCache[T]`: Generic cache with hit/miss stats
- `make_cache_key(*args)`: Hash-based key generation
- Used across embeddings, reranking, and database queries

### Security Enhancements

- **Nonce-based CSP**: All templates use `{{ csp_nonce }}` for inline scripts
- **CORS/CSRF fixes**: Proper origin validation, double-submit cookie pattern
- **OAuth encryption**: PBKDF2 iterations increased to 480K
- **Timing-safe comparison**: Prevents timing attacks on auth

### Answer Export Feature (v0.6.2)

Users can export AI responses in multiple formats: Markdown, Plain Text, HTML, PDF
