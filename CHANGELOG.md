# Changelog

All notable changes to CLORAG will be documented in this file.

## [0.6.5] - 2026-02-09

### Changed

- **RIO Terminology Prompt Improvements**
  - Updated `ANALYSIS_RIO_TERMINOLOGY` Haiku prompt with authoritative product definitions
  - **RIO** = Generic hardware reference (dimensions, ports, grounding, power, wiring, mounting, weight)
  - **RIO +WAN** = Full license (LAN & WAN, Cyanview cloud, REMI mode, Internet, 1-128 cameras)
  - **RIO +LAN** = Formerly "RIO-Live" (LAN only, 1-2 camera companion, no WAN/cloud/REMI)
  - Added "RIOLive" to legacy terms list (alongside "RIO-Live", "RIO Live", "RIO +WAN Live")
  - More precise context rules: cloud/REMI/Internet/remote/multi-camera(>2) → +WAN, LAN-only/local/1-2 cameras → +LAN

- **Query-Time RIO Term Normalization**
  - Legacy terms in search queries are auto-normalized before embedding via `apply_product_name_transforms()`
  - Injected in `MultiSourceRetriever.retrieve()` — single point covers web, MCP, and CLI
  - "RIO-Live" / "RIO Live" / "RIOLive" → "RIO +LAN" in queries
  - "RIO +WAN Live" → "RIO +LAN" in queries
  - Normalized query also used for reranking

### Files

- `src/clorag/services/default_prompts.py` - Updated RIO terminology Haiku prompt
- `src/clorag/core/retriever.py` - Added query normalization before embedding
- `CLAUDE.md` - Updated RIO terminology section with accurate definitions
- `README.md` - Added query normalization to pipeline diagram

## [0.6.3] - 2026-01-23

### Added

- **CSP-Compliant Event Delegation System**
  - New `AdminActions` object in `admin.js` for global event delegation
  - Supports `data-action` attributes instead of inline `onclick` handlers
  - Built-in handlers: `toggle-section`, `close-modal`, `open-modal`, `add-row`, `remove-row`, `confirm-delete`, `call`
  - Custom handler registration via `AdminActions.register('action-name', handler)`
  - Automatic modal close on overlay click and Escape key

- **Generic LRU Cache Module**
  - New `src/clorag/core/cache.py` with thread-safe `LRUCache[T]` class
  - Optional TTL-based expiration for cached items
  - Hit/miss statistics tracking with `stats()` method
  - `make_cache_key(*args)` helper for hash-based key generation
  - Designed for embeddings, reranking, and database query caching

- **Modular Web Architecture**
  - Refactored monolithic `app.py` (2700+ lines) into clean modular structure
  - New `web/auth/` module: `admin.py`, `csrf.py`, `sessions.py`
  - New `web/search/` module: `pipeline.py`, `synthesis.py`, `utils.py`
  - New `web/routers/` with 11 admin routers + 3 public routers
  - New `web/schemas.py` for Pydantic request/response models
  - New `web/dependencies.py` for FastAPI dependency injection
  - `app.py` now ~200 lines: middleware, lifespan, app initialization

### Changed

- **Differentiated CSP Policy**
  - Admin pages: Allow inline handlers via `unsafe-inline` (behind authentication)
  - Public pages: Strict nonce-only policy for maximum security
  - Provides migration path while templates are updated to data-action pattern

- **Template Migration to Event Delegation**
  - `index.html` fully migrated to data-action pattern (strict CSP)
  - `admin_analytics.html` partially migrated with AdminActions handlers
  - Dynamic HTML now uses `data-action`, `data-exchange`, `data-format` attributes

### Security

- Nonce-based CSP maintained for public-facing pages
- Admin pages use relaxed CSP (behind authentication) until template migration complete
- Modal overlay click and Escape key handlers now global (reduce code duplication)

### Files

- `src/clorag/core/cache.py` - New generic LRU cache module
- `src/clorag/web/app.py` - CSP differentiation, reduced to ~200 lines
- `src/clorag/web/static/js/admin.js` - AdminActions event delegation system
- `src/clorag/web/templates/index.html` - Migrated to data-action pattern
- `src/clorag/web/templates/admin_analytics.html` - Partial migration
- `src/clorag/web/auth/` - New authentication module (3 files)
- `src/clorag/web/search/` - New search pipeline module (3 files)
- `src/clorag/web/routers/` - New router modules (14 files)
- `src/clorag/web/schemas.py` - Request/response Pydantic models
- `src/clorag/web/dependencies.py` - FastAPI dependency injection

## [0.6.2] - 2026-01-22

### Added
- **Answer Export Feature**
  - Export AI answers in multiple formats: Text (.txt), Markdown (.md), HTML with diagrams
  - HTML export options: SVG (vector quality) or PNG (raster, universal compatibility)
  - Export dropdown menu in each answer card header
  - Mermaid diagrams converted to embedded base64 images in HTML exports
  - PNG export uses 2x scaling for retina/HiDPI display quality
  - Mobile-friendly dropdown with tap-to-toggle support
  - Filenames: `cyanview-answer-YYYY-MM-DD-HH-MM-SS.{txt|md|html}`

### Technical Details
- Client-side JavaScript implementation (no server changes required)
- `answerStore` object stores raw markdown during streaming for accurate markdown export
- SVG to PNG conversion via Canvas API with `XMLSerializer`
- Blob-based file download with automatic memory cleanup via `URL.revokeObjectURL()`

### Files
- `src/clorag/web/templates/index.html` - Export dropdown UI, CSS, and JavaScript functions

## [0.5.8] - 2026-01-21

### Added
- **Jina Reader Integration**
  - Primary web scraper using `r.jina.ai` API for clean markdown extraction
  - Automatic retry logic (3 attempts) with exponential backoff on 429/503 errors
  - BeautifulSoup fallback when Jina Reader is unavailable
  - Statistics logging: success/fallback/total counts per ingestion run

- **HTML Table Preservation**
  - BeautifulSoup fallback now converts HTML tables to markdown format
  - Tables preserved as atomic units during chunking (not split across chunks)
  - Supports complex tables with headers, multiple rows, and varied column counts
  - Camera control matrices and feature comparison tables now searchable

### Changed
- Docusaurus ingestion now uses Jina Reader as primary content extractor
- Improved content quality for pages with structured data (tables, code blocks)

### Files
- `src/clorag/ingestion/docusaurus.py` - Added Jina Reader integration and table-to-markdown conversion

## [0.5.7] - 2026-01-20

### Added
- **Token-Aware Chunking System**
  - Token-based chunk sizing using `tiktoken` (cl100k_base encoding) for 15-20% more consistent chunks
  - Configurable via environment variables: `CHUNK_USE_TOKENS`, `CHUNK_SIZE_DOCS`, `CHUNK_SIZE_CASES`, etc.
  - Content-type specific sizing: Documentation (450 tokens), Support cases (350 tokens), Default (400 tokens)
  - `SemanticChunker.from_settings()` factory method for config-driven chunker creation
  - New `src/clorag/utils/tokenizer.py` module with `count_tokens()`, `truncate_to_tokens()` functions
  - Backward compatible: Set `CHUNK_USE_TOKENS=false` for character-based mode
  - Hierarchical chunking design document at `docs/architecture/hierarchical-chunking.md`

- **RIO Terminology Fix System**
  - CLI tool `uv run fix-rio-terminology` for scanning and fixing RIO product terminology
  - Context-aware Haiku analysis distinguishes license vs hardware contexts
  - Admin UI at `/admin/terminology-fixes` for reviewing and approving fixes
  - Batch approve/reject with diff view showing original → suggested changes
  - Human-in-the-loop for ambiguous cases (`needs_human_review` type)
  - Direct "Edit Chunk" links to chunk editor for manual corrections

- **RIO Product Terminology Convention**
  - **RIO +WAN**: Full-featured RIO, works via LAN and WAN, for 1-128 distant cameras (REMI toolbox)
  - **RIO +LAN**: Local version, LAN only, designed as companion for 1 camera
  - **RIO**: Generic reference to hardware (when license isn't relevant)
  - Legacy terms: "RIO-Live", "RIO Live", "RIO +WAN Live" → "RIO +LAN" (in license context)
  - Hardware context (grounding, power, wiring) → generic "RIO" (license not relevant)

- **New CLI Commands**
  - `uv run fix-rio-terminology --preview` - Scan chunks and save suggestions
  - `uv run fix-rio-terminology --apply` - Apply all approved fixes
  - `uv run fix-rio-terminology --stats` - Show fix statistics
  - `uv run fix-rio-terminology --export FILE` - Export fixes to JSON
  - `uv run fix-rio-terminology --import FILE` - Import fixes from JSON

- **New API Endpoints**
  - `GET /api/admin/terminology-fixes` - List fixes with filters
  - `GET /api/admin/terminology-fixes/stats` - Statistics by status/type/collection
  - `PUT /api/admin/terminology-fixes/{id}/status` - Update fix status
  - `PUT /api/admin/terminology-fixes/batch-status` - Batch status update
  - `POST /api/admin/terminology-fixes/apply` - Apply all approved fixes
  - `DELETE /api/admin/terminology-fixes/{id}` - Delete a fix

- **Metadata Updates on Apply**
  - Fixes also update `subject`, `problem_summary`, `solution_summary` fields
  - Keywords cleaned: legacy terms removed, new terminology added
  - Automatic re-embedding after text changes

### New Files
- `src/clorag/utils/tokenizer.py` - Token counting utilities using tiktoken
- `src/clorag/core/terminology_db.py` - SQLite storage for terminology fixes
- `src/clorag/analysis/rio_analyzer.py` - Haiku-based context analyzer
- `src/clorag/scripts/fix_rio_terminology.py` - CLI script
- `src/clorag/web/templates/admin_terminology_fixes.html` - Admin UI
- `docs/architecture/hierarchical-chunking.md` - Future chunking design doc
- `tests/test_chunker.py` - 20 unit tests for token-aware chunking

### New Configuration Options
- `CHUNK_USE_TOKENS` (default: `true`) - Use token-based chunking
- `CHUNK_SIZE_DOCS` (default: `450`) - Chunk size for documentation (tokens)
- `CHUNK_SIZE_CASES` (default: `350`) - Chunk size for support cases (tokens)
- `CHUNK_SIZE_DEFAULT` (default: `400`) - Default chunk size (tokens)
- `CHUNK_OVERLAP` (default: `50`) - Chunk overlap (~12.5%)
- `CHUNK_ADAPTIVE_THRESHOLD` (default: `200`) - Single-chunk threshold (tokens)

### Dependencies
- Added `tiktoken>=0.12.0` for token counting

### Changed
- Admin dashboard includes Terminology Fixes card
- Unified admin.css styles for diff view, approve/reject buttons

## [0.5.6] - 2026-01-19

### Added
- **Voyage AI Reranking Integration**
  - `RerankerClient` class using Voyage AI `rerank-2.5` cross-encoder model
  - 15-40% improvement in retrieval relevance for complex queries
  - Over-fetch strategy: retrieves 3x limit candidates, reranks, returns top-K
  - Thread-safe LRU cache (100 entries) for reranking results
  - Automatic retry with exponential backoff

- **New Configuration Options**
  - `RERANK_ENABLED` (default: `true`) - Enable/disable reranking globally
  - `VOYAGE_RERANK_MODEL` (default: `rerank-2.5`) - Reranker model selection
  - `RERANK_TOP_K` (default: `5`) - Number of results after reranking

- **Per-Query Reranking Control**
  - `use_reranking` parameter in `MultiSourceRetriever.retrieve()` to override global setting
  - Useful for disabling reranking on simple queries where speed is prioritized

- **Performance Monitoring System** (Phase 3)
  - `MetricsCollector` class with thread-safe sliding window (1000 entries)
  - Timing measurements with percentile calculations (p50, p90, p95, p99)
  - Query counter and error counter with error rate calculation
  - `measure()` context manager with automatic slow operation logging
  - Convenience functions: `measure_embedding_generation()`, `measure_vector_search()`, `measure_total_search()`, `measure_llm_synthesis()`

- **Performance Optimizations** (Phases 1-2)
  - Parallel embedding generation: Dense (Voyage AI) and sparse (BM25) embeddings generated concurrently
  - Sparse model preloading: BM25 model loaded at startup, not per-request
  - Async Voyage AI client: Migrated from sync to async API for non-blocking I/O
  - Dynamic prefetch scaling: Qdrant prefetch factor adjusts based on query complexity
  - SupportCaseDatabase connection pooling: 5-connection pool with WAL mode

- **Cache Statistics Endpoint**
  - `GET /api/admin/cache-stats` - View embedding cache hit/miss rates

- **New API Endpoints**
  - `GET /api/admin/metrics` - Performance metrics with thresholds and alerts
  - `GET /api/admin/metrics/recent/{metric_name}` - Recent measurements for debugging

- **New Tests**
  - 20 comprehensive tests for the metrics module (test_metrics.py)

### Changed
- `MultiSourceRetriever` now applies reranking after RRF fusion by default
- Search result scores now reflect reranker relevance scores (0-1 range)
- Agent tools include reranking status and cache stats in output
- Architecture diagram updated to show reranking stage

### Files
- `src/clorag/core/reranker.py` - NEW: Voyage AI reranker client with caching
- `src/clorag/core/retriever.py` - Added reranking integration
- `src/clorag/core/__init__.py` - Export `RerankerClient`
- `src/clorag/config.py` - Added reranking configuration options
- `src/clorag/agent/tools.py` - Updated output with rerank status
- `src/clorag/core/metrics.py` - NEW: Performance metrics collection module
- `src/clorag/web/app.py` - Added metrics instrumentation and API endpoints
- `tests/test_metrics.py` - NEW: Comprehensive metrics test suite

### Performance
- Reranking adds ~100-500ms latency for typical result sets (15 documents)
- First 200M tokens free with Voyage AI, then pay-per-token
- Cache hit rate typically 30-50% for repeated queries

### Documentation
- Admin docs: Added Performance Monitoring section with metrics table, optimizations, and API reference
- CLAUDE.md: Updated Core section and added Performance Monitoring documentation

## [0.5.5] - 2026-01-15

### Added
- **Support Cases SQLite Database**
  - `SupportCaseDatabase` class for storing complete support case documents
  - FTS5 full-text search with BM25 ranking across subject, problem, solution, document, keywords
  - Thread-safe database with WAL mode and connection pooling
  - Automatic FTS sync via SQLite triggers

- **Support Cases Admin UI** (`/admin/support-cases`)
  - Statistics grid showing total cases, breakdown by category/product/quality
  - Full-text search with category and product filters
  - Paginated table with subject, problem summary, category, product, quality stars, date
  - Detail modal with 3 tabs: Summary, Full Document, Raw Thread
  - Delete functionality

- **Thread Content Cleaning** (`clean_thread_quotes()`)
  - Removes quoted reply text (lines starting with `>`)
  - Removes reply headers ("On [date], [person] wrote:" in EN/FR/DE)
  - Removes forwarded message separators
  - Removes email signatures ("Best regards", "Sent from my iPhone", etc.)
  - Multi-message thread support with state reset at message boundaries

- **New API Endpoints**
  - `GET /api/admin/support-cases` - List cases with pagination and filters
  - `GET /api/admin/support-cases/stats` - Statistics by category/product/quality
  - `GET /api/admin/support-cases/search?q=` - FTS5 search
  - `GET /api/admin/support-cases/{id}` - Get case details
  - `GET /api/admin/support-cases/{id}/raw-thread` - Get cleaned raw thread
  - `DELETE /api/admin/support-cases/{id}` - Delete case

### Changed
- Gmail ingestion now stores full cases in SQLite alongside Qdrant chunks
- Raw thread content cleaned before storage (quotes and signatures removed)
- Admin dashboard includes Support Cases card

### Files
- `src/clorag/core/support_case_db.py` - NEW: SQLite database for support cases
- `src/clorag/utils/anonymizer.py` - Added `clean_thread_quotes()` function
- `src/clorag/ingestion/curated_gmail.py` - Added SQLite storage step
- `src/clorag/web/app.py` - Added support cases API endpoints
- `src/clorag/web/templates/admin_support_cases.html` - NEW: Admin UI
- `src/clorag/web/templates/admin_index.html` - Added Support Cases card

## [0.5.4] - 2026-01-15

### Added
- **Camera Database Performance (Phase 3)**
  - FTS5 full-text search with BM25 ranking and Porter stemming for fast camera search
  - Thread-safe TTL cache (100 entries, 5-minute TTL) for list/search/stats queries
  - SQLite connection pool (5 connections) with WAL mode and 64MB cache
  - Covering index for optimized list query patterns
  - Automatic FTS index corruption recovery

- **Camera Features (Phase 4)**
  - Camera comparison: Select up to 5 cameras for side-by-side spec comparison
  - Comparison modal with highlighted common values (green tags)
  - Related cameras API: Find similar cameras based on manufacturer, device type, ports, protocols
  - CSV export: Download all cameras or filtered subset as CSV
  - CSV import: Bulk upload cameras via CSV file (admin only, upsert on name+manufacturer)

- **New CLI Command**
  - `uv run rebuild-fts` - Rebuild camera FTS5 search index
  - `uv run rebuild-fts --check` - Check FTS index status without rebuilding

- **New API Endpoints**
  - `GET /api/cameras/{id}/related` - Get similar cameras
  - `POST /api/cameras/compare` - Compare multiple cameras (max 5)
  - `GET /api/cameras/export.csv` - Export cameras as CSV
  - `POST /api/admin/cameras/import` - Import cameras from CSV (admin)

- **New Database Methods**
  - `CameraDatabase.get_cameras_by_ids()` - Get multiple cameras by ID list
  - `CameraDatabase.find_related_cameras()` - Similarity-based camera search
  - `CameraDatabase.rebuild_fts_index()` - Rebuild FTS5 index with corruption recovery

### Changed
- Camera search now uses FTS5 with BM25 ranking (fallback to LIKE for unsupported queries)
- Admin camera list now uses server-side pagination (50 cameras per page)
- Camera list/search/stats queries are cached with automatic invalidation on writes

### Files
- `src/clorag/core/database.py` - Added TTLCache, ConnectionPool, FTS5, new query methods
- `src/clorag/scripts/rebuild_fts.py` - New CLI script for FTS index maintenance
- `src/clorag/web/app.py` - New camera API endpoints
- `src/clorag/web/templates/cameras.html` - Comparison UI with checkboxes and modal

## [0.5.3] - 2026-01-14

### Fixed
- **MultiSourceRetriever Hybrid Search** - Retriever now properly uses hybrid RRF search with both dense (Voyage AI) and sparse (BM25) vectors instead of dense-only search
- **Agent MCP Tools** - All search tools (`search_docs`, `search_cases`, `search_custom`, `hybrid_search`) now use the hybrid retriever with proper sparse embedding support

### Added
- **New `search_custom` Tool** - Agent can now search custom knowledge documents added by administrators
- **Technical Terms Detection** - 30+ domain-specific terms (firmware, protocol, rcp, rio, visca, etc.) for improved threshold calculation
- **Cache Statistics API** - `hybrid_search` tool now returns embedding cache hit rates for debugging

### Changed
- Agent MCP server version bumped to 1.1.0
- Dynamic score thresholds now properly applied with RRF-scaled values (threshold * 0.5 for RRF score range)

## [0.5.2] - 2026-01-12

### Added
- **Knowledge Graph Relationship Management**
  - Edit relationships: Change relationship type between entities via admin UI
  - Delete relationships: Remove unwanted relationships from the knowledge graph
  - Confirmation modals for safe delete/edit operations
  - Automatic refresh of relationship list after modifications

- **New API Endpoints**
  - `DELETE /api/admin/graph/relationships` - Delete a relationship
  - `PATCH /api/admin/graph/relationships` - Update a relationship type

- **GraphStore Methods**
  - `delete_relationship()` - Delete relationship by source/target/type
  - `update_relationship_type()` - Change relationship type (preserves properties)

- **Product Name Transformations**
  - Shared text transform utility (`utils/text_transforms.py`)
  - RIO → RIO +WAN, RIO-Live → RIO +LAN transformations during ingestion
  - Applied to both docusaurus_docs and gmail_cases collections

### Changed
- Knowledge Graph Explorer (`/admin/graph`) Relationships tab now includes Edit/Delete action buttons
- Gmail ingestion supports `--fresh` flag to delete and re-create collection

### Fixed
- Token encryption now uses `shutil.move()` instead of `pathlib.rename()` to fix Docker volume mount issues

## [0.5.1] - 2026-01-05

### Fixed
- **SecretStr handling in populate_graph** - Properly extract Neo4j password from Pydantic SecretStr
- **Dict access in scroll_chunks return type** - Fixed attribute access for scroll_chunks response handling

### Documentation
- Updated README with detailed GraphRAG workflow integration diagram
- Updated CLAUDE.md with Neo4j configuration and SSH tunnel instructions

## [0.5.0] - 2025-12-26

### Added
- **GraphRAG (Knowledge Graph Augmented Retrieval)**
  - Neo4j knowledge graph integration for entity-based context enrichment
  - Entity extraction using Claude Haiku from Qdrant chunks
  - 8 node types: Camera, Product, Protocol, Port, Control, Issue, Solution, Firmware
  - 11 relationship types: COMPATIBLE_WITH, USES_PROTOCOL, HAS_PORT, AFFECTS, etc.
  - Graph context automatically added to Claude synthesis when Neo4j is configured
  - Graceful degradation when Neo4j is unavailable

- **New Files**
  - `src/clorag/core/graph_store.py` - Neo4j async client wrapper
  - `src/clorag/core/entity_extractor.py` - LLM-based entity extraction
  - `src/clorag/graph/schema.py` - Pydantic models for graph entities
  - `src/clorag/graph/enrichment.py` - Graph context enrichment service
  - `src/clorag/scripts/populate_graph.py` - CLI command for graph population

- **New CLI Command**
  - `uv run populate-graph` - Extract entities from Qdrant chunks and populate Neo4j

- **New API Endpoint**
  - `GET /api/admin/graph/stats` - Get knowledge graph statistics

### Changed
- Search synthesis now includes graph relationship context when available
- Docker Compose includes Neo4j service (neo4j:5-community)

### Dependencies
- Added `neo4j>=6.0.0` for Neo4j async Python driver

### Configuration
- `NEO4J_URI` - Neo4j Bolt protocol URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (optional, disables GraphRAG if not set)
- `NEO4J_DATABASE` - Neo4j database name (default: `neo4j`)

## [0.4.0] - 2025-12-22

### Added
- **Query Embedding Cache**
  - Thread-safe LRU cache for both dense (Voyage AI) and sparse (BM25) query embeddings
  - Cache size: 200 unique queries per embedding type
  - Reduces API costs and latency for repeated queries
  - Hit rate statistics for monitoring (`cache.stats()`)

- **Dynamic Score Thresholds**
  - Adaptive filtering based on query characteristics
  - Short queries (≤2 words): permissive threshold (0.15)
  - Medium queries (3-5 words): moderate threshold (0.20)
  - Long/specific queries (>5 words): strict threshold (0.25)
  - Technical terms (rio, rcp, firmware, etc.) increase threshold by 0.05
  - Always returns minimum 3 results regardless of threshold

- **Paginated Document Listing**
  - Knowledge base API now supports pagination with offset parameter
  - Returns total count for pagination UI
  - Incremental scrolling (batches of 100) instead of loading all chunks
  - Early exit optimization when enough documents collected

### Changed
- Knowledge base list API response now includes `total`, `limit`, `offset` fields
- Improved error handling for document upload with detailed logging

### Documentation
- Updated admin docs with Performance Optimizations section
- Updated CLAUDE.md with new caching and threshold patterns
- Updated README.md features list

## [0.3.9] - 2025-12-11

### Added
- **File Upload for Custom Knowledge Documents**
  - Upload .txt, .md, and .pdf files directly to the knowledge base
  - Drag-and-drop file zone with visual feedback
  - PDF text extraction using pypdf library
  - Auto-suggests document title from filename
  - Toggle between "Paste Text" and "Upload File" input modes
  - New API endpoint: `POST /api/admin/knowledge/upload`

### Dependencies
- Added `pypdf>=5.0.0` for PDF text extraction

## [0.3.8] - 2025-12-11

### Security
- **Brute Force Protection**
  - Track failed login attempts per IP address
  - 5 failed attempts triggers 5-minute lockout
  - Logging for failed attempts and lockout events
  - `LoginAttemptTracker` class in `app.py`

- **XSS Protection with DOMPurify**
  - Added DOMPurify CDN for HTML sanitization
  - Custom `safeMarkdown()` function with SVG allowlist for Mermaid diagrams
  - Prevents script injection through markdown rendering

- **OAuth Token Encryption at Rest**
  - New `utils/token_encryption.py` module
  - Fernet symmetric encryption with PBKDF2 key derivation (480K iterations)
  - Admin password used as encryption key source
  - Backward compatible: reads unencrypted tokens, encrypts on save
  - Atomic file writes with restrictive permissions (chmod 600)

- **Secure Cookie Configuration**
  - New `SECURE_COOKIES` environment variable (default: `true`)
  - Set to `false` for local development without HTTPS
  - Applied to session cookies in login/logout endpoints

- **Removed localStorage Password Storage**
  - Admin pages now use session-based authentication exclusively
  - Removed client-side password storage from camera management pages
  - Added `checkAuth()` function with redirect to login

### Added
- **Admin Documentation Enhancements**
  - New API Connection Guide section (`/admin/docs#api-connection`)
  - New Security Information section (`/admin/docs#security`)
  - Security architecture diagram with session flow
  - Complete security checklist for deployments

### New Files
- `src/clorag/utils/token_encryption.py` - Encrypted OAuth token storage utilities

### Dependencies
- Added `cryptography>=44.0.0` for Fernet encryption

## [0.3.7] - 2025-12-10

### Added
- **Mermaid Diagram Generation**
  - Claude now autonomously generates Mermaid.js diagrams when explaining camera connections, network topology, or signal flows
  - Updated `SYNTHESIS_SYSTEM_PROMPT` with diagram instructions for integration scenarios
  - Added Mermaid.js v11 (ESM module) with Cyanview color theming
  - Diagrams render automatically after streaming completes using `renderMermaidDiagrams()` async function

- **Custom Knowledge Base**
  - New `custom_docs` Qdrant collection for manually added knowledge documents
  - Full metadata support: title, tags, category, URL reference, expiration date, notes
  - 9 document categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, internal, other
  - Admin UI at `/admin/knowledge` with Add/Browse tabs, edit modal, and filtering
  - Custom documents are chunked, embedded, and included in hybrid RAG search alongside docs and support cases

### New Files
- `src/clorag/models/custom_document.py` - Pydantic models for custom documents
- `src/clorag/services/custom_docs.py` - `CustomDocumentService` with CRUD and embedding
- `src/clorag/web/templates/admin_knowledge.html` - Admin UI for knowledge management

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/knowledge` | List custom documents |
| GET | `/api/admin/knowledge/categories` | Get available categories |
| GET | `/api/admin/knowledge/{id}` | Get document by ID |
| POST | `/api/admin/knowledge` | Create new document |
| PUT | `/api/admin/knowledge/{id}` | Update document |
| DELETE | `/api/admin/knowledge/{id}` | Delete document |

### Changed
- `VectorStore.hybrid_search_rrf()` now searches all 3 collections (docs, cases, custom_docs)
- `_build_context()` handles custom_docs source type for Claude synthesis
- Added `ChunkCollection.CUSTOM` enum value for chunk editor
- Admin dashboard now includes Knowledge Base card

## [0.3.6] - 2025-12-10

### Changed
- **Upgraded LLM to Claude Sonnet 4.5** for RAG search synthesis and draft generation
  - Improved answer quality and accuracy for support queries
  - Better contextual understanding and response generation
  - Applies to: search API, streaming responses, search debug, and draft auto-replies

## [0.3.5] - 2025-12-09

### Added
- **Chunk Editor** - Admin UI for browsing, searching, editing, and deleting individual chunks in Qdrant
  - Browse chunks by collection (docs/cases) with pagination
  - Text search using hybrid search (semantic + keyword)
  - Edit chunk text and metadata (title/subject) with automatic re-embedding
  - Delete individual chunks from collections
  - Warning displayed when text changes will trigger re-embedding cost
  - Read-only display of technical fields (ID, URL, chunk_index, etc.)

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/chunks` | List/search chunks (paginated) |
| GET | `/api/admin/chunks/{collection}/{id}` | Get chunk details |
| PUT | `/api/admin/chunks/{collection}/{id}` | Update chunk (re-embeds if text changed) |
| DELETE | `/api/admin/chunks/{collection}/{id}` | Delete chunk |

### Changed
- Added VectorStore CRUD methods: `get_chunk`, `scroll_chunks`, `update_chunk`, `delete_chunk`
- New templates: `admin_chunks.html` (browser), `admin_chunk_edit.html` (editor)
- Admin dashboard now includes Chunk Editor card

## [0.3.4] - 2025-12-09

### Added
- **Draft Auto-Reply System**
  - New `drafts/` module for automated draft creation in Gmail
  - `GmailDraftService` with `gmail.compose` scope for creating draft replies
  - `DraftResponseGenerator` uses RAG search + Claude to generate contextual responses
  - `DraftCreationPipeline` orchestrates: fetch pending → generate response → create draft
  - Admin UI at `/admin/drafts` with pending threads list, preview, and draft creation
  - CLI command `uv run draft-support` for manual/batch draft processing
  - Toggle-based thread view with expandable message history timeline
  - Direct Gmail links to open threads in a new tab
  - Session storage caching (5-min TTL) to avoid reloading data on page revisits

- **Google Groups Support**
  - Parse `X-Original-Sender` and `Reply-To` headers to show actual customer email
  - Handle "Name via Group" format to display clean sender names
  - Correctly identify Cyanview vs customer messages in thread timeline

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/drafts/status` | Draft system status |
| GET | `/api/admin/drafts/pending` | List unanswered threads |
| GET | `/api/admin/drafts/thread/{id}` | Get thread with all messages |
| POST | `/api/admin/drafts/preview/{id}` | Preview AI-generated draft |
| POST | `/api/admin/drafts/create/{id}` | Create draft in Gmail |
| POST | `/api/admin/drafts/run` | Run draft pipeline |

## [0.3.3] - 2025-12-05

### Added
- **Follow-up Conversations**
  - Server-side session storage with LRU cache (max 1000 sessions, 30-min TTL)
  - Conversation context includes last 3 Q&A exchanges for Claude synthesis
  - Session ID passed via `session_id` field in search requests/responses
  - Inline follow-up input appears below each AI answer for natural conversation flow
  - "New conversation" floating button to reset session and start fresh

- **Data Ingestion Documentation** (`/admin/docs#data-ingestion`)
  - Complete Docusaurus pipeline flow diagram (sitemap → fetch → extract → chunk → embed → store)
  - Complete Gmail curated pipeline flow diagram (8-step process with LLM analysis)
  - Hybrid embedding flow diagram showing dense + sparse vector generation
  - Stage-by-stage tables with models, tools, and descriptions
  - Incremental ingestion examples with `--offset` parameter

### Changed
- Chat-style UI with user question bubbles and AI answer cards
- Search results now stack vertically as conversation exchanges
- Focus automatically moves to follow-up input after answer completes

## [0.3.2] - 2025-12-02

### Added
- **Technical Documentation Page** (`/admin/docs`)
  - Comprehensive architecture overview
  - API reference with all endpoints documented
  - Authentication and security documentation
  - Deployment and configuration guide
  - Sidebar navigation for easy browsing

- **User Guide Page** (`/help`)
  - Getting started instructions
  - Example queries with click-to-search
  - Source filter explanations
  - Tips for better search results
  - Feature overview

### Security
- Fixed XSS vulnerability in camera detail modal (HTML escaping)
- Fixed XSS vulnerability in analytics chunk URLs (URL validation)
- Fixed open redirect vulnerability in login page

## [0.3.1] - 2025-12-01

### Added
- **Session-Based Admin Authentication**
  - Secure login page at `/admin/login` replacing header-based auth
  - Signed session cookies using `itsdangerous` (24-hour expiry)
  - Admin dashboard at `/admin` with links to all admin features
  - Logout functionality at `/admin/logout`

- **Camera Detail Modal**
  - Click any camera row to view full details in modal
  - Shows all controls, notes, ports, protocols (not truncated)
  - Keyboard support (Escape to close)

- **Search Analytics Enhancements**
  - Store full LLM response and retrieved chunks with each search
  - Click any search in history to view stored results (no re-query)
  - New API endpoint `GET /api/admin/search/{id}` for search details

- **Improved Help Popup**
  - Practical "How to Use" guide instead of technical explanation
  - 5 clickable example queries that auto-execute search
  - Tips for better results and source filter explanation

- **Documentation Date Tracking**
  - Extract `<lastmod>` from sitemap during docs ingestion
  - Stored in metadata alongside URL and title
  - Enables future recency-based ranking

### Changed
- Admin routes now require session cookie instead of `X-Admin-Password` header
- Analytics database schema adds `response` and `chunks` columns

## [0.3.0] - 2025-12-01

### Added
- **Search Analytics System**
  - Separate analytics database (`data/analytics.db`) for search tracking
  - `AnalyticsDatabase` class with query logging and statistics
  - Admin dashboard at `/admin/analytics` with password protection
  - Stats: total searches, avg response time, searches by source, daily chart
  - Popular queries and recent searches tables

- **UI Improvements**
  - Export CSV button on cameras page (client-side generation)
  - Copy response to clipboard button on search results
  - Type column moved to first position with icon-only display (tooltip on hover)
  - Device type icons: camera_cinema, camera_broadcast, camera_ptz, camera_mirrorless, camera_action

- **SearXNG Local Integration**
  - Optional local SearXNG instance support via Docker network
  - `SEARXNG_URL` environment variable (defaults to external search.sapti.me)
  - Secured port 8888 to localhost-only binding

### Changed
- **Protocol/Port Normalization**
  - Unified Canon XC variants (`XC`, `XC protocol`) → `Canon XC`
  - Unified Remote-A variants → `Remote-A`
  - Unified 8-pin variants → `8-pin Remote`
  - Unified Ethernet variants (`RJ45`, `LAN`) → `Ethernet`
  - Unified Wi-Fi variants → `Wi-Fi`
  - Split `Pelco RS-485` → Protocol: `Pelco`, Port: `RS-485`
  - LANC port → `Mini-jack 2.5mm` (LANC is protocol, 2.5mm is connector)
  - Moved USB/Wi-Fi from protocols to ports where appropriate

### Fixed
- SearXNG public exposure on port 8888 (now localhost only)

## [0.2.0] - 2025-11-28

### Changed
- **Upgraded to Claude Haiku 4.5** (`claude-haiku-4-5-20251001`)
  - Thread analyzer now uses Haiku 4.5 for faster, more accurate analysis
  - Camera extractor uses Haiku 4.5 for improved extraction quality

### Added
- **Incremental Gmail Ingestion** (`--offset` parameter)
  - `uv run ingest-curated --offset 300 --max-threads 300` to process threads 301-600
  - Enables batch processing without re-analyzing already ingested threads
  - Gmail API returns threads sorted by most recent first

#### Camera Compatibility Database System
- **Camera Models** (`src/clorag/models/camera.py`)
  - `Camera` Pydantic model with structured fields: name, manufacturer, ports, protocols, supported_controls, notes, doc_url, manufacturer_url
  - `CameraSource` enum for tracking data origin (documentation, support_case, manual)
  - `CameraCreate`, `CameraUpdate`, `CameraEnrichment` models for CRUD operations

- **SQLite Database Layer** (`src/clorag/core/database.py`)
  - `CameraDatabase` class with full CRUD operations
  - JSON columns for storing array fields (ports, protocols, controls, notes)
  - Upsert pattern for merging camera data from multiple sources
  - `get_camera_database()` singleton factory function
  - Search by name, manufacturer filtering, statistics queries

- **Camera Extractor** (`src/clorag/analysis/camera_extractor.py`)
  - LLM-based extraction using Claude Haiku 4.5 (claude-haiku-4-5-20251001)
  - `extract_cameras()` - Parse documentation/support cases for camera info
  - `extract_from_batch()` - Batch extraction with concurrency control
  - `enrich_from_manufacturer()` - Scrape manufacturer websites for additional specs
  - `search_manufacturer_url()` - Find official product pages

- **Ingestion Integration**
  - `docusaurus.py` - Added `extract_cameras` parameter, extracts camera data during doc ingestion
  - `curated_gmail.py` - Added `extract_cameras` parameter, extracts camera data from support cases

- **Web Interface**
  - Public camera list page at `/cameras` with search and manufacturer filter
  - Admin interface at `/admin/cameras` with full CRUD operations
  - Camera edit form at `/admin/cameras/{id}/edit` and `/admin/cameras/new`
  - REST API endpoints:
    - `GET /api/cameras` - List all cameras
    - `GET /api/cameras/search?q=...` - Search cameras
    - `GET /api/cameras/{id}` - Get single camera
    - `POST /api/admin/cameras` - Create camera (admin)
    - `PUT /api/admin/cameras/{id}` - Update camera (admin)
    - `DELETE /api/admin/cameras/{id}` - Delete camera (admin)
  - Admin authentication via `X-Admin-Password` header

- **Configuration**
  - `database_path` setting for SQLite location (default: `data/clorag.db`)
  - `admin_password` setting for admin interface access

- **Templates**
  - `cameras.html` - Public camera compatibility list
  - `admin_cameras.html` - Admin camera management page
  - `camera_edit.html` - Camera create/edit form
  - Updated `index.html` with camera links in navbar and quick links

## [0.1.0] - Initial Release

### Added
- Multi-RAG agent combining Docusaurus documentation and Gmail support cases
- Hybrid search with dense (voyage-context-3) and sparse (BM25) vectors
- RRF (Reciprocal Rank Fusion) for optimal retrieval
- Contextualized embeddings for improved document retrieval
- Claude Haiku for fast thread analysis and classification
- Claude Sonnet for quality control refinement
- Curated Gmail pipeline with anonymization
- FastAPI web interface with streaming responses
- CLI agent for interactive queries
- Docker deployment support
