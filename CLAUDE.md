# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLORAG is a Multi-RAG (Retrieval-Augmented Generation) agent for Cyanview support that combines:
- **Docusaurus documentation** from the support site
- **Gmail support threads** (curated and anonymized)
- **Custom knowledge documents** (manually added via admin UI)

Uses hybrid search (dense + sparse vectors with RRF fusion) for optimal retrieval of technical product information. Answer synthesis uses Claude Sonnet 4.5 for high-quality responses with automatic Mermaid diagram generation for integration scenarios.

## Development Commands

```bash
# Install dependencies
uv sync

# Run web server (FastAPI on port 8080)
uv run rag-web

# Run CLI agent (interactive or single query)
uv run clorag
uv run clorag "How do I configure the RIO IP address?"

# Ingest documentation from Docusaurus
uv run ingest-docs

# Ingest Gmail threads (curated with LLM analysis)
uv run ingest-curated --max-threads 300

# Incremental ingestion (skip first N threads)
uv run ingest-curated --offset 300 --max-threads 300

# Draft auto-reply for unanswered threads
uv run draft-support
uv run draft-support --preview  # Preview without creating

# Enrich camera model codes from documentation
uv run enrich-cameras

# Populate Neo4j knowledge graph from Qdrant chunks
uv run populate-graph
uv run populate-graph --collections docusaurus_docs gmail_cases
uv run populate-graph --max-chunks 100  # Limit for testing

# Linting and type checking
uv run ruff check src/
uv run mypy src/clorag --strict

# Tests
uv run pytest
uv run pytest tests/test_file.py::test_name -v  # Single test
```

## Architecture

### Data Flow
```
Query → EmbeddingsClient (voyage-context-3) → VectorStore (Qdrant) → GraphStore (Neo4j) → Claude synthesis
           ↓                                       ↓                       ↓
   SparseEmbeddingsClient (BM25)            Hybrid RRF fusion      Graph enrichment
```

### Key Components

**Core Layer** (`src/clorag/core/`):
- `vectorstore.py` - AsyncQdrantClient with hybrid search (dense + sparse vectors, RRF fusion)
- `embeddings.py` - Voyage AI client using `voyage-context-3` for contextualized embeddings
- `sparse_embeddings.py` - FastEmbed BM25 for keyword matching
- `graph_store.py` - Neo4j async client for knowledge graph operations
- `entity_extractor.py` - LLM-based entity extraction using Claude Haiku
- `database.py` - SQLite camera database with CRUD operations and upsert pattern
- `analytics_db.py` - Separate SQLite database for search analytics tracking

**Ingestion Layer** (`src/clorag/ingestion/`):
- `curated_gmail.py` - 7-step pipeline: Fetch → Anonymize → Haiku analysis → Filter resolved → Sonnet QC → Embed → Store
- `docusaurus.py` - Sitemap-based parallel page fetching with HTML extraction

**Analysis Layer** (`src/clorag/analysis/`):
- `thread_analyzer.py` - Claude Haiku for fast parallel thread classification
- `quality_controller.py` - Claude Sonnet for QC refinement of resolved cases
- `camera_extractor.py` - LLM-based camera info extraction from docs/support cases

**Drafts Layer** (`src/clorag/drafts/`):
- `gmail_service.py` - Gmail API with draft creation
- `draft_generator.py` - RAG-based response generator
- `draft_pipeline.py` - Draft creation orchestration

**Services Layer** (`src/clorag/services/`):
- `custom_docs.py` - `CustomDocumentService` for CRUD operations on custom knowledge documents (chunking, embedding, Qdrant storage)

**Graph Layer** (`src/clorag/graph/`):
- `schema.py` - Pydantic models for graph entities (Camera, Product, Protocol, Issue, Solution)
- `enrichment.py` - Graph context enrichment service for RAG search results

**Web Layer** (`src/clorag/web/`):
- `app.py` - FastAPI with streaming responses, hybrid RRF search across all 3 collections
- Camera management routes: public `/cameras`, admin `/admin/cameras`
- Knowledge base management: `/admin/knowledge` for custom documents
- Analytics dashboard: `/admin/analytics` with search stats and history
- Draft management: `/admin/drafts` for auto-reply system
- Chunk editor: `/admin/chunks` for browsing, searching, editing vector database chunks
- REST API for cameras: `GET/POST/PUT/DELETE /api/cameras`
- REST API for knowledge: `GET/POST/PUT/DELETE /api/admin/knowledge`, `POST /api/admin/knowledge/upload` (file upload)
- REST API for analytics: `GET /api/admin/search-stats`
- REST API for chunks: `GET/PUT/DELETE /api/admin/chunks`
- REST API for graph: `GET /api/admin/graph/stats`

**Models Layer** (`src/clorag/models/`):
- `camera.py` - Camera Pydantic models with CameraSource enum for tracking data origin
- `custom_document.py` - Custom document models with DocumentCategory enum, full metadata support

**Utils Layer** (`src/clorag/utils/`):
- `token_encryption.py` - Fernet encryption for OAuth tokens at rest (PBKDF2 key derivation)
- `logger.py` - Structured logging with structlog

### Vector Collections

Three Qdrant collections with named vectors:
- `docusaurus_docs` - Documentation chunks
- `gmail_cases` - Anonymized support cases
- `custom_docs` - Custom knowledge documents (admin-managed)

Each collection uses:
- `dense` - 1024-dim voyage-context-3 embeddings
- `sparse` - BM25 vectors for keyword matching

## Configuration

All settings via environment variables (see `.env.example`):
- `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` - Required API keys
- `QDRANT_URL` - Vector database URL (supports HTTPS reverse proxy)
- `DOCUSAURUS_URL` - Documentation site to scrape
- `GMAIL_LABEL` - Gmail label for support threads
- `DATABASE_PATH` - SQLite database for camera data (default: `data/clorag.db`)
- `ANALYTICS_DATABASE_PATH` - SQLite database for search analytics (default: `data/analytics.db`)
- `ADMIN_PASSWORD` - Admin authentication for camera management and analytics (also used for OAuth token encryption)
- `SEARXNG_URL` - SearXNG instance URL for web searches (default: `https://search.sapti.me`)
- `SECURE_COOKIES` - Enable secure cookies for HTTPS (default: `true`, set `false` for local development)
- `NEO4J_URI` - Neo4j Bolt protocol URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (optional, disables GraphRAG if not set)
- `NEO4J_DATABASE` - Neo4j database name (default: `neo4j`)

Settings loaded via `clorag.config.get_settings()` (cached singleton).

## Important Patterns

### Async Operations
VectorStore uses `AsyncQdrantClient`. All search methods are async and use `asyncio.gather()` for parallel dual-collection search.

### Hybrid Search
Search endpoints generate both dense and sparse query vectors, then use RRF (Reciprocal Rank Fusion) to combine semantic and keyword results across all three collections (docs, cases, custom_docs).

### Query Embedding Cache
Both dense (Voyage AI) and sparse (BM25) query embeddings are cached using thread-safe LRU caches (200 entries each). Cache is keyed by query text + model + dimensions. Check `embeddings.py:get_query_cache()` and `sparse_embeddings.py:SparseQueryCache`.

### Dynamic Score Thresholds
Search results are filtered using adaptive thresholds in `app.py:_compute_dynamic_threshold()`:
- Short queries (≤2 words): threshold 0.15 (permissive)
- Medium queries (3-5 words): threshold 0.20
- Long queries (>5 words): threshold 0.25 (strict)
- Technical terms (rio, rcp, firmware): +0.05 boost
- Always returns minimum 3 results regardless of threshold

### Contextualized Embeddings
Documents are embedded using `voyage-context-3`'s `contextualized_embed()` which encodes chunk content with full document context for improved retrieval.

### Anonymization
Gmail threads are anonymized before LLM analysis using placeholder tokens (`[SERIAL:XXX-N]`, `[EMAIL-N]`) to protect customer data.

### Camera Extraction
During ingestion, camera information is automatically extracted from documentation and support cases using Claude Haiku. Data is merged using upsert pattern to combine info from multiple sources.

### Human-Modified Data Protection
Chunks and cameras manually edited via admin UI must NOT be overwritten by automated processes (RAG ingestion, enrich-cameras). Check for human modification flags before updating existing records.

### Admin Authentication
Admin routes are protected by session-based authentication with brute force protection (5 failed attempts triggers 5-minute lockout per IP). Set `ADMIN_PASSWORD` env var to enable admin access. Login at `/admin/login`.

### Security Features
- **Brute force protection** - `LoginAttemptTracker` in `app.py` limits login attempts per IP
- **XSS protection** - DOMPurify sanitizes markdown rendering with SVG allowlist for Mermaid
- **OAuth token encryption** - `utils/token_encryption.py` uses Fernet encryption with PBKDF2 (480K iterations)
- **Secure cookies** - Session cookies use `secure=True` in production (controlled by `SECURE_COOKIES` env var)
- **PII anonymization** - Customer data anonymized before LLM processing

### Mermaid Diagram Generation
Claude automatically generates Mermaid.js diagrams when explaining camera connections, network topology, or signal flows. Diagrams are rendered client-side using Mermaid.js v11 ESM modules with Cyanview color theming.

### Custom Knowledge Documents
Admin-managed documents stored in `custom_docs` Qdrant collection. Features:
- **File upload**: Support for .txt, .md, .pdf files with drag-and-drop UI
- **PDF extraction**: Text extracted using pypdf library
- 9 categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, internal, other
- Full metadata: title, tags, URL reference, expiration date, notes
- Chunked, embedded, and included in hybrid RAG search
- Admin UI at `/admin/knowledge` with "Paste Text" and "Upload File" modes

### GraphRAG (Knowledge Graph Augmented Retrieval)
Optional Neo4j-based knowledge graph enrichment:
- **Entity types**: Camera, Product, Protocol, Port, Control, Issue, Solution, Firmware, Chunk
- **Relationships**: COMPATIBLE_WITH, USES_PROTOCOL, HAS_PORT, AFFECTS, RESOLVED_BY, MENTIONS
- **Population**: `uv run populate-graph` extracts entities from Qdrant chunks using Claude Haiku
- **Integration**: Graph context is automatically added to Claude synthesis when Neo4j is configured
- **Graceful degradation**: Works without Neo4j if `NEO4J_PASSWORD` is not set

## Deployment

Production runs on Docker:
```bash
rsync -avz --exclude '.venv' ... root@cyanview.cloud:/opt/clorag/
ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

Web UI at https://cyanview.cloud/ (reverse proxy to port 8085).
