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

# Linting and type checking
uv run ruff check src/
uv run mypy src/clorag --strict

# Tests
uv run pytest
```

## Architecture

### Data Flow
```
Query → EmbeddingsClient (voyage-context-3) → VectorStore (Qdrant) → Claude Sonnet 4.5 synthesis → Response
           ↓                                       ↓
   SparseEmbeddingsClient (BM25)            Hybrid RRF fusion (3 collections)
```

### Key Components

**Core Layer** (`src/clorag/core/`):
- `vectorstore.py` - AsyncQdrantClient with hybrid search (dense + sparse vectors, RRF fusion)
- `embeddings.py` - Voyage AI client using `voyage-context-3` for contextualized embeddings
- `sparse_embeddings.py` - FastEmbed BM25 for keyword matching
- `database.py` - SQLite camera database with CRUD operations and upsert pattern
- `analytics_db.py` - Separate SQLite database for search analytics tracking

**Ingestion Layer** (`src/clorag/ingestion/`):
- `curated_gmail.py` - 7-step pipeline: Fetch → Anonymize → Haiku analysis → Filter resolved → Sonnet QC → Embed → Store
- `docusaurus.py` - Sitemap-based parallel page fetching with HTML extraction

**Analysis Layer** (`src/clorag/analysis/`):
- `thread_analyzer.py` - Claude Haiku for fast parallel thread classification
- `quality_controller.py` - Claude Sonnet for QC refinement of resolved cases
- `camera_extractor.py` - LLM-based camera info extraction from docs/support cases

**Services Layer** (`src/clorag/services/`):
- `custom_docs.py` - `CustomDocumentService` for CRUD operations on custom knowledge documents (chunking, embedding, Qdrant storage)

**Web Layer** (`src/clorag/web/`):
- `app.py` - FastAPI with streaming responses, hybrid RRF search across all 3 collections
- Camera management routes: public `/cameras`, admin `/admin/cameras`
- Knowledge base management: `/admin/knowledge` for custom documents
- Analytics dashboard: `/admin/analytics` with search stats and history
- Draft management: `/admin/drafts` for auto-reply system
- REST API for cameras: `GET/POST/PUT/DELETE /api/cameras`
- REST API for knowledge: `GET/POST/PUT/DELETE /api/admin/knowledge`
- REST API for analytics: `GET /api/admin/search-stats`

**Models Layer** (`src/clorag/models/`):
- `camera.py` - Camera Pydantic models with CameraSource enum for tracking data origin
- `custom_document.py` - Custom document models with DocumentCategory enum, full metadata support

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
- `ADMIN_PASSWORD` - Admin authentication for camera management and analytics
- `SEARXNG_URL` - SearXNG instance URL for web searches (default: `https://search.sapti.me`)

Settings loaded via `clorag.config.get_settings()` (cached singleton).

## Important Patterns

### Async Operations
VectorStore uses `AsyncQdrantClient`. All search methods are async and use `asyncio.gather()` for parallel dual-collection search.

### Hybrid Search
Search endpoints generate both dense and sparse query vectors, then use RRF (Reciprocal Rank Fusion) to combine semantic and keyword results across all three collections (docs, cases, custom_docs).

### Contextualized Embeddings
Documents are embedded using `voyage-context-3`'s `contextualized_embed()` which encodes chunk content with full document context for improved retrieval.

### Anonymization
Gmail threads are anonymized before LLM analysis using placeholder tokens (`[SERIAL:XXX-N]`, `[EMAIL-N]`) to protect customer data.

### Camera Extraction
During ingestion, camera information is automatically extracted from documentation and support cases using Claude Haiku. Data is merged using upsert pattern to combine info from multiple sources.

### Admin Authentication
Admin routes are protected by session-based authentication. Set `ADMIN_PASSWORD` env var to enable admin access. Login at `/admin/login`.

### Mermaid Diagram Generation
Claude automatically generates Mermaid.js diagrams when explaining camera connections, network topology, or signal flows. Diagrams are rendered client-side using Mermaid.js v11 ESM modules with Cyanview color theming.

### Custom Knowledge Documents
Admin-managed documents stored in `custom_docs` Qdrant collection. Features:
- 9 categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, internal, other
- Full metadata: title, tags, URL reference, expiration date, notes
- Chunked, embedded, and included in hybrid RAG search

## Deployment

Production runs on Docker:
```bash
rsync -avz --exclude '.venv' ... root@cyanview.cloud:/opt/clorag/
ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

Web UI at https://cyanview.cloud/ (reverse proxy to port 8085).
