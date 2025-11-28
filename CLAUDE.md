# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLORAG is a Multi-RAG (Retrieval-Augmented Generation) agent for Cyanview support that combines:
- **Docusaurus documentation** from the support site
- **Gmail support threads** (curated and anonymized)

Uses hybrid search (dense + sparse vectors with RRF fusion) for optimal retrieval of technical product information.

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
Query → EmbeddingsClient (voyage-context-3) → VectorStore (Qdrant) → Claude Haiku synthesis → Response
           ↓                                       ↓
   SparseEmbeddingsClient (BM25)            Hybrid RRF fusion
```

### Key Components

**Core Layer** (`src/clorag/core/`):
- `vectorstore.py` - AsyncQdrantClient with hybrid search (dense + sparse vectors, RRF fusion)
- `embeddings.py` - Voyage AI client using `voyage-context-3` for contextualized embeddings
- `sparse_embeddings.py` - FastEmbed BM25 for keyword matching
- `database.py` - SQLite camera database with CRUD operations and upsert pattern

**Ingestion Layer** (`src/clorag/ingestion/`):
- `curated_gmail.py` - 7-step pipeline: Fetch → Anonymize → Haiku analysis → Filter resolved → Sonnet QC → Embed → Store
- `docusaurus.py` - Sitemap-based parallel page fetching with HTML extraction

**Analysis Layer** (`src/clorag/analysis/`):
- `thread_analyzer.py` - Claude Haiku for fast parallel thread classification
- `quality_controller.py` - Claude Sonnet for QC refinement of resolved cases
- `camera_extractor.py` - LLM-based camera info extraction from docs/support cases

**Web Layer** (`src/clorag/web/`):
- `app.py` - FastAPI with streaming responses, hybrid RRF search across both collections
- Camera management routes: public `/cameras`, admin `/admin/cameras`
- REST API for cameras: `GET/POST/PUT/DELETE /api/cameras`

**Models Layer** (`src/clorag/models/`):
- `camera.py` - Camera Pydantic models with CameraSource enum for tracking data origin

### Vector Collections

Two Qdrant collections with named vectors:
- `docusaurus_docs` - Documentation chunks
- `gmail_cases` - Anonymized support cases

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
- `ADMIN_PASSWORD` - Admin authentication for camera management

Settings loaded via `clorag.config.get_settings()` (cached singleton).

## Important Patterns

### Async Operations
VectorStore uses `AsyncQdrantClient`. All search methods are async and use `asyncio.gather()` for parallel dual-collection search.

### Hybrid Search
Search endpoints generate both dense and sparse query vectors, then use RRF (Reciprocal Rank Fusion) to combine semantic and keyword results.

### Contextualized Embeddings
Documents are embedded using `voyage-context-3`'s `contextualized_embed()` which encodes chunk content with full document context for improved retrieval.

### Anonymization
Gmail threads are anonymized before LLM analysis using placeholder tokens (`[SERIAL:XXX-N]`, `[EMAIL-N]`) to protect customer data.

### Camera Extraction
During ingestion, camera information is automatically extracted from documentation and support cases using Claude Haiku. Data is merged using upsert pattern to combine info from multiple sources.

### Admin Authentication
Camera admin routes are protected by `X-Admin-Password` header. Set `ADMIN_PASSWORD` env var to enable admin access.

## Deployment

Production runs on Docker:
```bash
rsync -avz --exclude '.venv' ... root@cyanview.cloud:/opt/clorag/
ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

Web UI at https://cyanview.cloud/ (reverse proxy to port 8085).
