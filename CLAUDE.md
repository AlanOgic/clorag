# CLAUDE.md

## Project Overview

CLORAG is a Multi-RAG agent for Cyanview support combining:
- **Docusaurus documentation** from the support site
- **Gmail support threads** (curated and anonymized)
- **Custom knowledge documents** (admin-managed)

Uses hybrid search (dense voyage-context-3 + sparse BM25 vectors with RRF fusion) + **Voyage rerank-2.5** cross-encoder for refined relevance across three Qdrant collections. Claude Sonnet synthesizes responses with automatic Mermaid diagrams for integration scenarios.

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
uv run enrich-cameras                      # Extract camera info from docs
uv run populate-graph                      # Build Neo4j knowledge graph
uv run draft-support                       # Auto-reply drafts (--preview)
uv run rebuild-fts                         # Rebuild camera FTS5 index

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

**Core** (`core/`): `vectorstore.py` (AsyncQdrantClient, RRF fusion), `embeddings.py` (voyage-context-3 with contextualized embedding), `sparse_embeddings.py` (BM25), `reranker.py` (Voyage rerank-2.5 cross-encoder), `retriever.py` (MultiSourceRetriever with reranking), `graph_store.py` (Neo4j), `entity_extractor.py` (Haiku), `database.py` (camera SQLite), `analytics_db.py`, `support_case_db.py` (support cases SQLite with FTS5)

**Ingestion** (`ingestion/`): `curated_gmail.py` (7-step: Fetch→Anonymize→Haiku→Filter→Sonnet QC→Embed→Store), `docusaurus.py` (sitemap crawler), `chunker.py`, `base.py`

**Analysis** (`analysis/`): `thread_analyzer.py` (Haiku classification), `quality_controller.py` (Sonnet QC), `camera_extractor.py`

**Agent** (`agent/`): `tools.py` (Claude Agent SDK MCP tools), `prompts.py`

**Graph** (`graph/`): `schema.py` (Camera, Product, Protocol, Issue, Solution entities), `enrichment.py`

**Services** (`services/`): `custom_docs.py` (CustomDocumentService CRUD)

**Drafts** (`drafts/`): `gmail_service.py`, `draft_generator.py`, `draft_pipeline.py`

**Web** (`web/app.py`): FastAPI with streaming, admin UI at `/admin/{cameras,knowledge,analytics,drafts,chunks,graph,support-cases}`, REST APIs at `/api/`

**Models** (`models/`): `camera.py`, `custom_document.py` (10 categories), `support_case.py`

**Utils** (`utils/`): `token_encryption.py` (Fernet/PBKDF2), `anonymizer.py`, `logger.py`

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

Settings via `clorag.config.get_settings()` (cached singleton).

## Key Patterns

### Search & Retrieval
- **Hybrid search**: Dense + sparse vectors combined via RRF across all collections
- **Reranking**: Voyage `rerank-2.5` cross-encoder refines top results (+15-40% relevance improvement)
- **Over-fetch strategy**: Retrieves 3x limit, reranks, returns top-K for optimal quality
- **Query cache**: LRU caches (200 entries) for embeddings + (100 entries) for reranking
- **Dynamic thresholds**: ≤2 words: 0.15, 3-5: 0.20, >5: 0.25; technical terms +0.05; minimum 3 results
- **Contextualized embeddings**: `voyage-context-3` encodes chunks with full document context

### Data Protection
- **Anonymization**: Gmail threads use placeholders (`[SERIAL:XXX-N]`, `[EMAIL-N]`) before LLM processing
- **Human edits preserved**: Admin-modified chunks/cameras are NOT overwritten by automated ingestion
- **Camera extraction**: Haiku extracts camera info during ingestion; upsert merges multiple sources

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
- **Admin UI**: Browse/search cases at `/admin/support-cases` with detail modal (Summary/Document/Raw tabs)

### Security
- Session-based admin auth with brute force protection (5 attempts → 5min lockout per IP)
- XSS: DOMPurify with SVG allowlist for Mermaid
- OAuth tokens: Fernet encryption with PBKDF2 (480K iterations)
- Secure cookies in production (configurable via `SECURE_COOKIES`)

### GraphRAG
Optional Neo4j knowledge graph with entities (Camera, Product, Protocol, Port, Issue, Solution, Firmware) and relationships (COMPATIBLE_WITH, USES_PROTOCOL, AFFECTS, RESOLVED_BY, etc.). Gracefully disabled if `NEO4J_PASSWORD` not set.

**Local dev**: SSH tunnel to production Neo4j:
```bash
ssh -L 7687:localhost:7687 root@cyanview.cloud -N -f
```

### Custom Documents
10 categories: product_info, troubleshooting, configuration, firmware, release_notes, faq, best_practices, pre_sales, internal, other. Supports .txt/.md/.pdf upload, full metadata, chunked and embedded into RAG search.

## Deployment

```bash
rsync -avz --exclude '.venv' ... root@cyanview.cloud:/opt/clorag/
ssh root@cyanview.cloud "cd /opt/clorag && docker compose build && docker compose up -d"
```

Production: https://cyanview.cloud/ (Docker maps 8085→8080)
