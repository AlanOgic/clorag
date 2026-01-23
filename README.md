```
   ██████╗██╗      ██████╗ ██████╗  █████╗  ██████╗
  ██╔════╝██║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝
  ██║     ██║     ██║   ██║██████╔╝███████║██║  ███╗
  ██║     ██║     ██║   ██║██╔══██╗██╔══██║██║   ██║
  ╚██████╗███████╗╚██████╔╝██║  ██║██║  ██║╚██████╔╝
   ╚═════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
```

# Multi-Source RAG Agent for Cyanview Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Claude SDK](https://img.shields.io/badge/Claude-Agent%20SDK-orange.svg)](https://github.com/anthropics/anthropic-sdk-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent support agent combining **Docusaurus documentation**, **Gmail support threads**, and **custom knowledge documents** through a hybrid RAG (Retrieval-Augmented Generation) system with Claude AI synthesis.

---

## Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Technology Stack](#technology-stack)
- [RAG System Architecture](#rag-system-architecture)
  - [Query Pipeline](#query-pipeline)
  - [Hybrid Search Strategy](#hybrid-search-strategy)
  - [Reranking Pipeline](#reranking-pipeline)
  - [Vector Collections](#vector-collections)
  - [GraphRAG Enrichment](#graphrag-enrichment)
- [Data Ingestion](#data-ingestion)
  - [Documentation Pipeline](#documentation-pipeline)
  - [Gmail Support Cases Pipeline](#gmail-support-cases-pipeline)
  - [Custom Documents](#custom-documents)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Web Interface](#web-interface)
  - [API Reference](#api-reference)
- [Administration](#administration)
  - [Admin Features](#admin-features)
  - [Camera Database](#camera-database)
  - [Prompt Management](#prompt-management)
- [Deployment](#deployment)
- [Security](#security)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

CLORAG is a production-ready Multi-RAG agent designed to power Cyanview's technical support. It intelligently searches across multiple knowledge sources, understands context from previous interactions, and synthesizes comprehensive answers with relevant documentation links.

### Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid RAG Search** | Dense vectors (Voyage AI) + sparse BM25 with RRF fusion |
| **Cross-Encoder Reranking** | Voyage rerank-2.5 for +15-40% relevance improvement |
| **Streaming Responses** | Real-time SSE streaming with Claude Sonnet synthesis |
| **Follow-up Conversations** | Context-aware with last 3 Q&A exchanges |
| **GraphRAG Enrichment** | Optional Neo4j knowledge graph for entity relationships |
| **Camera Compatibility DB** | Structured camera info with FTS5 full-text search |
| **Auto-Draft System** | AI-powered draft replies for unanswered support threads |
| **Admin Dashboard** | Full CRUD for cameras, documents, chunks, and analytics |
| **Token-Aware Chunking** | Configurable tiktoken-based chunking by content type |
| **Performance Monitoring** | Real-time metrics with percentile stats and alerts |

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | Claude Agent SDK 0.1.9+ |
| **LLM Synthesis** | Claude Sonnet 4.5 (streaming) |
| **Dense Embeddings** | Voyage AI (voyage-context-3, 1024-dim) |
| **Sparse Embeddings** | FastEmbed BM25 |
| **Reranking** | Voyage AI (rerank-2.5) |
| **Vector Database** | Qdrant (hybrid search) |
| **Graph Database** | Neo4j (optional GraphRAG) |
| **Relational Database** | SQLite (cameras, analytics, support cases, prompts) |
| **Web Framework** | FastAPI + Jinja2 |
| **Web Scraping** | Jina Reader (BeautifulSoup fallback) |
| **Package Manager** | uv |

---

## RAG System Architecture

### Query Pipeline

The RAG system processes queries through a multi-stage pipeline optimized for accuracy and speed:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
│                        "How to configure RIO?"                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ 
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING GENERATION                                 │
│  ┌────────────────────────────┐    ┌────────────────────────────┐            │
│  │     Dense Embeddings       │    │     Sparse Embeddings      │            │
│  │     voyage-context-3       │    │      FastEmbed BM25        │            │
│  │       (1024 dim)           │    │     (keyword match)        │            │
│  └────────────────────────────┘    └────────────────────────────┘            │
│                    └──────────────┬──────────────┘                           │
│                                   │                                          │
│                      LRU Cache (200 entries)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID SEARCH                                      │
│  ┌───────────────────────────────────────────────────────────────────┐       │
│  │                        Qdrant Vector DB                           │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │       │
│  │  │ docusaurus_docs │  │   gmail_cases   │  │   custom_docs   │    │       │
│  │  │ (documentation) │  │ (support cases) │  │ (admin uploads) │    │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│                        RRF Fusion (k=60)                                     │
│                   Over-fetch 3x, Dynamic Prefetch                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐ 
│                            RERANKING                                         │
│                   Voyage AI rerank-2.5 cross-encoder                         │
│               +15-40% relevance improvement on top results                   │
│                     LRU Cache (100 entries)                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      OPTIONAL: GRAPH ENRICHMENT                              │
│              Neo4j traversal for entity relationships                        │
│        (cameras, protocols, ports, issues, solutions)                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ANSWER SYNTHESIS                                     │
│                    Claude Sonnet 4.5 (streaming)                             │
│          • Warm, professional Cyanview support tone                          │
│          • Conversation context (last 3 Q&A)                                 │
│          • Related documentation links                                       │
│          • Automatic Mermaid diagrams for integrations                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        STREAMING RESPONSE                                    │
│                     Real-time SSE to frontend                                │
│              Session ID for follow-up questions (30min TTL)                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Search Strategy

The system combines semantic understanding with keyword matching for optimal retrieval:

| Strategy | Technology | Purpose |
|----------|------------|---------|
| **Dense Vectors** | voyage-context-3 (1024-dim) | Semantic similarity, concept matching |
| **Sparse Vectors** | BM25 via FastEmbed | Exact keyword matching, technical terms |
| **Fusion** | Reciprocal Rank Fusion (k=60) | Combines rankings from both strategies |
| **Over-fetch** | 3x limit, max 50 | Retrieves more results for better reranking |

**Dynamic Score Thresholds:**
- ≤2 words: 0.15 minimum score
- 3-5 words: 0.20 minimum score
- >5 words: 0.25 minimum score
- Technical terms: +0.05 boost
- Minimum 3 results always returned

### Reranking Pipeline

After hybrid search, results are refined using a cross-encoder model:

```
Initial Results (15-30 chunks)
           │
           ▼
┌─────────────────────────────┐
│   Voyage rerank-2.5         │
│   Cross-encoder scoring     │
│   Query ↔ Document pairs    │
└─────────────────────────────┘
           │
           ▼
   Top-K Results (default: 5)
   Sorted by relevance score
```

Configuration:
- `RERANK_ENABLED=true` - Enable/disable reranking
- `VOYAGE_RERANK_MODEL=rerank-2.5` - Model selection
- `RERANK_TOP_K=5` - Final results count

### Vector Collections

Three Qdrant collections store the knowledge base:

| Collection | Source | Content |
|------------|--------|---------|
| `docusaurus_docs` | Sitemap crawler | Official documentation pages |
| `gmail_cases` | Gmail API | Curated, anonymized support threads |
| `custom_docs` | Admin uploads | Custom knowledge (.txt, .md, .pdf) |

Each collection contains:
- **Dense vectors**: 1024-dimensional voyage-context-3 embeddings
- **Sparse vectors**: BM25 vectors for keyword matching
- **Metadata**: Source URL, timestamps, categories, extracted entities

### GraphRAG Enrichment

When Neo4j is configured, search results are enriched with knowledge graph context:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VECTOR SEARCH RESULTS                                  │
│                 Top N chunks from Qdrant hybrid search                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRAPH ENRICHMENT                                     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  1. enrich_from_chunks()                                           │     │
│  │     Traverse graph from chunk IDs to find:                         │     │
│  │     • Related cameras and their protocols                          │     │
│  │     • Products with compatibility info                             │     │
│  │     • Known issues and solutions                                   │     │
│  │                                                                    │     │
│  │  2. enrich_from_query()                                            │     │
│  │     Full-text search for query-relevant entities:                  │     │
│  │     • Cameras matching query terms                                 │     │
│  │     • Protocols and ports mentioned                                │     │
│  │     • Firmware versions                                            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLAUDE SYNTHESIS                                   │
│           Vector chunks + Graph context → Rich answer                       │
│      Example: "Sony cameras use Visca protocol via RS-422"                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Entity Types:** Camera, Product, Protocol, Port, Control, Issue, Solution, Firmware, Chunk

**Relationships:** COMPATIBLE_WITH, USES_PROTOCOL, HAS_PORT, AFFECTS, RESOLVED_BY, MENTIONS

---

## Data Ingestion

### Documentation Pipeline

Fetches and processes Docusaurus documentation pages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Sitemap.xml    │ ──▶ │   Jina Reader   │ ──▶ │    Chunking     │
│  URL Discovery  │     │   (+ fallback)  │     │  (token-based)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Qdrant      │ ◀── │   Embedding     │ ◀── │  RIO Fix        │
│    Storage      │     │ (contextualized)│     │ (auto-apply)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Features:**
- Jina Reader primary (clean markdown with table preservation)
- BeautifulSoup fallback on 429/503 errors
- Token-based chunking (450 tokens for docs)
- RIO terminology auto-correction before embedding
- Camera extraction post-ingestion via Claude Haiku

```bash
uv run ingest-docs                    # Full ingestion
uv run ingest-docs https://custom.url # Custom URL
```

### Gmail Support Cases Pipeline

A 7-step pipeline processes Gmail threads into high-quality knowledge:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: GMAIL API FETCH                                                    │
│  Fetch threads from configured label (supports)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ANONYMIZATION                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  PII Removal (regex patterns):                                     │     │
│  │  • Serial numbers: CY-RIO-48-12 → [SERIAL:RIO-1]                   │     │
│  │  • Emails: john@company.com → [EMAIL-1]                            │     │
│  │  • Phone numbers: +1-555-1234 → [PHONE-1]                          │     │
│  │  • Cyanview emails preserved: support@cyanview.com                 │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: HAIKU ANALYSIS (parallel processing)                               │
│  Extract: problem summary, solution steps, keywords, confidence, status     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: FILTER                                                             │
│  Keep only resolved cases with confidence >= 0.7                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SONNET QC (quality control)                                        │
│  Refine and validate Haiku extractions                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: CONTEXTUALIZED EMBEDDING                                           │
│  voyage-context-3 with full document context + BM25 sparse vectors          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: STORAGE                                                            │
│  Qdrant (gmail_cases) + SQLite (support_cases with FTS5)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

```bash
uv run ingest-curated --max-threads 300      # Full ingestion
uv run ingest-curated --offset 300           # Incremental
```

### Custom Documents

Admin-uploaded documents for specialized knowledge:

**Supported formats:** .txt, .md, .pdf

**Categories:**
| Category | Description |
|----------|-------------|
| `product_info` | Product specifications and datasheets |
| `troubleshooting` | Problem-solving guides |
| `configuration` | Setup and configuration guides |
| `firmware` | Firmware documentation |
| `release_notes` | Version release notes |
| `faq` | Frequently asked questions |
| `best_practices` | Recommended practices |
| `pre_sales` | Pre-sales technical information |
| `internal` | Internal documentation |
| `other` | Miscellaneous documents |

```bash
uv run import-docs ./folder --category pre_sales  # Bulk import
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Qdrant** server (local Docker or remote)
- **API Keys**: Anthropic, Voyage AI
- **Optional**: Neo4j for GraphRAG, Gmail OAuth for support cases

### Quick Start

```bash
# Clone and install
cd clorag
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the web server
uv run rag-web
```

### Configuration

Create a `.env` file with the following variables:

```env
# ═══════════════════════════════════════════════════════════════════════════
# REQUIRED - API Keys
# ═══════════════════════════════════════════════════════════════════════════
ANTHROPIC_API_KEY=your_anthropic_key
VOYAGE_API_KEY=your_voyage_key

# ═══════════════════════════════════════════════════════════════════════════
# REQUIRED - Qdrant Vector Database
# ═══════════════════════════════════════════════════════════════════════════
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key                # Optional for local

# ═══════════════════════════════════════════════════════════════════════════
# REQUIRED - Admin Authentication
# ═══════════════════════════════════════════════════════════════════════════
ADMIN_PASSWORD=your_secure_password           # Also used for OAuth encryption

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Data Sources
# ═══════════════════════════════════════════════════════════════════════════
DOCUSAURUS_URL=https://your-docs-site.com
GMAIL_LABEL=supports

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Database Paths
# ═══════════════════════════════════════════════════════════════════════════
DATABASE_PATH=data/clorag.db                  # Camera database
ANALYTICS_DATABASE_PATH=data/analytics.db     # Search analytics

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Neo4j GraphRAG
# ═══════════════════════════════════════════════════════════════════════════
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Reranking
# ═══════════════════════════════════════════════════════════════════════════
RERANK_ENABLED=true
VOYAGE_RERANK_MODEL=rerank-2.5
RERANK_TOP_K=5

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Chunking Configuration
# ═══════════════════════════════════════════════════════════════════════════
CHUNK_USE_TOKENS=true                         # Token-based (recommended)
CHUNK_SIZE_DOCS=450                           # Documentation (tokens)
CHUNK_SIZE_CASES=350                          # Support cases (tokens)
CHUNK_SIZE_DEFAULT=400                        # Default (tokens)
CHUNK_OVERLAP=50                              # Overlap (~12.5%)
CHUNK_ADAPTIVE_THRESHOLD=200                  # Single-chunk threshold

# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL - Other Settings
# ═══════════════════════════════════════════════════════════════════════════
SEARXNG_URL=https://search.sapti.me           # Web search augmentation
SECURE_COOKIES=true                           # Set false for local dev
PROMPTS_CACHE_TTL=300                         # Prompt cache TTL (seconds)
```

---

## Usage

### CLI Commands

#### Core Operations
```bash
uv run rag-web                                # Start web server (port 8080)
uv run clorag "query"                         # CLI agent query
uv run clorag                                 # Interactive mode
```

#### Data Ingestion
```bash
uv run ingest-docs                            # Docusaurus documentation
uv run ingest-curated --max-threads 300       # Gmail support threads
uv run ingest-curated --offset N              # Incremental ingestion
uv run import-docs ./folder --category X      # Bulk import custom docs
```

#### Maintenance
```bash
uv run enrich-cameras                         # Extract camera info from docs
uv run populate-graph                         # Build Neo4j knowledge graph
uv run rebuild-fts                            # Rebuild camera FTS5 index
uv run draft-support                          # Generate auto-reply drafts
uv run draft-support --preview                # Preview without creating
```

#### RIO Terminology Fixes
```bash
uv run fix-rio-terminology --preview          # Scan for issues
uv run fix-rio-terminology --stats            # View statistics
uv run fix-rio-terminology --apply            # Apply approved fixes
```

#### Prompt Management
```bash
uv run init-prompts                           # Initialize prompt database
uv run init-prompts --list                    # List all prompts
uv run init-prompts --stats                   # Show statistics
```

#### Quality Checks
```bash
uv run ruff check src/                        # Linting
uv run mypy src/clorag --strict               # Type checking
uv run pytest                                 # Tests
```

### Web Interface

#### Public Pages

| URL | Description |
|-----|-------------|
| `/` | AI Search - Natural language query interface |
| `/cameras` | Camera Compatibility - Browse supported cameras |
| `/help` | User Guide - How to use the search features |

#### Admin Pages (requires login)

| URL | Description |
|-----|-------------|
| `/admin` | Dashboard with links to all features |
| `/admin/cameras` | Camera CRUD management |
| `/admin/knowledge` | Custom knowledge base (upload/paste) |
| `/admin/analytics` | Search analytics and statistics |
| `/admin/drafts` | Draft auto-reply management |
| `/admin/support-cases` | Browse ingested Gmail cases |
| `/admin/chunks` | Vector chunk browser/editor |
| `/admin/graph` | Knowledge Graph Explorer |
| `/admin/prompts` | LLM prompt editor with version history |
| `/admin/terminology-fixes` | RIO terminology review |
| `/admin/search-debug` | Debug RAG: chunks, prompts, timing |
| `/admin/docs` | Technical documentation |

### API Reference

#### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/search` | RAG search with synthesis |
| `POST` | `/api/search/stream` | Streaming search (SSE) |

**Follow-up conversations:**
```json
{
  "query": "What ports does it support?",
  "source": "both",
  "session_id": "uuid-from-previous-response"
}
```

#### Camera Endpoints (Public)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cameras` | List cameras (with filters) |
| `GET` | `/api/cameras/search?q=` | FTS5 search with BM25 |
| `GET` | `/api/cameras/{id}` | Get single camera |
| `GET` | `/api/cameras/{id}/related` | Similar cameras |
| `POST` | `/api/cameras/compare` | Compare up to 5 cameras |
| `GET` | `/api/cameras/export.csv` | CSV export |

#### Admin Endpoints

See the [API documentation](/admin/docs) for complete admin endpoint reference including:
- Camera CRUD
- Knowledge base management
- Chunk editing
- Graph operations
- Analytics
- Draft management
- Support cases
- Prompt management

---

## Administration

### Admin Features

- **Session-based auth** with signed cookies (24-hour expiry)
- **Brute force protection** (5 attempts → 5min lockout per IP)
- **Rate limiting** on login and admin endpoints

### Camera Database

SQLite database with:
- **FTS5 full-text search** with BM25 ranking and Porter stemming
- **Connection pool** (5 connections, WAL mode, 64MB cache)
- **TTL cache** (100 entries, 5-min TTL)
- **CSV import/export** with upsert logic
- **Side-by-side comparison** (up to 5 cameras)
- **Related cameras** based on similarity scoring

### Prompt Management

All 11 LLM prompts are stored in SQLite and editable via admin UI:

| Category | Prompts |
|----------|---------|
| `agent` | System prompt, tool descriptions |
| `analysis` | Thread analyzer, quality controller |
| `synthesis` | Answer generation, Mermaid diagrams |
| `drafts` | Auto-reply generation |
| `graph` | Entity extraction |
| `scripts` | Camera extraction, RIO analysis |

**Features:**
- Version history for audit and rollback
- Variable substitution with `{placeholder}` syntax
- In-memory caching with configurable TTL
- Fallback to hardcoded defaults

---

## Deployment

### Docker

```bash
# Build and deploy
docker compose build
docker compose up -d
```

### Manual Deployment

```bash
rsync -avz --exclude '.venv' --exclude 'data' . root@server:/opt/clorag/
ssh root@server "cd /opt/clorag && docker compose build && docker compose up -d"
```

### Infrastructure Setup

#### Qdrant (required)
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  -e QDRANT__SERVICE__API_KEY=your_api_key \
  qdrant/qdrant
```

#### Neo4j (optional)
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v $(pwd)/neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5-community

# Then populate the graph
uv run populate-graph
```

---

## Security

| Feature | Implementation |
|---------|---------------|
| **Session Authentication** | Signed cookies (itsdangerous) |
| **Brute Force Protection** | 5 failed attempts → 5min lockout/IP |
| **Rate Limiting** | 10/min login, configurable admin |
| **XSS Protection** | DOMPurify with SVG allowlist |
| **Content Security Policy** | Strict nonce-based CSP for public pages |
| **Open Redirect Prevention** | Validated login redirects |
| **HTTPS Cookies** | Secure, HttpOnly, SameSite=Strict |
| **Timing-Safe Comparison** | Prevents timing attacks |
| **OAuth Token Encryption** | Fernet + PBKDF2 (480K iterations) |
| **PII Anonymization** | Before LLM processing |
| **SQL Injection Prevention** | Parameterized queries |

---

## Project Structure

```
clorag/
├── src/clorag/
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Pydantic settings
│   │
│   ├── core/                      # Core infrastructure
│   │   ├── vectorstore.py         # Qdrant client, RRF fusion
│   │   ├── embeddings.py          # Voyage AI dense embeddings
│   │   ├── sparse_embeddings.py   # FastEmbed BM25
│   │   ├── reranker.py            # Voyage rerank-2.5
│   │   ├── retriever.py           # Multi-source retriever
│   │   ├── graph_store.py         # Neo4j async client
│   │   ├── entity_extractor.py    # LLM entity extraction
│   │   ├── database.py            # Camera SQLite
│   │   ├── analytics_db.py        # Analytics SQLite
│   │   ├── support_case_db.py     # Support cases SQLite
│   │   ├── prompt_db.py           # Prompts SQLite
│   │   ├── terminology_db.py      # RIO terminology fixes SQLite
│   │   ├── cache.py               # Generic LRU cache with TTL
│   │   └── metrics.py             # Performance instrumentation
│   │
│   ├── agent/                     # Claude Agent SDK
│   │   ├── tools.py               # MCP tools (RAG)
│   │   └── prompts.py             # System prompts
│   │
│   ├── analysis/                  # LLM analysis
│   │   ├── thread_analyzer.py     # Haiku classification
│   │   ├── quality_controller.py  # Sonnet QC
│   │   ├── camera_extractor.py    # Camera info extraction
│   │   └── rio_analyzer.py        # RIO terminology analysis
│   │
│   ├── ingestion/                 # Data pipelines
│   │   ├── docusaurus.py          # Documentation crawler
│   │   ├── curated_gmail.py       # Gmail 7-step pipeline
│   │   ├── chunker.py             # Token-aware chunking
│   │   └── base.py                # Base classes
│   │
│   ├── graph/                     # GraphRAG
│   │   ├── schema.py              # Entity models
│   │   └── enrichment.py          # Context enrichment
│   │
│   ├── services/                  # Business logic
│   │   ├── custom_docs.py         # Document CRUD
│   │   ├── prompt_manager.py      # Prompt management
│   │   └── default_prompts.py     # Hardcoded defaults
│   │
│   ├── drafts/                    # Auto-reply system
│   │   ├── gmail_service.py       # Gmail API
│   │   ├── draft_generator.py     # RAG-based generation
│   │   ├── draft_pipeline.py      # Orchestration
│   │   └── models.py              # Data models
│   │
│   ├── models/                    # Data models
│   │   ├── camera.py
│   │   ├── custom_document.py
│   │   └── support_case.py
│   │
│   ├── utils/                     # Utilities
│   │   ├── token_encryption.py    # Fernet/PBKDF2
│   │   ├── anonymizer.py          # PII removal
│   │   ├── logger.py              # Logging
│   │   ├── tokenizer.py           # tiktoken counting
│   │   └── text_transforms.py     # RIO product name transforms
│   │
│   ├── web/                       # FastAPI application
│   │   ├── app.py                 # Middleware and app init
│   │   ├── routers/               # API routes by domain
│   │   │   ├── admin/             # Admin endpoints (12 files)
│   │   │   ├── cameras.py         # Camera API
│   │   │   ├── pages.py           # Page routes
│   │   │   └── search.py          # Search API
│   │   ├── auth/                  # Authentication module
│   │   ├── schemas.py             # Request/response models
│   │   ├── search/                # Search pipeline
│   │   ├── dependencies.py        # FastAPI dependencies
│   │   ├── templates/             # Jinja2 templates (29 files)
│   │   └── static/                # CSS, JS assets
│   │
│   └── scripts/                   # CLI scripts
│       ├── ingest_docs.py
│       ├── ingest_curated.py
│       ├── populate_graph.py
│       ├── draft_support.py
│       ├── fix_rio_terminology.py
│       └── run_web.py
│
├── tests/                         # Test suite
├── data/                          # SQLite databases
├── pyproject.toml                 # Project configuration
└── docker-compose.yml             # Deployment config
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built for <a href="https://cyanview.com">Cyanview</a> Technical Support</strong>
</p>
