# CLORAG - Multi-RAG Agent avec Claude Agent SDK

Agent intelligent de support combinant documentation Docusaurus et cas de support Gmail via RAG (Retrieval-Augmented Generation).

## Features

- **AI-Powered Search** - Natural language queries with Claude Sonnet 4.5 synthesis
- **Follow-up Conversations** - Ask follow-up questions with context from last 3 exchanges
- **Hybrid RAG Search** - Combines semantic (Voyage AI) and keyword (BM25) matching with RRF fusion
- **Custom Knowledge Base** - Upload .txt, .md, .pdf files or paste text to add custom documents
- **Camera Compatibility Database** - Structured camera info with automatic extraction from docs/support
- **Camera Comparison** - Side-by-side comparison of up to 5 cameras with highlighted common specs
- **FTS5 Full-Text Search** - SQLite FTS5 with BM25 ranking for fast camera search
- **Support Cases Database** - SQLite storage for Gmail cases with FTS5 search and thread cleaning
- **Search Analytics** - Track popular queries, response times, and usage patterns
- **Session-Based Admin** - Secure login with signed cookies for all admin features
- **Streaming Responses** - Real-time answer streaming for better UX
- **Draft Auto-Reply System** - AI-powered draft creation for unanswered support threads
- **Query Embedding Cache** - LRU cache reduces API calls for repeated queries
- **Dynamic Score Thresholds** - Adaptive filtering based on query characteristics
- **GraphRAG** - Neo4j knowledge graph enrichment with entity extraction from chunks

## Architecture

### Query Flow

```
┌────────────────────────────────────────────────────────────┐
│                        USER QUERY                          │
│                    "How to configure RIO?"                 │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                   EMBEDDING GENERATION                     │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │  Dense Embeddings   │    │  Sparse Embeddings  │        │
│  │  voyage-context-3   │    │   FastEmbed BM25    │        │
│  │     (1024 dim)      │    │   (keyword match)   │        │
│  └─────────────────────┘    └─────────────────────┘        │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                      HYBRID SEARCH                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    Qdrant Vector DB                   │ │
│  │  ┌──────────────────┐      ┌──────────────────┐       │ │
│  │  │  docusaurus_docs │      │   gmail_cases    │       │ │
│  │  │  (documentation) │      │ (support cases)  │       │ │
│  │  └──────────────────┘      └──────────────────┘       │ │
│  └───────────────────────────────────────────────────────┘ │
│                              │                             │
│                              ▼                             │
│                 RRF Fusion (k=60)                          │
│         Combines semantic + keyword results                │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    ANSWER SYNTHESIS                        │
│              Claude Sonnet 4.5 (streaming)                 │
│         Warm, professional Cyanview support tone           │
│           + Conversation context (last 3 Q&A)              │
│                 + Related documentation links              │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                     STREAMING RESPONSE                     │
│                  Real-time SSE to frontend                 │
│             + Session ID for follow-up questions           │
└────────────────────────────────────────────────────────────┘
```

### GraphRAG Enrichment

When Neo4j is configured, search results are enriched with knowledge graph context:

```
┌────────────────────────────────────────────────────────────┐
│                    VECTOR SEARCH RESULTS                   │
│              Top N chunks from Qdrant hybrid search        │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    GRAPH ENRICHMENT                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  1. enrich_from_chunks()                              │ │
│  │     Traverse graph from chunk IDs to find:            │ │
│  │     • Related cameras and their protocols             │ │
│  │     • Products with compatibility info                │ │
│  │     • Known issues and solutions                      │ │
│  │                                                       │ │
│  │  2. enrich_from_query()                               │ │
│  │     Full-text search for query-relevant entities:     │ │
│  │     • Cameras matching query terms                    │ │
│  │     • Protocols and ports mentioned                   │ │
│  │     • Firmware versions                               │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                 CLAUDE SYNTHESIS                           │
│      Vector chunks + Graph context → Rich answer           │
│  Example: "Sony cameras use Visca protocol via RS-422"     │
└────────────────────────────────────────────────────────────┘
```

#### Graph Population Pipeline

```bash
# Extract entities from Qdrant chunks and populate Neo4j
uv run populate-graph

# Process specific collections
uv run populate-graph --collections docusaurus_docs gmail_cases

# Limit for testing
uv run populate-graph --max-chunks 100
```

**Entity Types**: Camera, Product, Protocol, Port, Control, Issue, Solution, Firmware, Chunk

**Relationships**: COMPATIBLE_WITH, USES_PROTOCOL, HAS_PORT, AFFECTS, RESOLVED_BY, MENTIONS

### Ingestion Pipelines

Two data ingestion pipelines populate the vector database:

| Pipeline | Command | Description |
|----------|---------|-------------|
| Documentation | `uv run ingest-docs` | Fetches sitemap, scrapes pages, chunks, embeds |
| Support Cases | `uv run ingest-curated` | Gmail → Anonymize → Haiku → Filter → Sonnet QC → Embed |

> **📖 Complete flow diagrams available in [Admin Docs](/admin/docs#data-ingestion)** after authentication.

#### Gmail Support Cases Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    GMAIL API                               │
│                 Fetch threads from label                   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    ANONYMIZATION                           │
│  ┌───────────────────────────────────────────--──────────┐ │
│  │  PII Removal                                          │ │
│  │  • Serial numbers: CY-RIO-48-12 → [SERIAL:RIO-1]      │ │
│  │  • Emails: john@company.com → [EMAIL-1]               │ │
│  │  • Phone numbers: +1-555-1234 → [PHONE-1]             │ │
│  │  • Cyanview emails preserved: support@cyanview.com    │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                  HAIKU ANALYSIS (parallel)                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Extract:                                             │ │
│  │  • Problem summary                                    │ │
│  │  • Solution steps                                     │ │
│  │  • Technical keywords                                 │ │
│  │  • Resolution confidence                              │ │
│  │  • Case status (resolved/pending/abandoned)           │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    FILTER: RESOLVED ONLY                   │
│              Keep cases with confidence >= 0.7             │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                  SONNET QC (quality control)               │
│                Refine and validate extractions             │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────-───┐
│                CONTEXTUALIZED EMBEDDING                    │
│         voyage-context-3 with document context             │
│             + BM25 sparse vectors                          │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    QDRANT STORAGE                          │
│                 gmail_cases collection                     │
│              Rich metadata for filtering                   │
└────────────────────────────────────────────────────────────┘
```

### Data Sources

| Source | Description | Vector Collection |
|--------|-------------|-------------------|
| Documentation | Docusaurus support site pages | `docusaurus_docs` |
| Support Cases | Curated, anonymized Gmail threads | `gmail_cases` |
| Custom Knowledge | Admin-uploaded documents (.txt, .md, .pdf) | `custom_docs` |
| Camera Database | Structured camera compatibility data | SQLite (relational) |

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Orchestration | Claude Agent SDK 0.1.9+ |
| Vector DB | Qdrant |
| Graph DB | Neo4j (optional, for GraphRAG) |
| Dense Embeddings | Voyage AI (voyage-context-3) |
| Sparse Embeddings | FastEmbed BM25 |
| LLM Synthesis | Claude Sonnet 4.5 |
| Database | SQLite (camera + analytics) |
| Web | FastAPI + Jinja2 |
| Sessions | itsdangerous (signed cookies) |
| Config | Pydantic Settings |
| Async | AnyIO |

## Installation

### Prerequis

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Qdrant server (local Docker ou VPS)

### Setup

```bash
# Clone le repo
cd clorag

# Installer les dependances
uv sync

# Copier et configurer l'environnement
cp .env.example .env
# Editer .env avec vos cles API
```

### Configuration

Creer un fichier `.env` avec :

```env
# API Keys (required)
ANTHROPIC_API_KEY=your_anthropic_key
VOYAGE_API_KEY=your_voyage_key

# Qdrant (required)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key  # optionnel

# Sources
DOCUSAURUS_URL=https://your-docs-site.com
GMAIL_LABEL=supports

# Database paths (optional, defaults shown)
DATABASE_PATH=data/clorag.db
ANALYTICS_DATABASE_PATH=data/analytics.db

# Admin authentication (required for admin features)
ADMIN_PASSWORD=your_secure_password

# SearXNG (optional, for web search augmentation)
SEARXNG_URL=https://search.sapti.me

# Neo4j (optional, for GraphRAG enrichment)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## Usage

### Lancer l'agent (mode interactif)

```bash
uv run clorag
```

### Query unique

```bash
uv run clorag "Comment configurer l'authentification ?"
```

### Ingestion des donnees

```bash
# Ingerer la documentation Docusaurus
uv run ingest-docs

# Ou avec URL specifique
uv run ingest-docs https://docs.example.com

# Ingerer les threads Gmail (label: supports)
uv run ingest-gmail

# Ingestion curatee avec analyse LLM (recommande)
uv run ingest-curated --max-threads 300

# Ingestion incrementale (skip premiers N threads)
uv run ingest-curated --offset 300 --max-threads 300
```

### Draft Auto-Reply

```bash
# Process all unanswered threads (max 10)
uv run draft-support

# Preview drafts without creating
uv run draft-support --preview

# Process specific thread
uv run draft-support --thread THREAD_ID

# Increase limit
uv run draft-support --max 20
```

### Database Maintenance

```bash
# Rebuild camera FTS5 search index
uv run rebuild-fts

# Check FTS index status
uv run rebuild-fts --check
```

### Web Interface

```bash
# Launch web server (port 8080)
uv run rag-web
```

#### Public Pages

| URL | Description |
|-----|-------------|
| `/` | AI Search - Natural language query interface |
| `/cameras` | Camera Compatibility - Browse supported cameras |
| `/help` | User Guide - How to use the search features |

#### Admin Pages (requires login)

| URL | Description |
|-----|-------------|
| `/admin/login` | Admin login page |
| `/admin` | Admin dashboard with links to all features |
| `/admin/cameras` | Camera CRUD management |
| `/admin/knowledge` | Custom knowledge base: upload files or paste text |
| `/admin/analytics` | Search analytics and statistics |
| `/admin/drafts` | Draft auto-reply management |
| `/admin/support-cases` | Browse and search ingested Gmail support cases |
| `/admin/search-debug` | Debug RAG: view chunks, prompts, timing |
| `/admin/docs` | Technical documentation |
| `/admin/chunks` | Chunk editor: browse, search, edit, delete vectors |
| `/admin/graph` | Knowledge Graph Explorer: browse entities, edit/delete relationships |

Admin authentication uses secure session cookies (24-hour expiry). Set `ADMIN_PASSWORD` in your `.env` file.

## API Endpoints

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | RAG search with synthesis |
| POST | `/api/search/stream` | Streaming search (SSE) |

Both search endpoints support follow-up conversations via `session_id`:

```json
{
  "query": "What ports does it support?",
  "source": "both",
  "session_id": "uuid-from-previous-response"
}
```

Sessions maintain the last 3 Q&A exchanges for context. Session timeout: 30 minutes.

### Cameras (Public)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras` | List all cameras (with filters) |
| GET | `/api/cameras/search?q=` | Search cameras (FTS5 with BM25 ranking) |
| GET | `/api/cameras/{id}` | Get single camera |
| GET | `/api/cameras/{id}/related` | Get similar cameras |
| GET | `/api/cameras/stats` | Database statistics |
| POST | `/api/cameras/compare` | Compare multiple cameras (max 5) |
| GET | `/api/cameras/export.csv` | Export cameras as CSV |

### Cameras (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/cameras` | Create camera |
| PUT | `/api/admin/cameras/{id}` | Update camera |
| DELETE | `/api/admin/cameras/{id}` | Delete camera |
| POST | `/api/admin/cameras/import` | Import cameras from CSV |

### Analytics (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/search-stats?days=30` | Search statistics |
| GET | `/api/admin/search/{id}` | Get stored search details |

### Drafts (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/drafts/status` | Draft system status |
| GET | `/api/admin/drafts/pending` | List unanswered threads |
| GET | `/api/admin/drafts/thread/{id}` | Get thread with messages |
| POST | `/api/admin/drafts/preview/{id}` | Preview AI-generated draft |
| POST | `/api/admin/drafts/create/{id}` | Create draft in Gmail |
| POST | `/api/admin/drafts/run` | Run draft pipeline |

### Chunks (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/chunks` | List/search chunks (paginated) |
| GET | `/api/admin/chunks/{collection}/{id}` | Get chunk details |
| PUT | `/api/admin/chunks/{collection}/{id}` | Update chunk (re-embeds if text changed) |
| DELETE | `/api/admin/chunks/{collection}/{id}` | Delete chunk |

### Knowledge Base (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/knowledge` | List custom documents |
| GET | `/api/admin/knowledge/categories` | Get available categories |
| GET | `/api/admin/knowledge/{id}` | Get document by ID |
| POST | `/api/admin/knowledge` | Create document (JSON body) |
| POST | `/api/admin/knowledge/upload` | Upload file (.txt, .md, .pdf) |
| PUT | `/api/admin/knowledge/{id}` | Update document |
| DELETE | `/api/admin/knowledge/{id}` | Delete document |

### Support Cases (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/support-cases` | List cases (paginated) |
| GET | `/api/admin/support-cases/stats` | Statistics by category/product/quality |
| GET | `/api/admin/support-cases/search?q=` | FTS5 full-text search |
| GET | `/api/admin/support-cases/{id}` | Get case details |
| GET | `/api/admin/support-cases/{id}/raw-thread` | Get cleaned raw thread |
| DELETE | `/api/admin/support-cases/{id}` | Delete case |

### Graph (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/graph/stats` | Knowledge graph statistics |
| GET | `/api/admin/graph/entity-types` | List entity types with counts |
| GET | `/api/admin/graph/relationship-types` | List relationship types with counts |
| GET | `/api/admin/graph/entities` | List entities (paginated, filterable) |
| GET | `/api/admin/graph/entities/{type}/{id}` | Get entity with relationships |
| GET | `/api/admin/graph/relationships` | List relationships (filterable) |
| DELETE | `/api/admin/graph/relationships` | Delete a relationship |
| PATCH | `/api/admin/graph/relationships` | Update relationship type |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/login` | Login (returns session cookie) |
| POST | `/api/admin/logout` | Logout (clears cookie) |
| GET | `/api/admin/session` | Check session status |

## Tools RAG disponibles

L'agent dispose de 4 outils de recherche avec hybrid RRF (dense + sparse vectors) :

| Tool | Description |
|------|-------------|
| `search_docs` | Recherche dans la documentation officielle |
| `search_cases` | Recherche dans les cas de support Gmail |
| `search_custom` | Recherche dans les documents custom (admin-managed) |
| `hybrid_search` | Recherche combinee (docs + cases + custom) avec RRF fusion |

## Structure du Projet

```
clorag/
   src/clorag/
      main.py              # Point d'entree agent
      config.py            # Configuration Pydantic
      core/
         embeddings.py     # Client Voyage AI
         sparse_embeddings.py  # FastEmbed BM25
         vectorstore.py    # Client Qdrant
         graph_store.py    # Neo4j async client
         entity_extractor.py  # LLM entity extraction
         retriever.py      # Multi-source retriever
         database.py       # SQLite camera database
         analytics_db.py   # SQLite analytics database
         support_case_db.py  # SQLite support cases database
      agent/
         tools.py          # MCP tools RAG
         prompts.py        # System prompts
      analysis/
         thread_analyzer.py     # Haiku analysis
         quality_controller.py  # Sonnet QC
         camera_extractor.py    # LLM camera extraction
      models/
         support_case.py   # Data models
         camera.py         # Camera models
      drafts/
         gmail_service.py      # Gmail API with draft creation
         draft_generator.py    # RAG-based response generator
         draft_pipeline.py     # Draft creation orchestration
         models.py             # Draft data models
      ingestion/
         docusaurus.py     # Pipeline Docusaurus
         gmail.py          # Pipeline Gmail
         curated_gmail.py  # Pipeline curated
         chunker.py        # Text chunking
      graph/
         schema.py         # Graph entity models
         enrichment.py     # Context enrichment service
      web/
         app.py            # FastAPI application
         templates/        # Jinja2 templates
            index.html           # AI Search page
            cameras.html         # Camera compatibility
            help.html            # User guide
            admin_index.html     # Admin dashboard
            admin_login.html     # Admin login
            admin_cameras.html   # Camera management
            admin_knowledge.html # Knowledge base (file upload)
            admin_analytics.html # Analytics dashboard
            admin_drafts.html    # Draft management
            admin_support_cases.html # Support cases browser
            admin_search_debug.html  # Search debug
            admin_docs.html      # Technical documentation
            admin_chunks.html    # Chunk browser
            admin_chunk_edit.html # Chunk editor
            camera_edit.html     # Camera edit form
         static/           # CSS, JS assets
      scripts/
         ingest_docs.py    # CLI ingestion docs
         ingest_gmail.py   # CLI ingestion Gmail
         ingest_curated.py # CLI ingestion curated
         populate_graph.py # CLI graph population
         draft_support.py  # CLI draft creation
         run_web.py        # CLI run web server
   tests/
   data/
      clorag.db           # Camera database
      analytics.db        # Search analytics
```

## Setup Qdrant (Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  -e QDRANT__SERVICE__API_KEY=your_api_key \
  qdrant/qdrant
```

## Setup Neo4j (Docker) - Optional

Neo4j enables GraphRAG for knowledge graph enrichment:

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v $(pwd)/neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5-community
```

After starting Neo4j, populate the graph from Qdrant chunks:

```bash
uv run populate-graph
```

**Note**: GraphRAG is optional. The system gracefully degrades to vector-only search when Neo4j is not configured.

## Setup Gmail OAuth

1. Creer un projet sur [Google Cloud Console](https://console.cloud.google.com/)
2. Activer Gmail API
3. Creer OAuth 2.0 Client ID (Desktop app)
4. Telecharger `credentials.json` a la racine du projet
5. Au premier run, suivre le flow d'authentification

## Developpement

```bash
# Installer les dependances de developpement
uv sync --dev

# Linter
uv run ruff check src/

# Type checking
uv run mypy src/clorag --strict

# Tests
uv run pytest
```

## Deployment

### Docker

```bash
# Build and deploy
docker compose build
docker compose up -d

# Or manual deployment
rsync -avz --exclude '.venv' --exclude 'data' . root@server:/opt/clorag/
ssh root@server "cd /opt/clorag && docker compose build && docker compose up -d"
```

### Environment Variables for Production

Ensure all required environment variables are set in your deployment environment:

```bash
ANTHROPIC_API_KEY=...
VOYAGE_API_KEY=...
QDRANT_URL=...
ADMIN_PASSWORD=...  # Strong password for admin access
```

## Security Features

- **Session-based authentication** with signed cookies (itsdangerous)
- **Brute force protection** - 5 failed attempts triggers 5-minute lockout per IP
- **Rate limiting** on login (10/min) and admin API endpoints
- **XSS protection** with DOMPurify sanitization and HTML escaping
- **Open redirect prevention** on login redirects
- **HTTPS-only cookies** in production (Secure, HttpOnly, SameSite=Strict)
- **Timing-safe password comparison** to prevent timing attacks
- **OAuth token encryption** - Fernet encryption with PBKDF2 key derivation (480K iterations)
- **PII anonymization** - Customer data anonymized before LLM processing
- **SQL injection prevention** - Parameterized queries with column whitelist

For detailed security documentation, see [Admin Docs](/admin/docs#security) after authentication.

## Licence

MIT
