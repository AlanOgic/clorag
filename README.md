# CLORAG - Multi-RAG Agent avec Claude Agent SDK

Agent intelligent de support combinant documentation Docusaurus et cas de support Gmail via RAG (Retrieval-Augmented Generation).

## Features

- **AI-Powered Search** - Natural language queries with Claude Haiku 4.5 synthesis
- **Follow-up Conversations** - Ask follow-up questions with context from last 3 exchanges
- **Hybrid RAG Search** - Combines semantic (Voyage AI) and keyword (BM25) matching with RRF fusion
- **Camera Compatibility Database** - Structured camera info with automatic extraction from docs/support
- **Search Analytics** - Track popular queries, response times, and usage patterns
- **Session-Based Admin** - Secure login with signed cookies for all admin features
- **Streaming Responses** - Real-time answer streaming for better UX

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
│              Claude Haiku 4.5 (streaming)                  │
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
| Camera Database | Structured camera compatibility data | SQLite (relational) |

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Orchestration | Claude Agent SDK 0.1.9+ |
| Vector DB | Qdrant |
| Dense Embeddings | Voyage AI (voyage-context-3) |
| Sparse Embeddings | FastEmbed BM25 |
| LLM Synthesis | Claude Haiku 4.5 |
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
| `/admin/analytics` | Search analytics and statistics |
| `/admin/search-debug` | Debug RAG: view chunks, prompts, timing |
| `/admin/docs` | Technical documentation |

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
| GET | `/api/cameras/search?q=` | Search cameras |
| GET | `/api/cameras/{id}` | Get single camera |
| GET | `/api/cameras/stats` | Database statistics |

### Cameras (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/cameras` | Create camera |
| PUT | `/api/admin/cameras/{id}` | Update camera |
| DELETE | `/api/admin/cameras/{id}` | Delete camera |

### Analytics (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/search-stats?days=30` | Search statistics |
| GET | `/api/admin/search/{id}` | Get stored search details |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/login` | Login (returns session cookie) |
| POST | `/api/admin/logout` | Logout (clears cookie) |
| GET | `/api/admin/session` | Check session status |

## Tools RAG disponibles

L'agent dispose de 3 outils de recherche :

| Tool | Description |
|------|-------------|
| `search_docs` | Recherche dans la documentation officielle |
| `search_cases` | Recherche dans les cas de support Gmail |
| `hybrid_search` | Recherche combinee (docs + cases) |

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
         retriever.py      # Multi-source retriever
         database.py       # SQLite camera database
         analytics_db.py   # SQLite analytics database
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
      ingestion/
         docusaurus.py     # Pipeline Docusaurus
         gmail.py          # Pipeline Gmail
         curated_gmail.py  # Pipeline curated
         chunker.py        # Text chunking
      web/
         app.py            # FastAPI application
         templates/        # Jinja2 templates
            index.html           # AI Search page
            cameras.html         # Camera compatibility
            help.html            # User guide
            admin_index.html     # Admin dashboard
            admin_login.html     # Admin login
            admin_cameras.html   # Camera management
            admin_analytics.html # Analytics dashboard
            admin_search_debug.html  # Search debug
            admin_docs.html      # Technical documentation
            camera_edit.html     # Camera edit form
         static/           # CSS, JS assets
      scripts/
         ingest_docs.py    # CLI ingestion docs
         ingest_gmail.py   # CLI ingestion Gmail
         ingest_curated.py # CLI ingestion curated
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
- **Rate limiting** on login (5/min) and admin API endpoints
- **XSS protection** with HTML escaping and URL validation
- **Open redirect prevention** on login redirects
- **HTTPS-only cookies** in production (Secure, HttpOnly, SameSite=Strict)
- **Timing-safe password comparison** to prevent timing attacks

## Licence

MIT
