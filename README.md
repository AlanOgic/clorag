# CLORAG - Multi-RAG Agent avec Claude Agent SDK

Agent intelligent de support combinant documentation Docusaurus et cas de support Gmail via RAG (Retrieval-Augmented Generation).

**Features:**
- Hybrid RAG search (semantic + keyword) across documentation and support cases
- Camera Compatibility Database with automatic extraction from docs/support
- Web UI with AI-powered search and camera compatibility browser
- Admin interface for camera management

## Architecture

```
User Query -> Claude Agent -> MCP Tools -> RAG Retrieval -> Response
                                |
                     +----------+----------+
                     |          |          |
                search_docs  search_cases  hybrid_search
                     |          |          |
                 Voyage AI   Voyage AI   Voyage AI
              (voyage-context-3)         Embeddings
                     |          |          |
                  Qdrant      Qdrant      Qdrant
              (docs collection) (cases collection)  (both)
```

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Orchestration | Claude Agent SDK 0.1.9+ |
| Vector DB | Qdrant |
| Embeddings | Voyage AI (voyage-context-3) |
| Database | SQLite (camera + analytics) |
| Web | FastAPI + Jinja2 |
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
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
VOYAGE_API_KEY=your_voyage_key

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key  # optionnel

# Sources
DOCUSAURUS_URL=https://your-docs-site.com
GMAIL_LABEL=supports

# Camera Database (optional)
DATABASE_PATH=data/clorag.db
ADMIN_PASSWORD=your_admin_password
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

# Access:
# - AI Search: http://localhost:8080/
# - Camera Compatibility: http://localhost:8080/cameras
# - Admin Login: http://localhost:8080/admin/login
# - Admin Dashboard: http://localhost:8080/admin (requires login)
# - Admin Cameras: http://localhost:8080/admin/cameras
# - Admin Analytics: http://localhost:8080/admin/analytics
# - Search Debug: http://localhost:8080/admin/search-debug
```

Admin authentication uses secure session cookies (7-day expiry). Set `ADMIN_PASSWORD` in your `.env` file.

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
         embeddings.py    # Client Voyage AI
         vectorstore.py   # Client Qdrant
         retriever.py     # Multi-source retriever
         database.py      # SQLite camera database
      agent/
         tools.py         # MCP tools RAG
         prompts.py       # System prompts
      analysis/
         thread_analyzer.py     # Haiku analysis
         quality_controller.py  # Sonnet QC
         camera_extractor.py    # LLM camera extraction
      models/
         support_case.py  # Data models
         camera.py        # Camera models
      ingestion/
         docusaurus.py    # Pipeline Docusaurus
         gmail.py         # Pipeline Gmail
         curated_gmail.py # Pipeline curated
         chunker.py       # Text chunking
      web/
         app.py           # FastAPI application
         templates/       # Jinja2 templates
      scripts/
          ingest_docs.py   # CLI ingestion docs
          ingest_gmail.py  # CLI ingestion Gmail
          ingest_curated.py # CLI ingestion curated
   tests/
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

## Licence

MIT
