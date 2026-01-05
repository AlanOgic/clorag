# Changelog

All notable changes to CLORAG will be documented in this file.

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
