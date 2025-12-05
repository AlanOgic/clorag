# Changelog

All notable changes to CLORAG will be documented in this file.

## [0.3.3] - 2025-12-05

### Added
- **Follow-up Conversations**
  - Server-side session storage with LRU cache (max 1000 sessions, 30-min TTL)
  - Conversation context includes last 3 Q&A exchanges for Claude synthesis
  - Session ID passed via `session_id` field in search requests/responses
  - Inline follow-up input appears below each AI answer for natural conversation flow
  - "New conversation" floating button to reset session and start fresh

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
