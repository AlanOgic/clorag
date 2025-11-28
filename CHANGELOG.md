# Changelog

All notable changes to CLORAG will be documented in this file.

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
