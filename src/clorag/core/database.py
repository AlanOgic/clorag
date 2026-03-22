"""SQLite database for camera compatibility data."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from clorag.config import get_settings
from clorag.core.cache import LRUCache
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate, DeviceType
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


# TTLCache is now an alias to LRUCache with TTL for backwards compatibility
TTLCache = LRUCache


class ConnectionPool:
    """Simple SQLite connection pool for thread-safe access."""

    def __init__(self, db_path: str, pool_size: int = 5) -> None:
        """Initialize connection pool.

        Args:
            db_path: Path to SQLite database.
            pool_size: Number of connections to maintain.
        """
        self._db_path = db_path
        self._pool_size = pool_size
        self._connections: list[sqlite3.Connection] = []
        self._in_use = 0  # Track connections currently checked out
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool."""
        conn: sqlite3.Connection | None = None
        with self._condition:
            # Wait if pool is empty AND we've hit max connections
            while not self._connections and self._in_use >= self._pool_size:
                self._condition.wait(timeout=5.0)

            if self._connections:
                conn = self._connections.pop()
            else:
                conn = self._create_connection()
            self._in_use += 1

        try:
            yield conn
        finally:
            with self._condition:
                self._in_use -= 1
                if len(self._connections) < self._pool_size:
                    self._connections.append(conn)
                else:
                    conn.close()
                self._condition.notify()

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            self._in_use = 0

# Whitelist of allowed columns for UPDATE operations (SQL injection protection)
ALLOWED_UPDATE_COLUMNS = frozenset(
    {
        "name",
        "manufacturer",
        "code_model",
        "device_type",
        "ports",
        "protocols",
        "supported_controls",
        "notes",
        "doc_url",
        "manufacturer_url",
        "confidence",
        "needs_review",
        "updated_at",
    }
)


class CameraDatabase:
    """SQLite database for camera compatibility management."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the database.

        Args:
            db_path: Path to SQLite database file. Defaults to settings.database_path.
        """
        settings = get_settings()
        self._db_path = db_path or settings.database_path

        # Ensure directory exists
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool and cache (sizes configurable via admin settings)
        try:
            from clorag.services.settings_manager import get_setting
            cache_size = int(get_setting("caches.camera_db_size"))
            cache_ttl = float(get_setting("caches.camera_db_ttl"))
        except (KeyError, ImportError, Exception):
            cache_size = 200
            cache_ttl = 300.0
        self._pool = ConnectionPool(self._db_path, pool_size=5)
        self._cache: LRUCache[Any] = LRUCache(max_size=cache_size, ttl_seconds=cache_ttl)

        self._ensure_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool."""
        with self._pool.get_connection() as conn:
            yield conn

    def close(self) -> None:
        """Close all database connections."""
        self._pool.close_all()
        self._cache.invalidate()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    manufacturer TEXT,
                    code_model TEXT,
                    ports TEXT DEFAULT '[]',
                    protocols TEXT DEFAULT '[]',
                    supported_controls TEXT DEFAULT '[]',
                    notes TEXT DEFAULT '[]',
                    source TEXT DEFAULT 'manual',
                    doc_url TEXT,
                    manufacturer_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_manufacturer ON cameras(manufacturer)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_name ON cameras(name COLLATE NOCASE)")

            # Add code_model column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE cameras ADD COLUMN code_model TEXT")
                conn.commit()
                logger.info("Added code_model column to cameras table")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add device_type column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE cameras ADD COLUMN device_type TEXT")
                conn.commit()
                logger.info("Added device_type column to cameras table")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Index on code_model for fast lookups (after migration ensures column exists)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_code_model ON cameras(code_model)")

            # Add confidence column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE cameras ADD COLUMN confidence REAL DEFAULT 1.0")
                conn.commit()
                logger.info("Added confidence column to cameras table")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add needs_review column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE cameras ADD COLUMN needs_review INTEGER DEFAULT 0")
                conn.commit()
                logger.info("Added needs_review column to cameras table")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Index on needs_review for fast filtering of review queue
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_needs_review ON cameras(needs_review)")

            # Covering index for common list queries (manufacturer + name sort)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cameras_list_cover
                ON cameras(manufacturer, name, id, device_type, ports, protocols)
            """)

            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cameras_fts USING fts5(
                    name,
                    manufacturer,
                    code_model,
                    ports,
                    protocols,
                    notes,
                    content='cameras',
                    content_rowid='id',
                    tokenize='porter unicode61'
                )
            """)

            # Triggers to keep FTS index in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS cameras_ai AFTER INSERT ON cameras BEGIN
                    INSERT INTO cameras_fts(rowid, name, manufacturer, code_model, ports, protocols, notes)
                    VALUES (new.id, new.name, new.manufacturer, new.code_model, new.ports, new.protocols, new.notes);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS cameras_ad AFTER DELETE ON cameras BEGIN
                    INSERT INTO cameras_fts(cameras_fts, rowid, name, manufacturer, code_model, ports, protocols, notes)
                    VALUES ('delete', old.id, old.name, old.manufacturer, old.code_model, old.ports, old.protocols, old.notes);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS cameras_au AFTER UPDATE ON cameras BEGIN
                    INSERT INTO cameras_fts(cameras_fts, rowid, name, manufacturer, code_model, ports, protocols, notes)
                    VALUES ('delete', old.id, old.name, old.manufacturer, old.code_model, old.ports, old.protocols, old.notes);
                    INSERT INTO cameras_fts(rowid, name, manufacturer, code_model, ports, protocols, notes)
                    VALUES (new.id, new.name, new.manufacturer, new.code_model, new.ports, new.protocols, new.notes);
                END
            """)

            conn.commit()
            logger.debug("Database schema ensured", db_path=self._db_path)

    def _row_to_camera(self, row: sqlite3.Row) -> Camera:
        """Convert a database row to a Camera model."""
        # Parse device_type if present
        device_type = None
        if "device_type" in row.keys() and row["device_type"]:
            try:
                device_type = DeviceType(row["device_type"])
            except ValueError:
                pass  # Invalid device_type value, leave as None

        # Parse confidence (handle missing column gracefully)
        confidence = 1.0
        if "confidence" in row.keys() and row["confidence"] is not None:
            confidence = float(row["confidence"])

        # Parse needs_review (stored as INTEGER 0/1)
        needs_review = False
        if "needs_review" in row.keys() and row["needs_review"]:
            needs_review = bool(row["needs_review"])

        return Camera(
            id=row["id"],
            name=row["name"],
            manufacturer=row["manufacturer"],
            code_model=row["code_model"] if "code_model" in row.keys() else None,
            device_type=device_type,
            ports=json.loads(row["ports"]) if row["ports"] else [],
            protocols=json.loads(row["protocols"]) if row["protocols"] else [],
            supported_controls=json.loads(row["supported_controls"]) if row["supported_controls"] else [],
            notes=json.loads(row["notes"]) if row["notes"] else [],
            source=CameraSource(row["source"]) if row["source"] else CameraSource.MANUAL,
            doc_url=row["doc_url"],
            manufacturer_url=row["manufacturer_url"],
            confidence=confidence,
            needs_review=needs_review,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def list_cameras(
        self,
        manufacturer: str | None = None,
        device_type: str | None = None,
        port: str | None = None,
        protocol: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[Camera]:
        """List all cameras, optionally filtered by manufacturer, device_type, port, and/or protocol.

        Args:
            manufacturer: Filter by manufacturer name (case-insensitive).
            device_type: Filter by device type (e.g., 'camera_cinema', 'lens').
            port: Filter by port (e.g., 'RS-422', 'Ethernet').
            protocol: Filter by protocol (e.g., 'VISCA', 'Sony RCP').
            offset: Number of records to skip (for pagination).
            limit: Maximum number of records to return (None for all).

        Returns:
            List of Camera objects.
        """
        # Check cache for common queries
        cache_key = f"list:{manufacturer}:{device_type}:{port}:{protocol}:{offset}:{limit}"
        cached: list[Camera] | None = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._get_connection() as conn:
            conditions = []
            params: list[str | int] = []

            if manufacturer:
                conditions.append("manufacturer = ? COLLATE NOCASE")
                params.append(manufacturer)
            if device_type:
                conditions.append("device_type = ?")
                params.append(device_type)
            if port:
                # JSON array contains check (SQLite LIKE for simplicity)
                conditions.append("ports LIKE ?")
                params.append(f'%"{port}"%')
            if protocol:
                conditions.append("protocols LIKE ?")
                params.append(f'%"{protocol}"%')

            query = "SELECT * FROM cameras"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY manufacturer, name"

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor = conn.execute(query, params)
            results = [self._row_to_camera(row) for row in cursor.fetchall()]

            # Cache the results
            self._cache.set(cache_key, results)
            return results

    def count_cameras(
        self,
        manufacturer: str | None = None,
        device_type: str | None = None,
        port: str | None = None,
        protocol: str | None = None,
    ) -> int:
        """Count cameras matching the given filters.

        Args:
            manufacturer: Filter by manufacturer name (case-insensitive).
            device_type: Filter by device type.
            port: Filter by port.
            protocol: Filter by protocol.

        Returns:
            Total count of matching cameras.
        """
        with self._get_connection() as conn:
            conditions = []
            params: list[str] = []

            if manufacturer:
                conditions.append("manufacturer = ? COLLATE NOCASE")
                params.append(manufacturer)
            if device_type:
                conditions.append("device_type = ?")
                params.append(device_type)
            if port:
                conditions.append("ports LIKE ?")
                params.append(f'%"{port}"%')
            if protocol:
                conditions.append("protocols LIKE ?")
                params.append(f'%"{protocol}"%')

            query = "SELECT COUNT(*) FROM cameras"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            result = conn.execute(query, params).fetchone()
            return int(result[0]) if result else 0

    def list_cameras_needing_review(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Camera]:
        """List cameras flagged for human review.

        Args:
            offset: Number of records to skip (for pagination).
            limit: Maximum number of records to return.

        Returns:
            List of Camera objects needing review, ordered by confidence (lowest first).
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM cameras
                WHERE needs_review = 1
                ORDER BY confidence ASC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            return [self._row_to_camera(row) for row in cursor.fetchall()]

    def count_cameras_needing_review(self) -> int:
        """Count cameras needing review.

        Returns:
            Total count of cameras flagged for review.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM cameras WHERE needs_review = 1"
            ).fetchone()
            return int(result[0]) if result else 0

    def approve_camera(self, camera_id: int) -> Camera | None:
        """Approve a camera (clear needs_review flag, set confidence to 1.0).

        Args:
            camera_id: Camera database ID.

        Returns:
            Updated Camera object or None if not found.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE cameras
                SET needs_review = 0, confidence = 1.0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (camera_id,),
            )
            conn.commit()
            logger.info("Approved camera", id=camera_id)

            # Invalidate caches
            self._cache.invalidate("list:")
            self._cache.invalidate("review:")

            return self.get_camera(camera_id)

    def get_camera(self, camera_id: int) -> Camera | None:
        """Get a single camera by ID.

        Args:
            camera_id: Camera database ID.

        Returns:
            Camera object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
            row = cursor.fetchone()
            return self._row_to_camera(row) if row else None

    def get_camera_by_name(self, name: str) -> Camera | None:
        """Get a single camera by name (case-insensitive).

        Args:
            name: Camera model name.

        Returns:
            Camera object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM cameras WHERE name = ? COLLATE NOCASE", (name,))
            row = cursor.fetchone()
            return self._row_to_camera(row) if row else None

    @staticmethod
    def _normalize_camera_name(name: str, manufacturer: str | None = None) -> str:
        """Normalize camera name for fuzzy matching and duplicate detection.

        Strips manufacturer-specific model line prefixes (AW-, ILME-, ILCE-, AG-)
        that are often omitted in casual references. Normalizes Mark synonyms.
        Does NOT strip product series prefixes (BRC, HDC, PXW, etc.).

        Examples:
            "FX-3" -> "fx3"
            "AW-UE160" -> "ue160"
            "ILME-FX6" -> "fx6"
            "Pyxis 6K" -> "pyxis6k"
            "BRC-AM7" -> "brcam7"  (BRC is part of the name)
            "HDC-5500 Mark II" -> "hdc5500mk2"
        """
        n = name.strip().lower()
        # Strip manufacturer name prefix if present (e.g. "Sony FX6" -> "FX6")
        if manufacturer:
            mfr_lower = manufacturer.lower()
            if n.startswith(mfr_lower + " "):
                n = n[len(mfr_lower) + 1:]
        # Strip manufacturer model-line prefixes (commonly omitted)
        # AW- (Panasonic), AG- (Panasonic), ILME- (Sony), ILCE- (Sony)
        n = re.sub(r"^(aw-|ag-|ilme-|ilce-)", "", n)
        # Normalize Mark synonyms
        n = re.sub(r"mk\s*iii|mark\s*iii|mark\s*3|mk\s*3", "mk3", n)
        n = re.sub(r"mk\s*ii|mark\s*ii|mark\s*2|mk\s*2", "mk2", n)
        n = re.sub(r"mk\s*i\b|mark\s*i\b|mk\s*1\b|mark\s*1\b", "mk1", n)
        # Strip common suffixes
        n = re.sub(r"\s+head$", "", n)
        # Remove hyphens, spaces, underscores, dots
        n = re.sub(r"[-_.\s]", "", n)
        return n

    def find_camera_by_similar_name(
        self, name: str, manufacturer: str | None = None
    ) -> Camera | None:
        """Find an existing camera by normalized name matching.

        Handles variations like FX-3/FX3, AW-UE160/UE160, ILME-FX6/FX6.

        Args:
            name: Camera name to search for.
            manufacturer: Optional manufacturer to narrow search.

        Returns:
            Best matching Camera or None.
        """
        normalized = self._normalize_camera_name(name, manufacturer)
        if not normalized:
            return None

        # Search all cameras (checking manufacturer-filtered first is redundant
        # since we iterate the full list and the cache covers both)
        all_cameras = self.list_cameras()
        for camera in all_cameras:
            if self._normalize_camera_name(camera.name, camera.manufacturer) == normalized:
                return camera

        return None

    def get_cameras_by_ids(self, camera_ids: list[int]) -> list[Camera]:
        """Get multiple cameras by their IDs.

        Args:
            camera_ids: List of camera database IDs.

        Returns:
            List of Camera objects (preserves order, excludes not found).
        """
        if not camera_ids:
            return []

        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(camera_ids))
            cursor = conn.execute(
                f"SELECT * FROM cameras WHERE id IN ({placeholders})",
                camera_ids,
            )
            # Build a dict for ordering
            cameras_by_id = {row["id"]: self._row_to_camera(row) for row in cursor.fetchall()}
            # Return in original order
            return [cameras_by_id[cid] for cid in camera_ids if cid in cameras_by_id]

    def find_related_cameras(
        self,
        camera_id: int,
        limit: int = 5,
    ) -> list[Camera]:
        """Find cameras with similar characteristics.

        Finds cameras that share the same manufacturer, device type, ports, or protocols.

        Args:
            camera_id: Reference camera ID.
            limit: Maximum number of related cameras to return.

        Returns:
            List of related Camera objects, scored by similarity.
        """
        camera = self.get_camera(camera_id)
        if not camera:
            return []

        with self._get_connection() as conn:
            # Score cameras by similarity factors
            # Higher score = more similar
            scores: dict[int, int] = {}

            # Same manufacturer (high weight)
            if camera.manufacturer:
                cursor = conn.execute(
                    "SELECT id FROM cameras WHERE manufacturer = ? AND id != ?",
                    (camera.manufacturer, camera_id),
                )
                for row in cursor.fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0) + 3

            # Same device type (high weight)
            if camera.device_type:
                cursor = conn.execute(
                    "SELECT id FROM cameras WHERE device_type = ? AND id != ?",
                    (camera.device_type.value, camera_id),
                )
                for row in cursor.fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0) + 3

            # Shared ports (medium weight) — single query instead of N+1
            if camera.ports:
                port_likes = [f'%"{p}"%' for p in camera.ports]
                match_expr = " + ".join(["(ports LIKE ?)"] * len(port_likes))
                where_expr = " OR ".join(["ports LIKE ?"] * len(port_likes))
                cursor = conn.execute(
                    f"SELECT id, ({match_expr}) as match_count FROM cameras WHERE ({where_expr}) AND id != ?",
                    port_likes + port_likes + [camera_id],
                )
                for row in cursor.fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0) + row["match_count"]

            # Shared protocols (medium weight) — single query instead of N+1
            if camera.protocols:
                proto_likes = [f'%"{p}"%' for p in camera.protocols]
                match_expr = " + ".join(["(protocols LIKE ?)"] * len(proto_likes))
                where_expr = " OR ".join(["protocols LIKE ?"] * len(proto_likes))
                cursor = conn.execute(
                    f"SELECT id, ({match_expr}) as match_count FROM cameras WHERE ({where_expr}) AND id != ?",
                    proto_likes + proto_likes + [camera_id],
                )
                for row in cursor.fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0) + row["match_count"]

            if not scores:
                return []

            # Sort by score (descending) and get top results
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
            return self.get_cameras_by_ids(sorted_ids)

    def create_camera(self, camera: CameraCreate, source: CameraSource = CameraSource.MANUAL) -> Camera:
        """Create a new camera entry.

        Args:
            camera: Camera data to create.
            source: Source of the camera information.

        Returns:
            Created Camera object with ID.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO cameras (name, manufacturer, code_model, device_type, ports, protocols,
                    supported_controls, notes, source, doc_url, manufacturer_url, confidence, needs_review)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera.name,
                    camera.manufacturer,
                    camera.code_model,
                    camera.device_type.value if camera.device_type else None,
                    json.dumps(camera.ports),
                    json.dumps(camera.protocols),
                    json.dumps(camera.supported_controls),
                    json.dumps(camera.notes),
                    source.value,
                    camera.doc_url,
                    camera.manufacturer_url,
                    camera.confidence,
                    1 if camera.needs_review else 0,
                ),
            )
            conn.commit()
            logger.info("Created camera", name=camera.name, id=cursor.lastrowid)

            # Invalidate list and search caches
            self._cache.invalidate("list:")
            self._cache.invalidate("search:")
            self._cache.invalidate("stats")

            return self.get_camera(cursor.lastrowid)  # type: ignore

    def update_camera(self, camera_id: int, updates: CameraUpdate) -> Camera | None:
        """Update an existing camera.

        Args:
            camera_id: Camera database ID.
            updates: Fields to update (None values are ignored).

        Returns:
            Updated Camera object or None if not found.
        """
        existing = self.get_camera(camera_id)
        if not existing:
            return None

        # Build update query for non-None fields
        update_fields: list[str] = []
        values: list[str | int] = []

        if updates.name is not None:
            update_fields.append("name = ?")
            values.append(updates.name)
        if updates.manufacturer is not None:
            update_fields.append("manufacturer = ?")
            values.append(updates.manufacturer)
        if updates.code_model is not None:
            update_fields.append("code_model = ?")
            values.append(updates.code_model)
        if updates.device_type is not None:
            update_fields.append("device_type = ?")
            values.append(updates.device_type.value)
        if updates.ports is not None:
            update_fields.append("ports = ?")
            values.append(json.dumps(updates.ports))
        if updates.protocols is not None:
            update_fields.append("protocols = ?")
            values.append(json.dumps(updates.protocols))
        if updates.supported_controls is not None:
            update_fields.append("supported_controls = ?")
            values.append(json.dumps(updates.supported_controls))
        if updates.notes is not None:
            update_fields.append("notes = ?")
            values.append(json.dumps(updates.notes))
        if updates.doc_url is not None:
            update_fields.append("doc_url = ?")
            values.append(updates.doc_url)
        if updates.manufacturer_url is not None:
            update_fields.append("manufacturer_url = ?")
            values.append(updates.manufacturer_url)

        if not update_fields:
            return existing

        # Validate all columns against whitelist (SQL injection protection)
        for field in update_fields:
            column_name = field.split(" = ")[0]
            if column_name not in ALLOWED_UPDATE_COLUMNS:
                raise ValueError(f"Invalid column name for update: {column_name}")

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(camera_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE cameras SET {', '.join(update_fields)} WHERE id = ?",
                values,
            )
            conn.commit()
            logger.info("Updated camera", id=camera_id)

            # Invalidate caches
            self._cache.invalidate("list:")
            self._cache.invalidate("search:")
            self._cache.invalidate("stats")

            return self.get_camera(camera_id)

    def delete_camera(self, camera_id: int) -> bool:
        """Delete a camera entry.

        Args:
            camera_id: Camera database ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted camera", id=camera_id)
                # Invalidate caches
                self._cache.invalidate("list:")
                self._cache.invalidate("search:")
                self._cache.invalidate("stats")
            return deleted

    def upsert_camera(self, camera: CameraCreate, source: CameraSource) -> Camera:
        """Insert or update camera by name.

        If camera exists (exact or fuzzy match), merges arrays.
        If camera doesn't exist, creates new entry.
        Fuzzy matching handles FX-3/FX3, AW-UE160/UE160, ILME-FX6/FX6 etc.

        Args:
            camera: Camera data.
            source: Source of the information.

        Returns:
            Created or updated Camera object.
        """
        # Try exact match first, then fuzzy match
        existing = self.get_camera_by_name(camera.name)
        if not existing:
            existing = self.find_camera_by_similar_name(camera.name, camera.manufacturer)
            if existing:
                logger.info(
                    "Fuzzy matched camera",
                    new_name=camera.name,
                    existing_name=existing.name,
                    existing_id=existing.id,
                )

        if existing:
            # Merge arrays - combine unique values
            merged_ports = list(set(existing.ports + camera.ports))
            merged_protocols = list(set(existing.protocols + camera.protocols))
            merged_controls = list(set(existing.supported_controls + camera.supported_controls))
            merged_notes = list(set(existing.notes + camera.notes))

            updates = CameraUpdate(
                manufacturer=camera.manufacturer or existing.manufacturer,
                code_model=camera.code_model or existing.code_model,
                ports=merged_ports,
                protocols=merged_protocols,
                supported_controls=merged_controls,
                notes=merged_notes,
                doc_url=camera.doc_url or existing.doc_url,
                manufacturer_url=camera.manufacturer_url or existing.manufacturer_url,
            )
            logger.info("Merging camera data", name=camera.name, source=source.value)
            return self.update_camera(existing.id, updates)  # type: ignore
        else:
            logger.info("Creating new camera from extraction", name=camera.name, source=source.value)
            return self.create_camera(camera, source)

    def get_manufacturers(self) -> list[str]:
        """Get list of unique manufacturers.

        Returns:
            List of manufacturer names.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT manufacturer FROM cameras WHERE manufacturer IS NOT NULL ORDER BY manufacturer"
            )
            return [row["manufacturer"] for row in cursor.fetchall()]

    def get_device_types(self) -> list[str]:
        """Get list of unique device types.

        Returns:
            List of device type values.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT device_type FROM cameras WHERE device_type IS NOT NULL ORDER BY device_type"
            )
            return [row["device_type"] for row in cursor.fetchall()]

    def get_all_ports(self) -> list[str]:
        """Get list of unique ports across all cameras.

        Returns:
            Sorted list of unique port names.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT ports FROM cameras WHERE ports != '[]'")
            all_ports: set[str] = set()
            for row in cursor.fetchall():
                ports = json.loads(row["ports"]) if row["ports"] else []
                all_ports.update(ports)
            return sorted(all_ports)

    def get_all_protocols(self) -> list[str]:
        """Get list of unique protocols across all cameras.

        Returns:
            Sorted list of unique protocol names.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT protocols FROM cameras WHERE protocols != '[]'")
            all_protocols: set[str] = set()
            for row in cursor.fetchall():
                protocols = json.loads(row["protocols"]) if row["protocols"] else []
                all_protocols.update(protocols)
            return sorted(all_protocols)

    def get_camera_by_code_model(self, code_model: str) -> Camera | None:
        """Get a camera by its code_model (case-insensitive).

        Args:
            code_model: Manufacturer model code to search for.

        Returns:
            Camera object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM cameras WHERE code_model = ? COLLATE NOCASE", (code_model,)
            )
            row = cursor.fetchone()
            return self._row_to_camera(row) if row else None

    def search_cameras(self, query: str, use_fts: bool = True) -> list[Camera]:
        """Search cameras using FTS5 full-text search.

        Args:
            query: Search query string.
            use_fts: Whether to use FTS5 (faster) or fallback to LIKE (more flexible).

        Returns:
            List of matching Camera objects, ranked by relevance.
        """
        if not query or not query.strip():
            return []

        # Check cache first
        cache_key = f"search:{query}:{use_fts}"
        cached: list[Camera] | None = self._cache.get(cache_key)
        if cached is not None:
            return cached

        query = query.strip()

        with self._get_connection() as conn:
            if use_fts:
                try:
                    # FTS5 search with BM25 ranking
                    # Escape special FTS5 characters and add wildcard suffix
                    fts_query = self._prepare_fts_query(query)
                    cursor = conn.execute(
                        """
                        SELECT c.*, bm25(cameras_fts) as rank
                        FROM cameras c
                        JOIN cameras_fts fts ON c.id = fts.rowid
                        WHERE cameras_fts MATCH ?
                        ORDER BY rank
                        LIMIT 100
                        """,
                        (fts_query,),
                    )
                    results = [self._row_to_camera(row) for row in cursor.fetchall()]
                    self._cache.set(cache_key, results)
                    return results
                except sqlite3.OperationalError as e:
                    # FTS table might not be populated yet, fall back to LIKE
                    logger.warning("FTS search failed, falling back to LIKE", error=str(e))

            # Fallback to LIKE search
            cursor = conn.execute(
                """
                SELECT * FROM cameras
                WHERE name LIKE ? COLLATE NOCASE
                   OR manufacturer LIKE ? COLLATE NOCASE
                   OR code_model LIKE ? COLLATE NOCASE
                ORDER BY name
                LIMIT 100
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            )
            results = [self._row_to_camera(row) for row in cursor.fetchall()]
            self._cache.set(cache_key, results)
            return results

    def _prepare_fts_query(self, query: str) -> str:
        """Prepare a query string for FTS5.

        Handles escaping and adds prefix matching for better UX.

        Args:
            query: Raw user query.

        Returns:
            FTS5-safe query string.
        """
        # Remove FTS5 special characters that could cause syntax errors
        special_chars = ['"', "'", "(", ")", "*", ":", "^", "-", "+"]
        clean_query = query
        for char in special_chars:
            clean_query = clean_query.replace(char, " ")

        # Split into terms and add prefix matching
        terms = clean_query.split()
        if not terms:
            return '""'  # Empty query

        # Add wildcard suffix to each term for prefix matching
        # This allows "son" to match "Sony"
        fts_terms = [f'"{term}"*' for term in terms if term]
        return " OR ".join(fts_terms)

    def rebuild_fts_index(self) -> int:
        """Rebuild the FTS5 index from the cameras table.

        Use this after bulk imports or if the FTS index gets out of sync.
        Will recreate the FTS table if it's corrupted.

        Returns:
            Number of cameras indexed.
        """
        with self._get_connection() as conn:
            try:
                # Try to delete all FTS data
                conn.execute("DELETE FROM cameras_fts")
            except sqlite3.DatabaseError:
                # FTS table is corrupted - drop and recreate it
                logger.warning("FTS table corrupted, recreating...")

                # Drop the corrupted table and triggers
                conn.execute("DROP TABLE IF EXISTS cameras_fts")
                conn.execute("DROP TRIGGER IF EXISTS cameras_ai")
                conn.execute("DROP TRIGGER IF EXISTS cameras_ad")
                conn.execute("DROP TRIGGER IF EXISTS cameras_au")
                conn.commit()

                # Recreate FTS5 table
                conn.execute("""
                    CREATE VIRTUAL TABLE cameras_fts USING fts5(
                        name,
                        manufacturer,
                        code_model,
                        ports,
                        protocols,
                        notes,
                        content='cameras',
                        content_rowid='id',
                        tokenize='porter unicode61'
                    )
                """)

                # Recreate triggers
                conn.execute("""
                    CREATE TRIGGER cameras_ai AFTER INSERT ON cameras BEGIN
                        INSERT INTO cameras_fts(rowid, name, manufacturer, code_model, ports, protocols, notes)
                        VALUES (new.id, new.name, new.manufacturer, new.code_model, new.ports, new.protocols, new.notes);
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER cameras_ad AFTER DELETE ON cameras BEGIN
                        INSERT INTO cameras_fts(cameras_fts, rowid, name, manufacturer, code_model, ports, protocols, notes)
                        VALUES ('delete', old.id, old.name, old.manufacturer, old.code_model, old.ports, old.protocols, old.notes);
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER cameras_au AFTER UPDATE ON cameras BEGIN
                        INSERT INTO cameras_fts(cameras_fts, rowid, name, manufacturer, code_model, ports, protocols, notes)
                        VALUES ('delete', old.id, old.name, old.manufacturer, old.code_model, old.ports, old.protocols, old.notes);
                        INSERT INTO cameras_fts(rowid, name, manufacturer, code_model, ports, protocols, notes)
                        VALUES (new.id, new.name, new.manufacturer, new.code_model, new.ports, new.protocols, new.notes);
                    END
                """)
                conn.commit()

            # Rebuild from cameras table
            conn.execute("""
                INSERT INTO cameras_fts(rowid, name, manufacturer, code_model, ports, protocols, notes)
                SELECT id, name, manufacturer, code_model, ports, protocols, notes
                FROM cameras
            """)
            conn.commit()

            # Get count
            count = conn.execute("SELECT COUNT(*) FROM cameras_fts").fetchone()[0]
            logger.info("Rebuilt FTS index", indexed_count=count)

            # Invalidate search cache
            self._cache.invalidate("search:")

            return int(count)

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with counts and stats.
        """
        # Check cache
        cache_key = "stats"
        cached: dict[str, Any] | None = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
            by_source = dict(
                conn.execute(
                    "SELECT source, COUNT(*) FROM cameras GROUP BY source"
                ).fetchall()
            )
            manufacturers = conn.execute(
                "SELECT COUNT(DISTINCT manufacturer) FROM cameras WHERE manufacturer IS NOT NULL"
            ).fetchone()[0]

            stats = {
                "total_cameras": total,
                "by_source": by_source,
                "manufacturers": manufacturers,
            }

            self._cache.set(cache_key, stats)
            return stats


    def clean_camera_names(self) -> int:
        """Remove manufacturer prefix from camera names.

        Fixes entries like "Sony HDC-5500" -> "HDC-5500" where
        the manufacturer is already stored separately.

        Returns:
            Number of cameras updated.
        """
        cameras = self.list_cameras()
        updated = 0

        for camera in cameras:
            if not camera.manufacturer or not camera.id:
                continue

            # Check if name starts with manufacturer (case-insensitive)
            manufacturer_lower = camera.manufacturer.lower()
            name_lower = camera.name.lower()

            if name_lower.startswith(manufacturer_lower + " "):
                # Remove manufacturer prefix
                new_name = camera.name[len(camera.manufacturer) + 1:].strip()

                if new_name:
                    # Check if the cleaned name already exists
                    existing = self.get_camera_by_name(new_name)
                    if existing and existing.id != camera.id:
                        # Name conflict - merge into existing and delete this one
                        logger.info(
                            "Merging duplicate camera",
                            old_name=camera.name,
                            new_name=new_name,
                            keeping_id=existing.id,
                        )
                        # Merge arrays
                        merged_ports = list(set(existing.ports + camera.ports))
                        merged_protocols = list(set(existing.protocols + camera.protocols))
                        merged_controls = list(set(existing.supported_controls + camera.supported_controls))
                        merged_notes = list(set(existing.notes + camera.notes))

                        updates = CameraUpdate(
                            ports=merged_ports,
                            protocols=merged_protocols,
                            supported_controls=merged_controls,
                            notes=merged_notes,
                            doc_url=camera.doc_url or existing.doc_url,
                            manufacturer_url=camera.manufacturer_url or existing.manufacturer_url,
                        )
                        self.update_camera(existing.id, updates)
                        self.delete_camera(camera.id)
                    else:
                        # Safe to rename
                        updates = CameraUpdate(name=new_name)
                        self.update_camera(camera.id, updates)
                        logger.info(
                            "Cleaned camera name",
                            old_name=camera.name,
                            new_name=new_name,
                        )
                    updated += 1

        logger.info("Camera name cleanup complete", updated=updated)
        return updated


    def merge_duplicate_cameras(self, dry_run: bool = False) -> int:
        """Merge cameras with duplicate code_model values.

        Keeps the camera with the friendlier name (shorter, no code prefix).
        Merges ports, protocols, controls, notes from duplicates.

        Args:
            dry_run: If True, only log what would be done.

        Returns:
            Number of cameras deleted.
        """
        deleted_count = 0

        # Find all duplicate code_models
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT code_model, GROUP_CONCAT(id) as ids
                FROM cameras
                WHERE code_model IS NOT NULL
                GROUP BY code_model
                HAVING COUNT(*) > 1
            """)
            duplicates = cursor.fetchall()

        for row in duplicates:
            code_model = row["code_model"]
            ids = [int(i) for i in row["ids"].split(",")]

            # Get all cameras with this code_model (batch query)
            cameras = self.get_cameras_by_ids(ids)

            if len(cameras) < 2:
                continue

            # Decide which camera to keep:
            # - Prefer shorter names (usually friendly names like "FX6" over "ILME-FX6")
            # - Prefer names that don't start with the code_model
            def score_camera(c: Camera) -> tuple[int, int, str]:
                name_lower = c.name.lower()
                code_lower = code_model.lower()
                # Higher score = keep this one
                is_code_name = name_lower == code_lower or name_lower.replace("-", "") == code_lower.replace("-", "")
                return (
                    0 if is_code_name else 1,  # Prefer non-code names
                    -len(c.name),  # Prefer shorter names
                    c.name,  # Alphabetical tiebreaker
                )

            cameras.sort(key=score_camera, reverse=True)
            keep_camera = cameras[0]
            delete_cameras = cameras[1:]

            # Merge data from duplicates into the camera we're keeping
            merged_ports = list(set(keep_camera.ports))
            merged_protocols = list(set(keep_camera.protocols))
            merged_controls = list(set(keep_camera.supported_controls))
            merged_notes = list(set(keep_camera.notes))

            for dup in delete_cameras:
                merged_ports.extend(dup.ports)
                merged_protocols.extend(dup.protocols)
                merged_controls.extend(dup.supported_controls)
                merged_notes.extend(dup.notes)

            merged_ports = list(set(merged_ports))
            merged_protocols = list(set(merged_protocols))
            merged_controls = list(set(merged_controls))
            merged_notes = list(set(merged_notes))

            if dry_run:
                logger.info(
                    "[DRY RUN] Would merge duplicates",
                    code_model=code_model,
                    keep=keep_camera.name,
                    delete=[c.name for c in delete_cameras],
                )
            else:
                # Update the kept camera with merged data
                updates = CameraUpdate(
                    ports=merged_ports,
                    protocols=merged_protocols,
                    supported_controls=merged_controls,
                    notes=merged_notes,
                    doc_url=keep_camera.doc_url or next((c.doc_url for c in delete_cameras if c.doc_url), None),
                    manufacturer_url=keep_camera.manufacturer_url or next((c.manufacturer_url for c in delete_cameras if c.manufacturer_url), None),
                )
                self.update_camera(keep_camera.id, updates)

                # Delete duplicates
                for dup in delete_cameras:
                    self.delete_camera(dup.id)
                    deleted_count += 1

                logger.info(
                    "Merged duplicates",
                    code_model=code_model,
                    kept=keep_camera.name,
                    deleted=[c.name for c in delete_cameras],
                )

        return deleted_count

    def find_duplicate_candidates(self) -> list[list[Camera]]:
        """Find groups of cameras that are likely duplicates.

        Normalizes camera names and groups by similarity:
        - Replaces known synonyms (Mark II → mk2, etc.)
        - Strips spaces/hyphens/dots for comparison
        - Cross-references code_model ↔ name matches

        Returns:
            List of camera groups (each group has 2+ cameras).
        """
        all_cameras = self.list_cameras()

        # Group by (normalized_name, manufacturer)
        groups: dict[tuple[str, str], list[Camera]] = defaultdict(list)
        for camera in all_cameras:
            assert camera.id is not None
            mfr_key = (camera.manufacturer or "").lower()
            key = (self._normalize_camera_name(camera.name, camera.manufacturer), mfr_key)
            groups[key].append(camera)

        # Also cross-reference: code_model of A matches name of B
        camera_by_name: dict[str, Camera] = {}
        for camera in all_cameras:
            camera_by_name[camera.name.lower()] = camera
            # Also index without manufacturer prefix
            if camera.manufacturer:
                stripped = camera.name.lower()
                mfr_lower = camera.manufacturer.lower()
                if stripped.startswith(mfr_lower + " "):
                    camera_by_name[stripped[len(mfr_lower) + 1:]] = camera

        code_model_merges: list[tuple[Camera, Camera]] = []
        for camera in all_cameras:
            if camera.code_model:
                match = camera_by_name.get(camera.code_model.lower())
                if match and match.id != camera.id:
                    code_model_merges.append((camera, match))

        # Merge code_model cross-references into groups
        # Build a union-find to merge groups that share cameras
        camera_to_group: dict[int, int] = {}  # camera_id → group_id
        result_groups: dict[int, set[int]] = {}  # group_id → set of camera_ids
        next_group_id = 0

        # First, add all normalized groups with 2+ cameras
        for group_cameras in groups.values():
            if len(group_cameras) >= 2:
                gid = next_group_id
                next_group_id += 1
                ids = {c.id for c in group_cameras if c.id is not None}
                result_groups[gid] = ids
                for cid in ids:
                    camera_to_group[cid] = gid

        # Then merge code_model cross-references
        for cam_a, cam_b in code_model_merges:
            assert cam_a.id is not None and cam_b.id is not None
            gid_a = camera_to_group.get(cam_a.id)
            gid_b = camera_to_group.get(cam_b.id)

            if gid_a is not None and gid_b is not None:
                if gid_a != gid_b:
                    # Merge groups
                    result_groups[gid_a].update(result_groups[gid_b])
                    for cid in result_groups[gid_b]:
                        camera_to_group[cid] = gid_a
                    del result_groups[gid_b]
            elif gid_a is not None:
                result_groups[gid_a].add(cam_b.id)
                camera_to_group[cam_b.id] = gid_a
            elif gid_b is not None:
                result_groups[gid_b].add(cam_a.id)
                camera_to_group[cam_a.id] = gid_b
            else:
                gid = next_group_id
                next_group_id += 1
                result_groups[gid] = {cam_a.id, cam_b.id}
                camera_to_group[cam_a.id] = gid
                camera_to_group[cam_b.id] = gid

        # Convert to lists of Camera objects
        camera_by_id = {c.id: c for c in all_cameras}
        result: list[list[Camera]] = []
        for ids in result_groups.values():
            group = [camera_by_id[cid] for cid in sorted(ids) if cid in camera_by_id]
            if len(group) >= 2:
                result.append(group)

        # Sort groups by first camera name for stable output
        result.sort(key=lambda g: g[0].name.lower())
        return result

    def merge_cameras(
        self,
        primary_id: int,
        merge_ids: list[int],
        custom_name: str | None = None,
    ) -> tuple[Camera, list[int], list[str]]:
        """Merge multiple cameras into a single primary camera.

        Union array fields, keep primary's scalars (fall back to merge targets),
        take max confidence, optionally apply custom name.

        Args:
            primary_id: ID of the camera to keep.
            merge_ids: IDs of cameras to merge into primary and then delete.
            custom_name: Optional custom name for the merged camera.

        Returns:
            Tuple of (updated primary camera, deleted IDs, deleted names).

        Raises:
            ValueError: If primary_id is in merge_ids or cameras not found.
        """
        if primary_id in merge_ids:
            raise ValueError("primary_id cannot be in merge_ids")

        primary = self.get_camera(primary_id)
        if not primary:
            raise ValueError(f"Primary camera {primary_id} not found")

        targets = self.get_cameras_by_ids(merge_ids)
        found_ids = {c.id for c in targets}
        missing = [mid for mid in merge_ids if mid not in found_ids]
        if missing:
            raise ValueError(f"Cameras not found: {missing}")

        # Union array fields
        all_ports: set[str] = set(primary.ports)
        all_protocols: set[str] = set(primary.protocols)
        all_controls: set[str] = set(primary.supported_controls)
        all_notes: set[str] = set(primary.notes)
        max_confidence = primary.confidence

        for target in targets:
            all_ports.update(target.ports)
            all_protocols.update(target.protocols)
            all_controls.update(target.supported_controls)
            all_notes.update(target.notes)
            max_confidence = max(max_confidence, target.confidence)

        # Scalar fallbacks: primary first, then first non-null from targets
        manufacturer = primary.manufacturer or next(
            (t.manufacturer for t in targets if t.manufacturer), None
        )
        code_model = primary.code_model or next(
            (t.code_model for t in targets if t.code_model), None
        )
        device_type = primary.device_type or next(
            (t.device_type for t in targets if t.device_type), None
        )
        doc_url = primary.doc_url or next(
            (t.doc_url for t in targets if t.doc_url), None
        )
        manufacturer_url = primary.manufacturer_url or next(
            (t.manufacturer_url for t in targets if t.manufacturer_url), None
        )

        # Build update
        updates = CameraUpdate(
            name=custom_name if custom_name else None,
            manufacturer=manufacturer,
            code_model=code_model,
            device_type=device_type,
            ports=sorted(all_ports),
            protocols=sorted(all_protocols),
            supported_controls=sorted(all_controls),
            notes=sorted(all_notes),
            doc_url=doc_url,
            manufacturer_url=manufacturer_url,
            confidence=max_confidence,
            needs_review=False,
        )

        self.update_camera(primary_id, updates)

        # Delete merge targets
        deleted_names: list[str] = []
        deleted_ids: list[int] = []
        for target in targets:
            assert target.id is not None
            deleted_names.append(target.name)
            deleted_ids.append(target.id)
            self.delete_camera(target.id)

        logger.info(
            "Merged cameras",
            primary_id=primary_id,
            deleted_ids=deleted_ids,
            deleted_names=deleted_names,
        )

        merged = self.get_camera(primary_id)
        assert merged is not None
        return merged, deleted_ids, deleted_names


@lru_cache(maxsize=1)
def get_camera_database() -> CameraDatabase:
    """Get or create the camera database instance (singleton via lru_cache)."""
    return CameraDatabase()
