"""SQLite database for camera compatibility data."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from clorag.config import get_settings
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


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

        self._ensure_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    manufacturer TEXT,
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
            conn.commit()
            logger.debug("Database schema ensured", db_path=self._db_path)

    def _row_to_camera(self, row: sqlite3.Row) -> Camera:
        """Convert a database row to a Camera model."""
        return Camera(
            id=row["id"],
            name=row["name"],
            manufacturer=row["manufacturer"],
            ports=json.loads(row["ports"]) if row["ports"] else [],
            protocols=json.loads(row["protocols"]) if row["protocols"] else [],
            supported_controls=json.loads(row["supported_controls"]) if row["supported_controls"] else [],
            notes=json.loads(row["notes"]) if row["notes"] else [],
            source=CameraSource(row["source"]) if row["source"] else CameraSource.MANUAL,
            doc_url=row["doc_url"],
            manufacturer_url=row["manufacturer_url"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def list_cameras(self, manufacturer: str | None = None) -> list[Camera]:
        """List all cameras, optionally filtered by manufacturer.

        Args:
            manufacturer: Filter by manufacturer name (case-insensitive).

        Returns:
            List of Camera objects.
        """
        with self._get_connection() as conn:
            if manufacturer:
                cursor = conn.execute(
                    "SELECT * FROM cameras WHERE manufacturer = ? COLLATE NOCASE ORDER BY name",
                    (manufacturer,),
                )
            else:
                cursor = conn.execute("SELECT * FROM cameras ORDER BY manufacturer, name")
            return [self._row_to_camera(row) for row in cursor.fetchall()]

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
                INSERT INTO cameras (name, manufacturer, ports, protocols, supported_controls, notes, source, doc_url, manufacturer_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera.name,
                    camera.manufacturer,
                    json.dumps(camera.ports),
                    json.dumps(camera.protocols),
                    json.dumps(camera.supported_controls),
                    json.dumps(camera.notes),
                    source.value,
                    camera.doc_url,
                    camera.manufacturer_url,
                ),
            )
            conn.commit()
            logger.info("Created camera", name=camera.name, id=cursor.lastrowid)
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
        update_fields = []
        values = []

        if updates.name is not None:
            update_fields.append("name = ?")
            values.append(updates.name)
        if updates.manufacturer is not None:
            update_fields.append("manufacturer = ?")
            values.append(updates.manufacturer)
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

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(camera_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE cameras SET {', '.join(update_fields)} WHERE id = ?",
                values,
            )
            conn.commit()
            logger.info("Updated camera", id=camera_id)
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
            return deleted

    def upsert_camera(self, camera: CameraCreate, source: CameraSource) -> Camera:
        """Insert or update camera by name.

        If camera exists, merges arrays (ports, protocols, controls, notes).
        If camera doesn't exist, creates new entry.

        Args:
            camera: Camera data.
            source: Source of the information.

        Returns:
            Created or updated Camera object.
        """
        existing = self.get_camera_by_name(camera.name)

        if existing:
            # Merge arrays - combine unique values
            merged_ports = list(set(existing.ports + camera.ports))
            merged_protocols = list(set(existing.protocols + camera.protocols))
            merged_controls = list(set(existing.supported_controls + camera.supported_controls))
            merged_notes = list(set(existing.notes + camera.notes))

            updates = CameraUpdate(
                manufacturer=camera.manufacturer or existing.manufacturer,
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

    def search_cameras(self, query: str) -> list[Camera]:
        """Search cameras by name or manufacturer.

        Args:
            query: Search query string.

        Returns:
            List of matching Camera objects.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM cameras
                WHERE name LIKE ? COLLATE NOCASE
                   OR manufacturer LIKE ? COLLATE NOCASE
                ORDER BY name
                """,
                (f"%{query}%", f"%{query}%"),
            )
            return [self._row_to_camera(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with counts and stats.
        """
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

            return {
                "total_cameras": total,
                "by_source": by_source,
                "manufacturers": manufacturers,
            }


# Singleton instance
_database: CameraDatabase | None = None


def get_camera_database() -> CameraDatabase:
    """Get or create the camera database instance."""
    global _database
    if _database is None:
        _database = CameraDatabase()
    return _database
