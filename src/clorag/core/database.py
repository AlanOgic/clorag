"""SQLite database for camera compatibility data."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Generator

from clorag.config import get_settings
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate, DeviceType
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

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
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def list_cameras(
        self,
        manufacturer: str | None = None,
        device_type: str | None = None,
        port: str | None = None,
        protocol: str | None = None,
    ) -> list[Camera]:
        """List all cameras, optionally filtered by manufacturer, device_type, port, and/or protocol.

        Args:
            manufacturer: Filter by manufacturer name (case-insensitive).
            device_type: Filter by device type (e.g., 'camera_cinema', 'lens').
            port: Filter by port (e.g., 'RS-422', 'Ethernet').
            protocol: Filter by protocol (e.g., 'VISCA', 'Sony RCP').

        Returns:
            List of Camera objects.
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

            cursor = conn.execute(query, params)
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
                INSERT INTO cameras (name, manufacturer, code_model, device_type, ports, protocols, supported_controls, notes, source, doc_url, manufacturer_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

            # Get all cameras with this code_model
            cameras = [self.get_camera(cid) for cid in ids]
            cameras = [c for c in cameras if c is not None]

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


@lru_cache(maxsize=1)
def get_camera_database() -> CameraDatabase:
    """Get or create the camera database instance (singleton via lru_cache)."""
    return CameraDatabase()
