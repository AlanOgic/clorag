"""SQLite database for RAG settings management.

Stores settings, their metadata, and version history for audit and rollback.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from clorag.config import get_settings
from clorag.core.database import ConnectionPool

logger = structlog.get_logger(__name__)


# Singleton instance
_settings_db: SettingsDatabase | None = None


def get_settings_database() -> SettingsDatabase:
    """Get or create the singleton SettingsDatabase instance."""
    global _settings_db
    if _settings_db is None:
        _settings_db = SettingsDatabase()
    return _settings_db


# Valid categories for settings
SETTING_CATEGORIES = frozenset({"retrieval", "reranking", "synthesis", "caches", "prefetch"})

# Valid value types
SETTING_VALUE_TYPES = frozenset({"int", "float", "bool"})


class Setting:
    """Model representing a stored setting."""

    def __init__(
        self,
        id: str,
        key: str,
        name: str,
        description: str | None,
        category: str,
        value_type: str,
        value: str,
        default_value: str,
        min_value: float | None,
        max_value: float | None,
        requires_restart: bool,
        is_active: bool,
        created_at: datetime,
        updated_at: datetime,
        updated_by: str | None = None,
    ) -> None:
        """Initialize setting model."""
        self.id = id
        self.key = key
        self.name = name
        self.description = description
        self.category = category
        self.value_type = value_type
        self.value = value
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.requires_restart = requires_restart
        self.is_active = is_active
        self.created_at = created_at
        self.updated_at = updated_at
        self.updated_by = updated_by

    @property
    def typed_value(self) -> int | float | bool:
        """Convert the string value to the declared type.

        Returns:
            The value converted to int, float, or bool.

        Raises:
            ValueError: If the value cannot be converted.
        """
        return _parse_typed_value(self.value, self.value_type)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "value_type": self.value_type,
            "value": self.value,
            "typed_value": self.typed_value,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "requires_restart": self.requires_restart,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "updated_by": self.updated_by,
        }


class SettingVersion:
    """Model representing a historical version of a setting."""

    def __init__(
        self,
        id: str,
        setting_id: str,
        version: int,
        value: str,
        change_note: str | None,
        created_at: datetime,
        created_by: str | None = None,
    ) -> None:
        """Initialize version model."""
        self.id = id
        self.setting_id = setting_id
        self.version = version
        self.value = value
        self.change_note = change_note
        self.created_at = created_at
        self.created_by = created_by

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "setting_id": self.setting_id,
            "version": self.version,
            "value": self.value,
            "change_note": self.change_note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


def _parse_typed_value(value: str, value_type: str) -> int | float | bool:
    """Parse a string value into its declared type.

    Args:
        value: The string representation of the value.
        value_type: One of "int", "float", "bool".

    Returns:
        The parsed value.

    Raises:
        ValueError: If the value cannot be parsed as the declared type.
    """
    if value_type == "int":
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot parse '{value}' as int") from e
    elif value_type == "float":
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot parse '{value}' as float") from e
    elif value_type == "bool":
        if value.lower() in ("true", "1", "yes"):
            return True
        elif value.lower() in ("false", "0", "no"):
            return False
        else:
            raise ValueError(f"Cannot parse '{value}' as bool")
    else:
        raise ValueError(f"Unknown value_type: {value_type}")


def _validate_value(
    value: str,
    value_type: str,
    min_value: float | None,
    max_value: float | None,
) -> None:
    """Validate a value against its type and bounds.

    Args:
        value: The string value to validate.
        value_type: One of "int", "float", "bool".
        min_value: Optional minimum bound (for int/float).
        max_value: Optional maximum bound (for int/float).

    Raises:
        ValueError: If validation fails.
    """
    parsed = _parse_typed_value(value, value_type)

    # Check bounds for numeric types
    if value_type in ("int", "float"):
        numeric_value = float(parsed)
        if min_value is not None and numeric_value < min_value:
            raise ValueError(
                f"Value {parsed} is below minimum {min_value}"
            )
        if max_value is not None and numeric_value > max_value:
            raise ValueError(
                f"Value {parsed} is above maximum {max_value}"
            )


class SettingsDatabase:
    """SQLite database for managing RAG settings and their versions.

    Stores settings with metadata, supports version history for audit and rollback,
    and provides CRUD operations for settings management.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the settings database.

        Args:
            db_path: Path to SQLite database. Defaults to settings.database_path.
        """
        settings = get_settings()
        self._db_path = db_path or str(settings.database_path)

        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool (reuse pattern from CameraDatabase)
        self._pool = ConnectionPool(self._db_path, pool_size=5)

        # Initialize schema
        self._ensure_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool."""
        with self._pool.get_connection() as conn:
            yield conn

    def close(self) -> None:
        """Close all database connections."""
        self._pool.close_all()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            # Create settings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    default_value TEXT NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    requires_restart INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT
                )
            """)

            # Create setting versions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS setting_versions (
                    id TEXT PRIMARY KEY,
                    setting_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    change_note TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    FOREIGN KEY (setting_id) REFERENCES settings(id) ON DELETE CASCADE,
                    UNIQUE (setting_id, version)
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_setting_versions_setting_id "
                "ON setting_versions(setting_id)"
            )

            conn.commit()

        logger.info("Settings database schema initialized", path=self._db_path)

    def _row_to_setting(self, row: sqlite3.Row) -> Setting:
        """Convert a database row to a Setting object."""
        return Setting(
            id=row["id"],
            key=row["key"],
            name=row["name"],
            description=row["description"],
            category=row["category"],
            value_type=row["value_type"],
            value=row["value"],
            default_value=row["default_value"],
            min_value=row["min_value"],
            max_value=row["max_value"],
            requires_restart=bool(row["requires_restart"]),
            is_active=bool(row["is_active"]),
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now()
            ),
            updated_by=row["updated_by"],
        )

    def _row_to_version(self, row: sqlite3.Row) -> SettingVersion:
        """Convert a database row to a SettingVersion object."""
        return SettingVersion(
            id=row["id"],
            setting_id=row["setting_id"],
            version=row["version"],
            value=row["value"],
            change_note=row["change_note"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            created_by=row["created_by"],
        )

    def get_by_key(self, key: str) -> Setting | None:
        """Get a setting by its unique key.

        Args:
            key: The setting key (e.g., "retrieval.top_k").

        Returns:
            Setting object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM settings WHERE key = ? AND is_active = 1",
                (key,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_setting(row)

    def get_by_id(self, setting_id: str) -> Setting | None:
        """Get a setting by its ID.

        Args:
            setting_id: The setting UUID.

        Returns:
            Setting object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM settings WHERE id = ?",
                (setting_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_setting(row)

    def list_settings(
        self,
        category: str | None = None,
        include_inactive: bool = False,
    ) -> list[Setting]:
        """List all settings, optionally filtered by category.

        Args:
            category: Filter by category (retrieval, reranking, synthesis, caches, prefetch).
            include_inactive: Include deactivated settings.

        Returns:
            List of Setting objects.
        """
        conditions: list[str] = []
        params: list[str | int] = []

        if not include_inactive:
            conditions.append("is_active = 1")

        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM settings {where_clause} ORDER BY category, key",
                params,
            )
            rows = cursor.fetchall()

        return [self._row_to_setting(row) for row in rows]

    def update_setting(
        self,
        setting_id: str,
        value: str,
        change_note: str | None = None,
        updated_by: str | None = None,
    ) -> Setting | None:
        """Update a setting value and create a new version.

        Validates the value against the declared type and min/max bounds
        before persisting.

        Args:
            setting_id: The setting UUID.
            value: New value (as string).
            change_note: Note for version history.
            updated_by: Who made this update.

        Returns:
            Updated Setting object or None if not found.

        Raises:
            ValueError: If the value fails type or bounds validation.
        """
        existing = self.get_by_id(setting_id)
        if not existing:
            return None

        # Validate before saving
        _validate_value(value, existing.value_type, existing.min_value, existing.max_value)

        # Skip if value unchanged
        if existing.value == value:
            return existing

        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE settings SET value = ?, updated_at = ?, updated_by = ? WHERE id = ?",
                (value, now, updated_by, setting_id),
            )

            # Get latest version number
            cursor = conn.execute(
                "SELECT MAX(version) FROM setting_versions WHERE setting_id = ?",
                (setting_id,),
            )
            max_version = cursor.fetchone()[0] or 0
            new_version = max_version + 1

            # Create version entry
            version_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO setting_versions (
                    id, setting_id, version, value, change_note, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    setting_id,
                    new_version,
                    value,
                    change_note or f"Update to version {new_version}",
                    now,
                    updated_by,
                ),
            )

            conn.commit()

        logger.info(
            "Updated setting",
            id=setting_id,
            key=existing.key,
            new_version=new_version,
        )
        return self.get_by_id(setting_id)

    def upsert_setting(
        self,
        key: str,
        name: str,
        description: str | None,
        category: str,
        value_type: str,
        value: str,
        default_value: str,
        min_value: float | None = None,
        max_value: float | None = None,
        requires_restart: bool = False,
        created_by: str | None = None,
    ) -> Setting:
        """Create or update a setting by key.

        If the setting exists, updates only if something changed.
        Used by initialization to seed defaults without overwriting admin edits.

        Args:
            key: Unique key (e.g., "retrieval.top_k").
            name: Human-readable name.
            description: Optional description.
            category: Category (retrieval, reranking, synthesis, caches, prefetch).
            value_type: One of "int", "float", "bool".
            value: The value as string.
            default_value: The default value as string.
            min_value: Optional minimum bound.
            max_value: Optional maximum bound.
            requires_restart: Whether changes require a restart.
            created_by: Who created this setting.

        Returns:
            Setting object.

        Raises:
            ValueError: If category, value_type, or value is invalid.
        """
        if category not in SETTING_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {SETTING_CATEGORIES}")

        if value_type not in SETTING_VALUE_TYPES:
            raise ValueError(
                f"Invalid value_type: {value_type}. Must be one of {SETTING_VALUE_TYPES}"
            )

        # Validate the value
        _validate_value(value, value_type, min_value, max_value)

        existing = self.get_by_key(key)

        if existing:
            # Only update if something changed
            if (
                existing.value != value
                or existing.name != name
                or existing.description != description
                or existing.default_value != default_value
                or existing.min_value != min_value
                or existing.max_value != max_value
                or existing.requires_restart != requires_restart
            ):
                logger.debug("Updating existing setting", key=key)
                now = datetime.utcnow().isoformat()

                with self._get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE settings SET
                            name = ?, description = ?, default_value = ?,
                            min_value = ?, max_value = ?, requires_restart = ?,
                            updated_at = ?, updated_by = ?
                        WHERE id = ?
                        """,
                        (
                            name,
                            description,
                            default_value,
                            min_value,
                            max_value,
                            1 if requires_restart else 0,
                            now,
                            created_by,
                            existing.id,
                        ),
                    )
                    conn.commit()

                # If value changed, use update_setting to create a version
                if existing.value != value:
                    return self.update_setting(
                        setting_id=existing.id,
                        value=value,
                        change_note="Updated from defaults",
                        updated_by=created_by,
                    )  # type: ignore

                return self.get_by_id(existing.id)  # type: ignore

            return existing

        # Create new setting
        setting_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO settings (
                        id, key, name, description, category, value_type,
                        value, default_value, min_value, max_value,
                        requires_restart, is_active, created_at, updated_at, updated_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        setting_id,
                        key,
                        name,
                        description,
                        category,
                        value_type,
                        value,
                        default_value,
                        min_value,
                        max_value,
                        1 if requires_restart else 0,
                        now,
                        now,
                        created_by,
                    ),
                )

                # Create initial version (version 1)
                version_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO setting_versions (
                        id, setting_id, version, value, change_note, created_at, created_by
                    ) VALUES (?, ?, 1, ?, 'Initial version', ?, ?)
                    """,
                    (version_id, setting_id, value, now, created_by),
                )

                conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Setting with key '{key}' already exists") from e
                raise

        logger.info("Created setting", key=key, id=setting_id)
        return self.get_by_id(setting_id)  # type: ignore

    def update_metadata(
        self,
        setting_id: str,
        min_value: float | None = None,
        max_value: float | None = None,
        default_value: str | None = None,
        updated_by: str | None = None,
    ) -> Setting | None:
        """Update setting metadata (min, max, default) without changing the value.

        Args:
            setting_id: The setting UUID.
            min_value: New minimum bound (None to keep current).
            max_value: New maximum bound (None to keep current).
            default_value: New default value string (None to keep current).
            updated_by: Who made this update.

        Returns:
            Updated Setting object or None if not found.

        Raises:
            ValueError: If default_value fails type validation.
        """
        existing = self.get_by_id(setting_id)
        if not existing:
            return None

        new_min = min_value if min_value is not None else existing.min_value
        new_max = max_value if max_value is not None else existing.max_value
        new_default = default_value if default_value is not None else existing.default_value

        # Validate default value against type and new bounds
        if default_value is not None:
            _validate_value(new_default, existing.value_type, new_min, new_max)

        # Validate current value against new bounds
        _validate_value(existing.value, existing.value_type, new_min, new_max)

        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """UPDATE settings
                   SET min_value = ?, max_value = ?, default_value = ?,
                       updated_at = ?, updated_by = ?
                   WHERE id = ?""",
                (new_min, new_max, new_default, now, updated_by, setting_id),
            )
            conn.commit()

        return self.get_by_id(setting_id)

    def get_versions(self, setting_id: str) -> list[SettingVersion]:
        """Get all versions of a setting.

        Args:
            setting_id: The setting UUID.

        Returns:
            List of SettingVersion objects, newest first.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM setting_versions
                WHERE setting_id = ?
                ORDER BY version DESC
                """,
                (setting_id,),
            )
            rows = cursor.fetchall()

        return [self._row_to_version(row) for row in rows]

    def rollback_to_version(
        self,
        setting_id: str,
        version: int,
        rolled_back_by: str | None = None,
    ) -> Setting | None:
        """Rollback a setting to a previous version.

        Creates a new version with the old value.

        Args:
            setting_id: The setting UUID.
            version: Version number to rollback to.
            rolled_back_by: Who performed the rollback.

        Returns:
            Updated Setting object or None if not found.
        """
        # Get the version value
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM setting_versions WHERE setting_id = ? AND version = ?",
                (setting_id, version),
            )
            row = cursor.fetchone()

        if not row:
            logger.warning("Version not found for rollback", setting_id=setting_id, version=version)
            return None

        old_value = row["value"]

        # Update setting with old value (creates new version)
        return self.update_setting(
            setting_id=setting_id,
            value=old_value,
            change_note=f"Rollback to version {version}",
            updated_by=rolled_back_by,
        )

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get statistics about stored settings.

        Returns:
            Dict with stats (total, by_category, versions_count).
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM settings WHERE is_active = 1")
            total = cursor.fetchone()["total"]

            cursor = conn.execute(
                """
                SELECT category, COUNT(*) as count
                FROM settings
                WHERE is_active = 1
                GROUP BY category
                ORDER BY count DESC
                """
            )
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

            cursor = conn.execute("SELECT COUNT(*) as total FROM setting_versions")
            versions_count = cursor.fetchone()["total"]

        return {
            "total": total,
            "by_category": by_category,
            "versions_count": versions_count,
        }
