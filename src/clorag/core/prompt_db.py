"""SQLite database for LLM prompt management.

Stores prompts, their metadata, and version history for audit and rollback.
"""

from __future__ import annotations

import json
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
_prompt_db: PromptDatabase | None = None


def get_prompt_database() -> PromptDatabase:
    """Get or create the singleton PromptDatabase instance."""
    global _prompt_db
    if _prompt_db is None:
        _prompt_db = PromptDatabase()
    return _prompt_db


class Prompt:
    """Model representing a stored prompt."""

    def __init__(
        self,
        id: str,
        key: str,
        name: str,
        description: str | None,
        model: str | None,
        content: str,
        variables: list[str],
        category: str,
        is_active: bool,
        created_at: datetime,
        updated_at: datetime,
        created_by: str | None = None,
    ) -> None:
        """Initialize prompt model."""
        self.id = id
        self.key = key
        self.name = name
        self.description = description
        self.model = model
        self.content = content
        self.variables = variables
        self.category = category
        self.is_active = is_active
        self.created_at = created_at
        self.updated_at = updated_at
        self.created_by = created_by

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "content": self.content,
            "variables": self.variables,
            "category": self.category,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
        }


class PromptVersion:
    """Model representing a historical version of a prompt."""

    def __init__(
        self,
        id: str,
        prompt_id: str,
        version: int,
        content: str,
        change_note: str | None,
        created_at: datetime,
        created_by: str | None = None,
    ) -> None:
        """Initialize version model."""
        self.id = id
        self.prompt_id = prompt_id
        self.version = version
        self.content = content
        self.change_note = change_note
        self.created_at = created_at
        self.created_by = created_by

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "change_note": self.change_note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


# Valid categories for prompts
PROMPT_CATEGORIES = frozenset({"agent", "analysis", "synthesis", "drafts", "graph", "scripts"})


class PromptDatabase:
    """SQLite database for managing LLM prompts and their versions.

    Stores prompts with metadata, supports version history for audit and rollback,
    and provides CRUD operations for prompt management.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the prompt database.

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
            # Create prompts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT,
                    model TEXT,
                    content TEXT NOT NULL,
                    variables TEXT DEFAULT '[]',
                    category TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT
                )
            """)

            # Create prompt versions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    change_note TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
                    UNIQUE (prompt_id, version)
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_key ON prompts(key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id "
                "ON prompt_versions(prompt_id)"
            )

            conn.commit()

        logger.info("Prompt database schema initialized", path=self._db_path)

    def _row_to_prompt(self, row: sqlite3.Row) -> Prompt:
        """Convert a database row to a Prompt object."""
        variables = json.loads(row["variables"]) if row["variables"] else []

        return Prompt(
            id=row["id"],
            key=row["key"],
            name=row["name"],
            description=row["description"],
            model=row["model"],
            content=row["content"],
            variables=variables,
            category=row["category"],
            is_active=bool(row["is_active"]),
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now()
            ),
            created_by=row["created_by"],
        )

    def _row_to_version(self, row: sqlite3.Row) -> PromptVersion:
        """Convert a database row to a PromptVersion object."""
        return PromptVersion(
            id=row["id"],
            prompt_id=row["prompt_id"],
            version=row["version"],
            content=row["content"],
            change_note=row["change_note"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            created_by=row["created_by"],
        )

    def get_prompt_by_key(self, key: str) -> Prompt | None:
        """Get a prompt by its unique key.

        Args:
            key: The prompt key (e.g., "analysis.thread_analyzer").

        Returns:
            Prompt object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM prompts WHERE key = ? AND is_active = 1",
                (key,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_prompt(row)

    def get_prompt_by_id(self, prompt_id: str) -> Prompt | None:
        """Get a prompt by its ID.

        Args:
            prompt_id: The prompt UUID.

        Returns:
            Prompt object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM prompts WHERE id = ?",
                (prompt_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_prompt(row)

    def list_prompts(
        self,
        category: str | None = None,
        include_inactive: bool = False,
    ) -> list[Prompt]:
        """List all prompts, optionally filtered by category.

        Args:
            category: Filter by category (agent, analysis, synthesis, drafts, graph, scripts).
            include_inactive: Include deactivated prompts.

        Returns:
            List of Prompt objects.
        """
        conditions = []
        params: list[str | int] = []

        if not include_inactive:
            conditions.append("is_active = 1")

        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM prompts {where_clause} ORDER BY category, key",
                params,
            )
            rows = cursor.fetchall()

        return [self._row_to_prompt(row) for row in rows]

    def create_prompt(
        self,
        key: str,
        name: str,
        content: str,
        category: str,
        description: str | None = None,
        model: str | None = None,
        variables: list[str] | None = None,
        created_by: str | None = None,
    ) -> Prompt:
        """Create a new prompt.

        Args:
            key: Unique key (e.g., "analysis.thread_analyzer").
            name: Human-readable name.
            content: The prompt text.
            category: Category (agent, analysis, synthesis, drafts, graph, scripts).
            description: Optional description.
            model: Target model (sonnet, claude).
            variables: List of variable names in the prompt.
            created_by: Who created this prompt.

        Returns:
            Created Prompt object.

        Raises:
            ValueError: If category is invalid or key already exists.
        """
        if category not in PROMPT_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {PROMPT_CATEGORIES}")

        prompt_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO prompts (
                        id, key, name, description, model, content,
                        variables, category, is_active, created_at, updated_at, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        prompt_id,
                        key,
                        name,
                        description,
                        model,
                        content,
                        json.dumps(variables or []),
                        category,
                        now,
                        now,
                        created_by,
                    ),
                )

                # Create initial version (version 1)
                version_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO prompt_versions (
                        id, prompt_id, version, content, change_note, created_at, created_by
                    ) VALUES (?, ?, 1, ?, 'Initial version', ?, ?)
                    """,
                    (version_id, prompt_id, content, now, created_by),
                )

                conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Prompt with key '{key}' already exists") from e
                raise

        logger.info("Created prompt", key=key, id=prompt_id)
        return self.get_prompt_by_id(prompt_id)  # type: ignore

    def update_prompt(
        self,
        prompt_id: str,
        content: str | None = None,
        name: str | None = None,
        description: str | None = None,
        model: str | None = None,
        variables: list[str] | None = None,
        is_active: bool | None = None,
        change_note: str | None = None,
        updated_by: str | None = None,
    ) -> Prompt | None:
        """Update a prompt and create a new version if content changed.

        Args:
            prompt_id: The prompt UUID.
            content: New prompt content (creates a version if changed).
            name: New name.
            description: New description.
            model: New target model.
            variables: New variable list.
            is_active: Activate/deactivate.
            change_note: Note for version history.
            updated_by: Who made this update.

        Returns:
            Updated Prompt object or None if not found.
        """
        existing = self.get_prompt_by_id(prompt_id)
        if not existing:
            return None

        now = datetime.utcnow().isoformat()
        updates: list[str] = ["updated_at = ?"]
        params: list[str | int] = [now]

        # Track if content changed (to create version)
        content_changed = False

        if content is not None and content != existing.content:
            updates.append("content = ?")
            params.append(content)
            content_changed = True

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if model is not None:
            updates.append("model = ?")
            params.append(model)

        if variables is not None:
            updates.append("variables = ?")
            params.append(json.dumps(variables))

        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)

        params.append(prompt_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE prompts SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            # Create new version if content changed
            if content_changed and content:
                # Get latest version number
                cursor = conn.execute(
                    "SELECT MAX(version) FROM prompt_versions WHERE prompt_id = ?",
                    (prompt_id,),
                )
                max_version = cursor.fetchone()[0] or 0
                new_version = max_version + 1

                version_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO prompt_versions (
                        id, prompt_id, version, content, change_note, created_at, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        prompt_id,
                        new_version,
                        content,
                        change_note or f"Update to version {new_version}",
                        now,
                        updated_by,
                    ),
                )

            conn.commit()

        logger.info(
            "Updated prompt",
            id=prompt_id,
            content_changed=content_changed,
            new_version=new_version if content_changed else None,
        )
        return self.get_prompt_by_id(prompt_id)

    def get_prompt_versions(self, prompt_id: str) -> list[PromptVersion]:
        """Get all versions of a prompt.

        Args:
            prompt_id: The prompt UUID.

        Returns:
            List of PromptVersion objects, newest first.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM prompt_versions
                WHERE prompt_id = ?
                ORDER BY version DESC
                """,
                (prompt_id,),
            )
            rows = cursor.fetchall()

        return [self._row_to_version(row) for row in rows]

    def rollback_to_version(
        self,
        prompt_id: str,
        version: int,
        rolled_back_by: str | None = None,
    ) -> Prompt | None:
        """Rollback a prompt to a previous version.

        Creates a new version with the old content.

        Args:
            prompt_id: The prompt UUID.
            version: Version number to rollback to.
            rolled_back_by: Who performed the rollback.

        Returns:
            Updated Prompt object or None if not found.
        """
        # Get the version content
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT content FROM prompt_versions WHERE prompt_id = ? AND version = ?",
                (prompt_id, version),
            )
            row = cursor.fetchone()

        if not row:
            logger.warning("Version not found for rollback", prompt_id=prompt_id, version=version)
            return None

        old_content = row["content"]

        # Update prompt with old content (creates new version)
        return self.update_prompt(
            prompt_id=prompt_id,
            content=old_content,
            change_note=f"Rollback to version {version}",
            updated_by=rolled_back_by,
        )

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt and all its versions.

        Args:
            prompt_id: The prompt UUID.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            # Delete versions first (CASCADE should handle this, but be explicit)
            conn.execute("DELETE FROM prompt_versions WHERE prompt_id = ?", (prompt_id,))
            cursor = conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted prompt", id=prompt_id)

        return deleted

    def upsert_prompt(
        self,
        key: str,
        name: str,
        content: str,
        category: str,
        description: str | None = None,
        model: str | None = None,
        variables: list[str] | None = None,
        created_by: str | None = None,
    ) -> Prompt:
        """Create or update a prompt by key.

        If prompt exists, updates only if content is different.
        Used by initialization to seed defaults without overwriting admin edits.

        Args:
            key: Unique key.
            name: Human-readable name.
            content: The prompt text.
            category: Category.
            description: Optional description.
            model: Target model.
            variables: Variable names.
            created_by: Creator.

        Returns:
            Prompt object.
        """
        existing = self.get_prompt_by_key(key)

        if existing:
            # Only update if something changed
            if (
                existing.content != content
                or existing.name != name
                or existing.description != description
                or existing.model != model
            ):
                logger.debug("Updating existing prompt", key=key)
                return self.update_prompt(
                    prompt_id=existing.id,
                    content=content,
                    name=name,
                    description=description,
                    model=model,
                    variables=variables,
                    change_note="Updated from defaults",
                    updated_by=created_by,
                )  # type: ignore
            return existing

        return self.create_prompt(
            key=key,
            name=name,
            content=content,
            category=category,
            description=description,
            model=model,
            variables=variables,
            created_by=created_by,
        )

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get statistics about stored prompts.

        Returns:
            Dict with stats (total, by_category, versions_count).
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM prompts WHERE is_active = 1")
            total = cursor.fetchone()["total"]

            cursor = conn.execute(
                """
                SELECT category, COUNT(*) as count
                FROM prompts
                WHERE is_active = 1
                GROUP BY category
                ORDER BY count DESC
                """
            )
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

            cursor = conn.execute("SELECT COUNT(*) as total FROM prompt_versions")
            versions_count = cursor.fetchone()["total"]

        return {
            "total": total,
            "by_category": by_category,
            "versions_count": versions_count,
        }
