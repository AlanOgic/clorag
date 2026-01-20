"""SQLite database for storing terminology fix suggestions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog

from clorag.config import get_settings
from clorag.core.database import ConnectionPool

logger = structlog.get_logger(__name__)

# Type alias for suggestion status
SuggestionStatus = Literal["pending", "approved", "rejected", "applied"]

# Singleton instance
_terminology_db: TerminologyFixDatabase | None = None


def get_terminology_fix_database() -> TerminologyFixDatabase:
    """Get or create the singleton TerminologyFixDatabase instance."""
    global _terminology_db
    if _terminology_db is None:
        _terminology_db = TerminologyFixDatabase()
    return _terminology_db


class TerminologyFix:
    """Represents a terminology fix suggestion."""

    def __init__(
        self,
        id: str,
        chunk_id: str,
        collection: str,
        original_text: str,
        suggested_text: str,
        suggestion_type: str,
        confidence: float,
        reasoning: str,
        status: SuggestionStatus = "pending",
        created_at: datetime | None = None,
        applied_at: datetime | None = None,
    ) -> None:
        self.id = id
        self.chunk_id = chunk_id
        self.collection = collection
        self.original_text = original_text
        self.suggested_text = suggested_text
        self.suggestion_type = suggestion_type
        self.confidence = confidence
        self.reasoning = reasoning
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.applied_at = applied_at

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "collection": self.collection,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "suggestion_type": self.suggestion_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }


class TerminologyFixDatabase:
    """SQLite database for storing terminology fix suggestions.

    Stores suggestions for RIO terminology corrections, allowing human
    review before applying changes to vector database chunks.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the terminology fix database.

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

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor with auto-commit."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS terminology_fixes (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    suggested_text TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    applied_at TEXT
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_terminology_fixes_status
                ON terminology_fixes(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_terminology_fixes_collection
                ON terminology_fixes(collection)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_terminology_fixes_chunk_id
                ON terminology_fixes(chunk_id)
            """)

        logger.info("Terminology fix database schema initialized", path=self._db_path)

    def insert_fix(self, fix: TerminologyFix) -> str:
        """Insert a new terminology fix suggestion.

        Args:
            fix: The terminology fix to insert.

        Returns:
            The fix ID.
        """
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO terminology_fixes (
                    id, chunk_id, collection, original_text, suggested_text,
                    suggestion_type, confidence, reasoning, status, created_at, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fix.id,
                    fix.chunk_id,
                    fix.collection,
                    fix.original_text,
                    fix.suggested_text,
                    fix.suggestion_type,
                    fix.confidence,
                    fix.reasoning,
                    fix.status,
                    fix.created_at.isoformat() if fix.created_at else datetime.utcnow().isoformat(),
                    fix.applied_at.isoformat() if fix.applied_at else None,
                ),
            )

        logger.debug("Inserted terminology fix", fix_id=fix.id, chunk_id=fix.chunk_id)
        return fix.id

    def insert_fixes_batch(self, fixes: list[TerminologyFix]) -> int:
        """Insert multiple terminology fixes in a batch.

        Args:
            fixes: List of terminology fixes to insert.

        Returns:
            Number of fixes inserted.
        """
        if not fixes:
            return 0

        with self._cursor() as cursor:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO terminology_fixes (
                    id, chunk_id, collection, original_text, suggested_text,
                    suggestion_type, confidence, reasoning, status, created_at, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        fix.id,
                        fix.chunk_id,
                        fix.collection,
                        fix.original_text,
                        fix.suggested_text,
                        fix.suggestion_type,
                        fix.confidence,
                        fix.reasoning,
                        fix.status,
                        (
                            fix.created_at.isoformat()
                            if fix.created_at
                            else datetime.utcnow().isoformat()
                        ),
                        fix.applied_at.isoformat() if fix.applied_at else None,
                    )
                    for fix in fixes
                ],
            )

        logger.info("Inserted terminology fixes batch", count=len(fixes))
        return len(fixes)

    def get_fix(self, fix_id: str) -> TerminologyFix | None:
        """Get a terminology fix by ID.

        Args:
            fix_id: The fix ID.

        Returns:
            TerminologyFix or None if not found.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM terminology_fixes WHERE id = ?",
                (fix_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_fix(row)

    def list_fixes(
        self,
        status: SuggestionStatus | None = None,
        collection: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TerminologyFix], int]:
        """List terminology fixes with optional filtering.

        Args:
            status: Filter by status.
            collection: Filter by collection.
            limit: Maximum number of fixes to return.
            offset: Number of fixes to skip.

        Returns:
            Tuple of (list of fixes, total count).
        """
        conditions = []
        params: list[str | int] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if collection:
            conditions.append("collection = ?")
            params.append(collection)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._cursor() as cursor:
            # Get total count
            cursor.execute(
                f"SELECT COUNT(*) FROM terminology_fixes {where_clause}",
                params,
            )
            total = cursor.fetchone()[0]

            # Get fixes
            cursor.execute(
                f"""
                SELECT * FROM terminology_fixes
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            )
            rows = cursor.fetchall()

        fixes = [self._row_to_fix(row) for row in rows]
        return fixes, total

    def update_status(
        self,
        fix_id: str,
        status: SuggestionStatus,
        applied_at: datetime | None = None,
    ) -> bool:
        """Update the status of a terminology fix.

        Args:
            fix_id: The fix ID.
            status: New status.
            applied_at: Timestamp when applied (for 'applied' status).

        Returns:
            True if updated, False if not found.
        """
        with self._cursor() as cursor:
            if status == "applied" and applied_at:
                cursor.execute(
                    """
                    UPDATE terminology_fixes
                    SET status = ?, applied_at = ?
                    WHERE id = ?
                    """,
                    (status, applied_at.isoformat(), fix_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE terminology_fixes
                    SET status = ?
                    WHERE id = ?
                    """,
                    (status, fix_id),
                )
            return cursor.rowcount > 0

    def update_statuses_batch(
        self,
        fix_ids: list[str],
        status: SuggestionStatus,
        applied_at: datetime | None = None,
    ) -> int:
        """Update status for multiple fixes at once.

        Args:
            fix_ids: List of fix IDs to update.
            status: New status.
            applied_at: Timestamp when applied.

        Returns:
            Number of fixes updated.
        """
        if not fix_ids:
            return 0

        placeholders = ",".join("?" * len(fix_ids))

        with self._cursor() as cursor:
            if status == "applied" and applied_at:
                cursor.execute(
                    f"""
                    UPDATE terminology_fixes
                    SET status = ?, applied_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [status, applied_at.isoformat(), *fix_ids],
                )
            else:
                cursor.execute(
                    f"""
                    UPDATE terminology_fixes
                    SET status = ?
                    WHERE id IN ({placeholders})
                    """,
                    [status, *fix_ids],
                )
            return cursor.rowcount

    def delete_fix(self, fix_id: str) -> bool:
        """Delete a terminology fix.

        Args:
            fix_id: The fix ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM terminology_fixes WHERE id = ?", (fix_id,))
            return cursor.rowcount > 0

    def clear_pending(self) -> int:
        """Clear all pending fixes (for fresh re-scan).

        Returns:
            Number of fixes deleted.
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM terminology_fixes WHERE status = 'pending'")
            deleted = cursor.rowcount

        logger.info("Cleared pending terminology fixes", count=deleted)
        return deleted

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get statistics about terminology fixes.

        Returns:
            Dict with stats (total, by_status, by_collection, etc.).
        """
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM terminology_fixes")
            total = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM terminology_fixes
                GROUP BY status
                ORDER BY count DESC
                """
            )
            by_status = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT collection, COUNT(*) as count
                FROM terminology_fixes
                GROUP BY collection
                ORDER BY count DESC
                """
            )
            by_collection = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT suggestion_type, COUNT(*) as count
                FROM terminology_fixes
                GROUP BY suggestion_type
                ORDER BY count DESC
                """
            )
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total": total,
            "by_status": by_status,
            "by_collection": by_collection,
            "by_type": by_type,
        }

    def get_approved_fixes(self) -> list[TerminologyFix]:
        """Get all approved fixes ready to be applied.

        Returns:
            List of approved TerminologyFix objects.
        """
        fixes, _ = self.list_fixes(status="approved", limit=1000)
        return fixes

    def export_to_json(self, filepath: str) -> int:
        """Export all fixes to a JSON file.

        Args:
            filepath: Path to output JSON file.

        Returns:
            Number of fixes exported.
        """
        fixes, total = self.list_fixes(limit=10000)

        with open(filepath, "w") as f:
            json.dump(
                {
                    "exported_at": datetime.utcnow().isoformat(),
                    "total": total,
                    "fixes": [fix.to_dict() for fix in fixes],
                },
                f,
                indent=2,
            )

        logger.info("Exported terminology fixes to JSON", filepath=filepath, count=total)
        return total

    def import_from_json(self, filepath: str) -> int:
        """Import fixes from a JSON file.

        Args:
            filepath: Path to input JSON file.

        Returns:
            Number of fixes imported.
        """
        with open(filepath) as f:
            data = json.load(f)

        fixes = []
        for fix_data in data.get("fixes", []):
            fix = TerminologyFix(
                id=fix_data["id"],
                chunk_id=fix_data["chunk_id"],
                collection=fix_data["collection"],
                original_text=fix_data["original_text"],
                suggested_text=fix_data["suggested_text"],
                suggestion_type=fix_data["suggestion_type"],
                confidence=fix_data["confidence"],
                reasoning=fix_data.get("reasoning", ""),
                status=fix_data.get("status", "pending"),
                created_at=(
                    datetime.fromisoformat(fix_data["created_at"])
                    if fix_data.get("created_at")
                    else None
                ),
                applied_at=(
                    datetime.fromisoformat(fix_data["applied_at"])
                    if fix_data.get("applied_at")
                    else None
                ),
            )
            fixes.append(fix)

        count = self.insert_fixes_batch(fixes)
        logger.info("Imported terminology fixes from JSON", filepath=filepath, count=count)
        return count

    def _row_to_fix(self, row: sqlite3.Row) -> TerminologyFix:
        """Convert a database row to a TerminologyFix object."""
        return TerminologyFix(
            id=row["id"],
            chunk_id=row["chunk_id"],
            collection=row["collection"],
            original_text=row["original_text"],
            suggested_text=row["suggested_text"],
            suggestion_type=row["suggestion_type"],
            confidence=row["confidence"],
            reasoning=row["reasoning"] or "",
            status=row["status"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            applied_at=(
                datetime.fromisoformat(row["applied_at"]) if row["applied_at"] else None
            ),
        )
