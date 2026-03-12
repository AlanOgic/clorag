"""SQLite database for ingestion job tracking and log storage.

Stores job metadata, status transitions, and per-job log entries
to enable admin UI monitoring of ingestion processes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from clorag.config import get_settings
from clorag.core.database import ConnectionPool

logger = structlog.get_logger(__name__)


# Singleton instance
_ingestion_db: IngestionDatabase | None = None


def get_ingestion_database() -> IngestionDatabase:
    """Get or create the singleton IngestionDatabase instance."""
    global _ingestion_db
    if _ingestion_db is None:
        _ingestion_db = IngestionDatabase()
    return _ingestion_db


# Valid job statuses
JOB_STATUSES = frozenset({"pending", "running", "completed", "failed", "cancelled"})


class IngestionJob:
    """Model representing an ingestion job."""

    def __init__(
        self,
        id: str,
        job_type: str,
        status: str,
        parameters: dict[str, Any],
        started_at: datetime | None,
        finished_at: datetime | None,
        duration_seconds: float | None,
        result_summary: dict[str, Any] | None,
        error_message: str | None,
        created_at: datetime,
    ) -> None:
        """Initialize job model."""
        self.id = id
        self.job_type = job_type
        self.status = status
        self.parameters = parameters
        self.started_at = started_at
        self.finished_at = finished_at
        self.duration_seconds = duration_seconds
        self.result_summary = result_summary
        self.error_message = error_message
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "parameters": self.parameters,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IngestionLog:
    """Model representing a log entry for an ingestion job."""

    def __init__(
        self,
        id: str,
        job_id: str,
        timestamp: datetime,
        level: str,
        message: str,
        event: str | None,
        extra: dict[str, Any] | None,
    ) -> None:
        """Initialize log model."""
        self.id = id
        self.job_id = job_id
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.event = event
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "message": self.message,
            "event": self.event,
            "extra": self.extra,
        }


class IngestionDatabase:
    """SQLite database for tracking ingestion jobs and their logs.

    Stores job metadata with status transitions and per-job log entries
    for admin UI monitoring and historical review.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the ingestion database.

        Args:
            db_path: Path to SQLite database. Defaults to settings.database_path.
        """
        settings = get_settings()
        self._db_path = db_path or str(settings.database_path)

        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    parameters TEXT DEFAULT '{}',
                    started_at TEXT,
                    finished_at TEXT,
                    duration_seconds REAL,
                    result_summary TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    event TEXT,
                    extra TEXT,
                    FOREIGN KEY (job_id) REFERENCES ingestion_jobs(id) ON DELETE CASCADE
                )
            """)

            # Indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_type "
                "ON ingestion_jobs(job_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status "
                "ON ingestion_jobs(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created "
                "ON ingestion_jobs(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_job_id "
                "ON ingestion_logs(job_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_timestamp "
                "ON ingestion_logs(job_id, timestamp)"
            )

            conn.commit()

        logger.info("Ingestion database schema initialized", path=self._db_path)

    def _row_to_job(self, row: sqlite3.Row) -> IngestionJob:
        """Convert a database row to an IngestionJob object."""
        return IngestionJob(
            id=row["id"],
            job_type=row["job_type"],
            status=row["status"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            finished_at=(
                datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            duration_seconds=row["duration_seconds"],
            result_summary=(
                json.loads(row["result_summary"]) if row["result_summary"] else None
            ),
            error_message=row["error_message"],
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"]
                else datetime.now()
            ),
        )

    def _row_to_log(self, row: sqlite3.Row) -> IngestionLog:
        """Convert a database row to an IngestionLog object."""
        return IngestionLog(
            id=row["id"],
            job_id=row["job_id"],
            timestamp=(
                datetime.fromisoformat(row["timestamp"])
                if row["timestamp"]
                else datetime.now()
            ),
            level=row["level"],
            message=row["message"],
            event=row["event"],
            extra=json.loads(row["extra"]) if row["extra"] else None,
        )

    # =========================================================================
    # Job CRUD
    # =========================================================================

    def create_job(
        self,
        job_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> IngestionJob:
        """Create a new ingestion job.

        Args:
            job_type: Type of ingestion job.
            parameters: Job parameters as dict.

        Returns:
            Created IngestionJob object.
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (id, job_type, status, parameters, created_at)
                VALUES (?, ?, 'pending', ?, ?)
                """,
                (job_id, job_type, json.dumps(parameters or {}), now),
            )
            conn.commit()

        logger.info("Created ingestion job", job_id=job_id, job_type=job_type)
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Get a job by ID.

        Args:
            job_id: The job UUID.

        Returns:
            IngestionJob object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(
        self,
        job_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionJob]:
        """List jobs with optional filtering.

        Args:
            job_type: Filter by job type.
            status: Filter by status.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of IngestionJob objects.
        """
        conditions: list[str] = []
        params: list[str | int] = []

        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM ingestion_jobs {where_clause} "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            )
            rows = cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    def update_status(
        self,
        job_id: str,
        status: str,
    ) -> None:
        """Update job status.

        Args:
            job_id: The job UUID.
            status: New status.
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            updates = ["status = ?"]
            params: list[str] = [status]

            if status == "running":
                updates.append("started_at = ?")
                params.append(now)

            params.append(job_id)
            conn.execute(
                f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

    def complete_job(
        self,
        job_id: str,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """Mark a job as completed.

        Args:
            job_id: The job UUID.
            result_summary: Summary of results.
        """
        now = datetime.utcnow()
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            # Calculate duration
            cursor = conn.execute(
                "SELECT started_at FROM ingestion_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()
            duration = None
            if row and row["started_at"]:
                started = datetime.fromisoformat(row["started_at"])
                duration = (now - started).total_seconds()

            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'completed', finished_at = ?, duration_seconds = ?,
                    result_summary = ?
                WHERE id = ?
                """,
                (now_iso, duration, json.dumps(result_summary) if result_summary else None, job_id),
            )
            conn.commit()

    def fail_job(
        self,
        job_id: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed.

        Args:
            job_id: The job UUID.
            error_message: Error description.
        """
        now = datetime.utcnow()
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT started_at FROM ingestion_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()
            duration = None
            if row and row["started_at"]:
                started = datetime.fromisoformat(row["started_at"])
                duration = (now - started).total_seconds()

            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'failed', finished_at = ?, duration_seconds = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (now_iso, duration, error_message, job_id),
            )
            conn.commit()

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its logs.

        Args:
            job_id: The job UUID.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM ingestion_logs WHERE job_id = ?", (job_id,))
            cursor = conn.execute("DELETE FROM ingestion_jobs WHERE id = ?", (job_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted ingestion job", job_id=job_id)
        return deleted

    # =========================================================================
    # Log Operations
    # =========================================================================

    def insert_log(
        self,
        job_id: str,
        level: str,
        message: str,
        event: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Insert a single log entry.

        Args:
            job_id: The job UUID.
            level: Log level (DEBUG, INFO, WARNING, ERROR).
            message: Log message.
            event: Structlog event name.
            extra: Additional structured data.
        """
        log_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_logs (id, job_id, timestamp, level, message, event, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    job_id,
                    now,
                    level,
                    message,
                    event,
                    json.dumps(extra) if extra else None,
                ),
            )
            conn.commit()

    def insert_logs_batch(
        self,
        logs: list[dict[str, Any]],
    ) -> None:
        """Insert multiple log entries in a batch.

        Args:
            logs: List of log dicts with keys: job_id, level, message, event, extra.
        """
        if not logs:
            return

        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO ingestion_logs (id, job_id, timestamp, level, message, event, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        log["job_id"],
                        log.get("timestamp", now),
                        log["level"],
                        log["message"],
                        log.get("event"),
                        json.dumps(log["extra"]) if log.get("extra") else None,
                    )
                    for log in logs
                ],
            )
            conn.commit()

    def get_logs(
        self,
        job_id: str,
        limit: int = 500,
        offset: int = 0,
        level: str | None = None,
    ) -> list[IngestionLog]:
        """Get logs for a job.

        Args:
            job_id: The job UUID.
            limit: Maximum results.
            offset: Pagination offset.
            level: Filter by log level.

        Returns:
            List of IngestionLog objects.
        """
        conditions = ["job_id = ?"]
        params: list[str | int] = [job_id]

        if level:
            conditions.append("level = ?")
            params.append(level)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM ingestion_logs {where_clause} "
                "ORDER BY timestamp ASC LIMIT ? OFFSET ?",
                params,
            )
            rows = cursor.fetchall()

        return [self._row_to_log(row) for row in rows]

    # =========================================================================
    # Maintenance
    # =========================================================================

    def mark_stale_running_as_failed(self) -> int:
        """Mark any jobs left in 'running' status as 'failed'.

        Called on startup to clean up jobs from a previous process that crashed.

        Returns:
            Number of jobs marked as failed.
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'failed', finished_at = ?,
                    error_message = 'Process restarted while job was running'
                WHERE status = 'running'
                """,
                (now,),
            )
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.warning("Marked stale running jobs as failed", count=count)
        return count

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of jobs deleted.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            # Delete logs first (CASCADE should handle, but be explicit)
            conn.execute(
                """
                DELETE FROM ingestion_logs WHERE job_id IN (
                    SELECT id FROM ingestion_jobs WHERE created_at < ?
                )
                """,
                (cutoff,),
            )
            cursor = conn.execute(
                "DELETE FROM ingestion_jobs WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.info("Cleaned up old ingestion jobs", count=count, days=days)
        return count

    def get_running_jobs(self) -> list[IngestionJob]:
        """Get all currently running jobs.

        Returns:
            List of running IngestionJob objects.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE status = 'running' ORDER BY started_at"
            )
            rows = cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    def get_job_count(
        self,
        job_type: str | None = None,
        status: str | None = None,
    ) -> int:
        """Get count of jobs matching filters.

        Args:
            job_type: Filter by job type.
            status: Filter by status.

        Returns:
            Number of matching jobs.
        """
        conditions: list[str] = []
        params: list[str] = []

        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) as count FROM ingestion_jobs {where_clause}",
                params,
            )
            row = cursor.fetchone()

        return row["count"] if row else 0
