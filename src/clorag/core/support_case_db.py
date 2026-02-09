"""SQLite database for storing full support case documents."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import structlog

from clorag.config import get_settings
from clorag.core.database import ConnectionPool
from clorag.models.support_case import CaseStatus, ResolutionQuality, SupportCase

logger = structlog.get_logger(__name__)

# Singleton instance
_support_case_db: SupportCaseDatabase | None = None


def get_support_case_database() -> SupportCaseDatabase:
    """Get or create the singleton SupportCaseDatabase instance."""
    global _support_case_db
    if _support_case_db is None:
        _support_case_db = SupportCaseDatabase()
    return _support_case_db


class SupportCaseDatabase:
    """SQLite database for storing full support case documents.

    Stores the complete anonymized, summarized content from Gmail support cases
    for easy retrieval without needing to reconstruct from chunks.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the support case database.

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
                CREATE TABLE IF NOT EXISTS support_cases (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT UNIQUE NOT NULL,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolution_quality INTEGER,
                    problem_summary TEXT,
                    solution_summary TEXT,
                    keywords TEXT,
                    category TEXT,
                    product TEXT,
                    document TEXT NOT NULL,
                    raw_thread TEXT,
                    messages_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    resolved_at TEXT,
                    ingested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_support_cases_thread_id
                ON support_cases(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_support_cases_category
                ON support_cases(category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_support_cases_product
                ON support_cases(product)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_support_cases_created_at
                ON support_cases(created_at)
            """)

            # FTS5 for full-text search on document content
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS support_cases_fts USING fts5(
                    subject, problem_summary, solution_summary, document, keywords,
                    content='support_cases',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)

            # Triggers to keep FTS in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS support_cases_ai AFTER INSERT ON support_cases BEGIN
                    INSERT INTO support_cases_fts(
                        rowid, subject, problem_summary, solution_summary, document, keywords
                    ) VALUES (
                        NEW.rowid, NEW.subject, NEW.problem_summary,
                        NEW.solution_summary, NEW.document, NEW.keywords
                    );
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS support_cases_ad AFTER DELETE ON support_cases BEGIN
                    INSERT INTO support_cases_fts(
                        support_cases_fts, rowid, subject, problem_summary,
                        solution_summary, document, keywords
                    ) VALUES (
                        'delete', OLD.rowid, OLD.subject, OLD.problem_summary,
                        OLD.solution_summary, OLD.document, OLD.keywords
                    );
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS support_cases_au AFTER UPDATE ON support_cases BEGIN
                    INSERT INTO support_cases_fts(
                        support_cases_fts, rowid, subject, problem_summary,
                        solution_summary, document, keywords
                    ) VALUES (
                        'delete', OLD.rowid, OLD.subject, OLD.problem_summary,
                        OLD.solution_summary, OLD.document, OLD.keywords
                    );
                    INSERT INTO support_cases_fts(
                        rowid, subject, problem_summary, solution_summary, document, keywords
                    ) VALUES (
                        NEW.rowid, NEW.subject, NEW.problem_summary,
                        NEW.solution_summary, NEW.document, NEW.keywords
                    );
                END
            """)

        logger.info("Support case database schema initialized", path=self._db_path)

    def upsert_case(self, case: SupportCase, raw_thread: str | None = None) -> str:
        """Insert or update a support case.

        Args:
            case: The support case to store.
            raw_thread: Optional anonymized raw thread content.

        Returns:
            The case ID.
        """
        now = datetime.utcnow().isoformat()

        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO support_cases (
                    id, thread_id, subject, status, resolution_quality,
                    problem_summary, solution_summary, keywords, category, product,
                    document, raw_thread, messages_count, created_at, resolved_at,
                    ingested_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    subject = excluded.subject,
                    status = excluded.status,
                    resolution_quality = excluded.resolution_quality,
                    problem_summary = excluded.problem_summary,
                    solution_summary = excluded.solution_summary,
                    keywords = excluded.keywords,
                    category = excluded.category,
                    product = excluded.product,
                    document = excluded.document,
                    raw_thread = COALESCE(excluded.raw_thread, raw_thread),
                    messages_count = excluded.messages_count,
                    created_at = excluded.created_at,
                    resolved_at = excluded.resolved_at,
                    updated_at = excluded.updated_at
                """,
                (
                    case.id,
                    case.thread_id,
                    case.subject,
                    case.status.value,
                    case.resolution_quality.value if case.resolution_quality else None,
                    case.problem_summary,
                    case.solution_summary,
                    json.dumps(case.keywords),
                    case.category,
                    case.product,
                    case.document,
                    raw_thread,
                    case.messages_count,
                    case.created_at.isoformat() if case.created_at else None,
                    case.resolved_at.isoformat() if case.resolved_at else None,
                    now,
                    now,
                ),
            )

        logger.debug("Upserted support case", case_id=case.id, thread_id=case.thread_id)
        return case.id

    def get_case_by_id(self, case_id: str) -> SupportCase | None:
        """Get a support case by its ID.

        Args:
            case_id: The case ID.

        Returns:
            The SupportCase or None if not found.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM support_cases WHERE id = ?",
                (case_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_case(row)

    def get_case_by_thread_id(self, thread_id: str) -> SupportCase | None:
        """Get a support case by Gmail thread ID.

        Args:
            thread_id: The Gmail thread ID.

        Returns:
            The SupportCase or None if not found.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM support_cases WHERE thread_id = ?",
                (thread_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_case(row)

    def list_cases(
        self,
        category: str | None = None,
        product: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SupportCase], int]:
        """List support cases with optional filtering.

        Args:
            category: Filter by category.
            product: Filter by product.
            limit: Maximum number of cases to return.
            offset: Number of cases to skip.

        Returns:
            Tuple of (list of cases, total count).
        """
        conditions = []
        params: list[str | int] = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if product:
            conditions.append("product = ?")
            params.append(product)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._cursor() as cursor:
            # Get total count
            cursor.execute(
                f"SELECT COUNT(*) FROM support_cases {where_clause}",
                params,
            )
            total = cursor.fetchone()[0]

            # Get cases
            cursor.execute(
                f"""
                SELECT * FROM support_cases
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            )
            rows = cursor.fetchall()

        cases = [self._row_to_case(row) for row in rows]
        return cases, total

    def _prepare_fts_query(self, query: str) -> str:
        """Prepare a query string for FTS5.

        SECURITY: Sanitizes user input to prevent FTS5 injection attacks.
        Removes FTS5 special operators (NEAR, NOT, OR, AND) and characters
        that could be used for syntax injection.

        Args:
            query: Raw user query.

        Returns:
            FTS5-safe query string.
        """
        if not query or not query.strip():
            return '""'

        # Remove FTS5 special characters that could cause syntax errors or injection
        special_chars = ['"', "'", "(", ")", "*", ":", "^", "-", "+"]
        clean_query = query
        for char in special_chars:
            clean_query = clean_query.replace(char, " ")

        # Remove FTS5 operators (case-insensitive)
        # These could be used for injection attacks like "NEAR(a,b,100000)" DoS
        import re

        for operator in ["NEAR", "NOT", "OR", "AND"]:
            clean_query = re.sub(
                rf"\b{operator}\b", " ", clean_query, flags=re.IGNORECASE
            )

        # Split into terms and add prefix matching
        terms = clean_query.split()
        if not terms:
            return '""'

        # Quote each term and add wildcard suffix for prefix matching
        fts_terms = [f'"{term}"*' for term in terms if term]
        return " OR ".join(fts_terms) if fts_terms else '""'

    def search_cases(self, query: str, limit: int = 20) -> list[SupportCase]:
        """Search support cases using FTS5.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching cases.
        """
        # SECURITY: Sanitize FTS5 query to prevent injection attacks
        fts_query = self._prepare_fts_query(query)

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT sc.* FROM support_cases sc
                JOIN support_cases_fts fts ON sc.rowid = fts.rowid
                WHERE support_cases_fts MATCH ?
                ORDER BY bm25(support_cases_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            )
            rows = cursor.fetchall()

        return [self._row_to_case(row) for row in rows]

    def get_raw_thread(self, case_id: str) -> str | None:
        """Get the raw anonymized thread content for a case.

        Args:
            case_id: The case ID.

        Returns:
            The raw thread content or None.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT raw_thread FROM support_cases WHERE id = ?",
                (case_id,),
            )
            row = cursor.fetchone()

        return row["raw_thread"] if row else None

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get statistics about stored support cases.

        Returns:
            Dict with stats (total, by_category, by_product, etc.).
        """
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM support_cases")
            total = cursor.fetchone()["total"]

            cursor.execute(
                """
                SELECT category, COUNT(*) as count
                FROM support_cases
                GROUP BY category
                ORDER BY count DESC
                """
            )
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT product, COUNT(*) as count
                FROM support_cases
                WHERE product IS NOT NULL
                GROUP BY product
                ORDER BY count DESC
                """
            )
            by_product = {row["product"]: row["count"] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT resolution_quality, COUNT(*) as count
                FROM support_cases
                WHERE resolution_quality IS NOT NULL
                GROUP BY resolution_quality
                ORDER BY resolution_quality DESC
                """
            )
            by_quality = {
                str(row["resolution_quality"]): row["count"] for row in cursor.fetchall()
            }

        return {
            "total": total,
            "by_category": by_category,
            "by_product": by_product,
            "by_quality": by_quality,
        }

    def delete_case(self, case_id: str) -> bool:
        """Delete a support case.

        Args:
            case_id: The case ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM support_cases WHERE id = ?", (case_id,))
            return cursor.rowcount > 0

    def _row_to_case(self, row: sqlite3.Row) -> SupportCase:
        """Convert a database row to a SupportCase object."""
        keywords = json.loads(row["keywords"]) if row["keywords"] else []

        return SupportCase(
            id=row["id"],
            thread_id=row["thread_id"],
            subject=row["subject"],
            status=CaseStatus(row["status"]),
            resolution_quality=(
                ResolutionQuality(row["resolution_quality"])
                if row["resolution_quality"]
                else None
            ),
            problem_summary=row["problem_summary"] or "",
            solution_summary=row["solution_summary"] or "",
            keywords=keywords,
            category=row["category"] or "",
            product=row["product"],
            document=row["document"],
            raw_thread=row["raw_thread"] or "",
            messages_count=row["messages_count"] or 0,
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None
            ),
        )
