"""Search analytics database - separate from camera database."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class AnalyticsDatabase:
    """SQLite database for search analytics - completely separate from cameras."""

    def __init__(self, db_path: str = "data/analytics.db") -> None:
        """Initialize analytics database.

        Args:
            db_path: Path to the SQLite database file (separate from clorag.db).
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    source TEXT DEFAULT 'both',
                    response_time_ms INTEGER,
                    results_count INTEGER,
                    response TEXT,
                    chunks TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_queries_created
                ON search_queries(created_at)
            """)
            # Migration: add columns if they don't exist
            cursor = conn.execute("PRAGMA table_info(search_queries)")
            columns = {row[1] for row in cursor.fetchall()}
            if "response" not in columns:
                conn.execute("ALTER TABLE search_queries ADD COLUMN response TEXT")
            if "chunks" not in columns:
                conn.execute("ALTER TABLE search_queries ADD COLUMN chunks TEXT")
            if "session_id" not in columns:
                conn.execute("ALTER TABLE search_queries ADD COLUMN session_id TEXT")
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_search_queries_session
                    ON search_queries(session_id)
                """)
            if "reranked" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN reranked INTEGER DEFAULT 0"
                )
            conn.commit()

    def log_search(
        self,
        query: str,
        source: str = "both",
        response_time_ms: int | None = None,
        results_count: int | None = None,
        response: str | None = None,
        chunks: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        reranked: bool = False,
    ) -> int:
        """Log a search query with full response data.

        Args:
            query: The search query text.
            source: Data source used (docs, gmail, both).
            response_time_ms: Response time in milliseconds.
            results_count: Number of results returned.
            response: The LLM-generated response text.
            chunks: List of retrieved chunks with metadata.
            session_id: Conversation session ID for grouping follow-ups.
            reranked: Whether reranking was applied to results.

        Returns:
            The ID of the inserted record.
        """
        chunks_json = json.dumps(chunks) if chunks else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO search_queries
                (query, source, response_time_ms, results_count, response, chunks, session_id,
                 reranked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (query, source, response_time_ms, results_count, response, chunks_json,
                 session_id, 1 if reranked else 0),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_popular_queries(
        self, limit: int = 10, days: int = 30
    ) -> list[dict[str, Any]]:
        """Get most popular queries in the given time period.

        Args:
            limit: Maximum number of results.
            days: Number of days to look back.

        Returns:
            List of dicts with query, count, and last_searched.
        """
        since = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    query,
                    COUNT(*) as count,
                    MAX(created_at) as last_searched
                FROM search_queries
                WHERE created_at >= ?
                GROUP BY LOWER(query)
                ORDER BY count DESC
                LIMIT ?
                """,
                (since.isoformat(), limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_search_stats(self, days: int = 30) -> dict[str, Any]:
        """Get aggregated search statistics.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with total_searches, avg_response_time_ms, searches_by_source, etc.
        """
        since = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total searches and avg response time
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_searches,
                    AVG(response_time_ms) as avg_response_time_ms,
                    AVG(results_count) as avg_results_count
                FROM search_queries
                WHERE created_at >= ?
                """,
                (since.isoformat(),),
            )
            row = cursor.fetchone()
            stats: dict[str, Any] = dict(row) if row else {}

            # Searches by source
            cursor = conn.execute(
                """
                SELECT source, COUNT(*) as count
                FROM search_queries
                WHERE created_at >= ?
                GROUP BY source
                """,
                (since.isoformat(),),
            )
            stats["searches_by_source"] = {
                row["source"]: row["count"] for row in cursor.fetchall()
            }

            # Searches per day (last 7 days)
            seven_days_ago = datetime.now() - timedelta(days=7)
            cursor = conn.execute(
                """
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as count
                FROM search_queries
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                """,
                (seven_days_ago.isoformat(),),
            )
            stats["searches_per_day"] = [dict(row) for row in cursor.fetchall()]

            return stats

    def get_recent_searches(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most recent searches.

        Args:
            limit: Maximum number of results.

        Returns:
            List of recent search records.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, query, source, response_time_ms, results_count, reranked, created_at
                FROM search_queries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r["reranked"] = bool(r.get("reranked", 0))
                results.append(r)
            return results

    def get_search_by_id(self, search_id: int) -> dict[str, Any] | None:
        """Get a single search by ID with full data including response and chunks.

        Args:
            search_id: The ID of the search record.

        Returns:
            Dict with all search data including response and chunks, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, query, source, response_time_ms, results_count,
                       response, chunks, session_id, reranked, created_at
                FROM search_queries
                WHERE id = ?
                """,
                (search_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            # Parse chunks JSON if present
            if result.get("chunks"):
                result["chunks"] = json.loads(result["chunks"])
            # Convert reranked to boolean
            result["reranked"] = bool(result.get("reranked", 0))
            return result

    def get_recent_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent conversations grouped by session_id.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversations, each with session_id, query count,
            first query, last query time, and all queries in the session.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Get recent unique sessions with their stats
            cursor = conn.execute(
                """
                SELECT
                    session_id,
                    COUNT(*) as query_count,
                    MIN(created_at) as started_at,
                    MAX(created_at) as last_query_at,
                    GROUP_CONCAT(id) as query_ids
                FROM search_queries
                WHERE session_id IS NOT NULL
                GROUP BY session_id
                ORDER BY MAX(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            )
            conversations = []
            for row in cursor.fetchall():
                conv = dict(row)
                # Get all queries for this session
                query_ids = conv.pop("query_ids", "").split(",")
                if query_ids and query_ids[0]:
                    placeholders = ",".join("?" * len(query_ids))
                    queries_cursor = conn.execute(
                        f"""
                        SELECT id, query, source, response_time_ms, results_count, created_at
                        FROM search_queries
                        WHERE id IN ({placeholders})
                        ORDER BY created_at ASC
                        """,
                        query_ids,
                    )
                    conv["queries"] = [dict(q) for q in queries_cursor.fetchall()]
                else:
                    conv["queries"] = []
                conversations.append(conv)
            return conversations
