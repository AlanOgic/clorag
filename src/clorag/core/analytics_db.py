"""Search analytics database - separate from camera database."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_PREVIEW_MAX_CHARS = 240


def _extract_preview(response: str | None) -> str | None:
    """Return a short single-line teaser of an answer (~240 chars).

    Strips markdown headings, collapses whitespace, and truncates on a word
    boundary with an ellipsis. Numbered/bulleted lists flatten naturally
    onto one line. Returns None when empty so the UI can omit the preview.
    """
    if not response:
        return None
    text = response.strip()
    # Drop leading markdown heading line(s) ("# ...", "## ..." etc.)
    lines = text.split("\n")
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    text = "\n".join(lines).strip()
    # Collapse all internal whitespace (including newlines) into single spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading quote/bullet markers left after heading cleanup
    text = re.sub(r"^[>\-*•\s]+", "", text).strip()
    if not text:
        return None
    if len(text) <= _PREVIEW_MAX_CHARS:
        return text
    # Prefer cutting on a sentence boundary within the budget, else word boundary
    cut = text[: _PREVIEW_MAX_CHARS]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary >= _PREVIEW_MAX_CHARS // 2:
        return cut[: boundary + 1].rstrip() + " …"
    ws = cut.rfind(" ")
    if ws > 0:
        cut = cut[:ws]
    return cut.rstrip() + "…"


def _src_bucket(sources: dict[str, dict[str, Any]], stype: str) -> dict[str, Any]:
    """Return (creating if absent) the accumulator bucket for a source type."""
    if stype not in sources:
        sources[stype] = {
            "wins": 0,
            "appearances": 0,
            "positions": [0, 0, 0, 0, 0],
            "score_sum": 0.0,
            "score_n": 0,
        }
    return sources[stype]


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
            if "scores" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN scores TEXT"
                )
            if "source_types" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN source_types TEXT"
                )
            if "normalized_query" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN normalized_query TEXT"
                )
            if "rewritten_query" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN rewritten_query TEXT"
                )
            if "pipeline" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN pipeline TEXT DEFAULT 'web'"
                )
            if "tool_calls" not in columns:
                conn.execute(
                    "ALTER TABLE search_queries ADD COLUMN tool_calls TEXT"
                )

            # Brute-force login protection (persisted across restarts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    ip TEXT NOT NULL,
                    attempted_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_login_attempts_ip
                ON login_attempts(ip, attempted_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_lockouts (
                    ip TEXT PRIMARY KEY,
                    locked_until REAL NOT NULL
                )
            """)

            # Feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_id INTEGER NOT NULL UNIQUE,
                    session_id TEXT,
                    rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON search_feedback(created_at)
            """)

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
        scores: list[float] | None = None,
        source_types: list[str] | None = None,
        normalized_query: str | None = None,
        rewritten_query: str | None = None,
        pipeline: str = "web",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> int:
        """Log a search query with full response data and quality metrics.

        Args:
            query: The raw user query as typed.
            source: Data source used (docs, gmail, both).
            response_time_ms: Response time in milliseconds.
            results_count: Number of results returned.
            response: The LLM-generated response text.
            chunks: List of retrieved chunks with metadata.
            session_id: Conversation session ID for grouping follow-ups.
            reranked: Whether reranking was applied to results.
            scores: List of result scores (reranker scores if reranked).
            source_types: List of source types for each result.
            normalized_query: Query after deterministic text transforms
                (e.g. RIO terminology). Stored only when it differs from raw.
            rewritten_query: Standalone query produced by LLM rewrite for
                conversational follow-ups. Stored only when rewritten.
            pipeline: Origin of the query — 'web', 'chat', 'cli_agent', 'mcp'.
            tool_calls: For agent pipelines, list of tool invocations with
                their per-tool queries and result counts.

        Returns:
            The ID of the inserted record.
        """
        chunks_json = json.dumps(chunks) if chunks else None
        scores_json = json.dumps(scores) if scores else None
        source_types_json = json.dumps(source_types) if source_types else None
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO search_queries
                (query, source, response_time_ms, results_count, response, chunks, session_id,
                 reranked, scores, source_types, normalized_query, rewritten_query,
                 pipeline, tool_calls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (query, source, response_time_ms, results_count, response, chunks_json,
                 session_id, 1 if reranked else 0, scores_json, source_types_json,
                 normalized_query, rewritten_query, pipeline, tool_calls_json),
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
                       response, chunks, session_id, reranked, scores,
                       source_types, normalized_query, rewritten_query,
                       pipeline, tool_calls, created_at
                FROM search_queries
                WHERE id = ?
                """,
                (search_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            # Parse JSON fields if present
            if result.get("chunks"):
                result["chunks"] = json.loads(result["chunks"])
            if result.get("scores"):
                result["scores"] = json.loads(result["scores"])
            if result.get("source_types"):
                result["source_types"] = json.loads(result["source_types"])
            if result.get("tool_calls"):
                result["tool_calls"] = json.loads(result["tool_calls"])
            # Convert reranked to boolean
            result["reranked"] = bool(result.get("reranked", 0))
            return result

    def get_low_quality_searches(
        self, limit: int = 50, days: int = 30, max_avg_score: float = 0.3
    ) -> list[dict[str, Any]]:
        """Get searches with low relevance scores for quality review.

        Args:
            limit: Maximum number of results.
            days: Number of days to look back.
            max_avg_score: Maximum average score to consider "low quality".

        Returns:
            List of search records with low scores, sorted by score ascending.
        """
        since = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, query, source, response_time_ms, results_count,
                       reranked, scores, source_types, created_at
                FROM search_queries
                WHERE created_at >= ? AND scores IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (since.isoformat(), limit * 5),  # Over-fetch to filter
            )
            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r["reranked"] = bool(r.get("reranked", 0))
                if r.get("scores"):
                    scores = json.loads(r["scores"])
                    r["scores"] = scores
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        r["avg_score"] = round(avg_score, 4)
                        if avg_score <= max_avg_score:
                            results.append(r)
                if r.get("source_types"):
                    r["source_types"] = json.loads(r["source_types"])

            # Sort by avg_score ascending (worst first)
            results.sort(key=lambda x: x.get("avg_score", 0))
            return results[:limit]

    def get_recent_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent conversations grouped by session_id.

        Each query includes a ``response_preview`` with the first 2–3 sentences
        of the LLM answer (max 240 chars) so the admin UI can show a teaser.

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
                        SELECT id, query, source, response_time_ms, results_count,
                               response, created_at
                        FROM search_queries
                        WHERE id IN ({placeholders})
                        ORDER BY created_at ASC
                        """,
                        query_ids,
                    )
                    queries = []
                    for q in queries_cursor.fetchall():
                        q_dict = dict(q)
                        q_dict["response_preview"] = _extract_preview(q_dict.pop("response", None))
                        queries.append(q_dict)
                    conv["queries"] = queries
                else:
                    conv["queries"] = []
                conversations.append(conv)
            return conversations

    def get_source_insights(self, days: int = 30) -> dict[str, Any]:
        """Compute per-source retrieval insight.

        Aggregates ``source_types`` + ``scores`` parallel arrays stored on
        each logged search to produce:
          - win_rate: % of searches where each source was ranked #1
          - avg_top5_score: mean reranker score of that source in top-5
          - position_mix: count of appearances at each rank 1..5
          - rerank_coverage: % of searches that were reranked

        Args:
            days: Lookback window in days.

        Returns:
            Dict with ``total``, ``reranked_total``, ``rerank_coverage``,
            ``sources`` keyed by source_type.
        """
        since = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT reranked, scores, source_types
                FROM search_queries
                WHERE created_at >= ? AND source_types IS NOT NULL
                """,
                (since.isoformat(),),
            )
            rows = cursor.fetchall()

        total = len(rows)
        reranked_total = 0
        sources: dict[str, dict[str, Any]] = {}

        for reranked_flag, scores_json, source_types_json in rows:
            if reranked_flag:
                reranked_total += 1
            try:
                stypes = json.loads(source_types_json) if source_types_json else []
            except (TypeError, ValueError):
                continue
            try:
                sscores = json.loads(scores_json) if scores_json else []
            except (TypeError, ValueError):
                sscores = []

            if not stypes:
                continue
            # Count wins (rank 1 only) — first source wins
            top_source = stypes[0]
            _src_bucket(sources, top_source)["wins"] += 1

            # Top-5 position mix + score accumulation
            for idx, stype in enumerate(stypes[:5]):
                bucket = _src_bucket(sources, stype)
                bucket["appearances"] += 1
                bucket["positions"][idx] += 1
                if idx < len(sscores) and isinstance(sscores[idx], (int, float)):
                    bucket["score_sum"] += float(sscores[idx])
                    bucket["score_n"] += 1

        # Finalize averages + rates
        for stype, bucket in sources.items():
            bucket["avg_top5_score"] = (
                round(bucket["score_sum"] / bucket["score_n"], 4)
                if bucket["score_n"]
                else None
            )
            bucket["win_rate"] = (
                round(bucket["wins"] / total * 100, 1) if total else 0.0
            )
            # Drop internal accumulators from output
            bucket.pop("score_sum", None)
            bucket.pop("score_n", None)

        return {
            "total": total,
            "reranked_total": reranked_total,
            "rerank_coverage": (
                round(reranked_total / total * 100, 1) if total else 0.0
            ),
            "sources": sources,
        }

    def anonymize_old_searches(self, older_than_days: int) -> int:
        """Strip free-text PII from searches older than the cutoff.

        Blanks out the raw query, LLM response, and retrieved chunks while
        preserving the aggregate columns used by analytics dashboards
        (response_time_ms, results_count, reranked, scores, source_types,
        pipeline, created_at).

        Args:
            older_than_days: Rows older than this many days are anonymized.
                Values <= 0 are treated as a no-op.

        Returns:
            Number of rows anonymized.
        """
        if older_than_days <= 0:
            return 0
        cutoff = datetime.now() - timedelta(days=older_than_days)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE search_queries
                SET query = '[redacted]',
                    response = NULL,
                    chunks = NULL,
                    normalized_query = NULL,
                    rewritten_query = NULL,
                    tool_calls = NULL
                WHERE created_at < ?
                  AND query != '[redacted]'
                """,
                (cutoff.isoformat(),),
            )
            # User-submitted feedback comments are free text (PII under the
            # same rationale as queries); null them in lockstep so the
            # anonymize window is honored for both tables.
            conn.execute(
                """
                UPDATE search_feedback
                SET comment = NULL
                WHERE comment IS NOT NULL
                  AND search_id IN (
                      SELECT id FROM search_queries
                      WHERE created_at < ?
                  )
                """,
                (cutoff.isoformat(),),
            )
            conn.commit()
            return cursor.rowcount or 0

    def purge_old_searches(self, older_than_days: int) -> int:
        """Hard-delete searches (and their feedback) older than the cutoff.

        Args:
            older_than_days: Rows older than this many days are deleted.
                Values <= 0 are treated as a no-op.

        Returns:
            Number of search rows deleted.
        """
        if older_than_days <= 0:
            return 0
        cutoff = datetime.now() - timedelta(days=older_than_days)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                DELETE FROM search_feedback
                WHERE search_id IN (
                    SELECT id FROM search_queries WHERE created_at < ?
                )
                """,
                (cutoff.isoformat(),),
            )
            cursor = conn.execute(
                "DELETE FROM search_queries WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            conn.commit()
            return cursor.rowcount or 0

    def get_login_lockout_until(self, ip: str) -> float:
        """Return the stored lockout expiry timestamp for ``ip`` (0.0 if none).

        Expired rows are cleaned up lazily when the caller observes them as
        expired, to keep the table bounded without a background job.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT locked_until FROM login_lockouts WHERE ip = ?",
                (ip,),
            )
            row = cursor.fetchone()
            return float(row[0]) if row else 0.0

    def record_login_attempt(
        self, ip: str, now: float, window_seconds: int, threshold: int
    ) -> tuple[int, bool]:
        """Record a failed login attempt and return (recent_count, should_lock).

        Only attempts within the last ``window_seconds`` count toward the
        threshold, matching the in-memory sliding-window semantics.
        """
        cutoff = now - window_seconds
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM login_attempts WHERE ip = ? AND attempted_at < ?",
                (ip, cutoff),
            )
            conn.execute(
                "INSERT INTO login_attempts (ip, attempted_at) VALUES (?, ?)",
                (ip, now),
            )
            cursor = conn.execute(
                "SELECT COUNT(*) FROM login_attempts WHERE ip = ?",
                (ip,),
            )
            count = int(cursor.fetchone()[0])
            conn.commit()
        return count, count >= threshold

    def set_login_lockout(self, ip: str, until: float) -> None:
        """Persist a lockout expiry for ``ip``."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO login_lockouts (ip, locked_until)
                VALUES (?, ?)
                ON CONFLICT(ip) DO UPDATE SET locked_until = excluded.locked_until
                """,
                (ip, until),
            )
            conn.commit()

    def clear_login_attempts(self, ip: str) -> None:
        """Clear all attempts and any lockout for ``ip`` (successful login)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM login_attempts WHERE ip = ?", (ip,))
            conn.execute("DELETE FROM login_lockouts WHERE ip = ?", (ip,))
            conn.commit()

    def purge_login_state(self, now: float, window_seconds: int) -> None:
        """Drop expired lockouts and stale attempt rows across all IPs.

        record_login_attempt() only prunes rows for the IP it was called
        with, so a distributed brute-force sweep that rotates through
        many source IPs leaves orphaned rows behind. This global prune
        keeps login_attempts bounded regardless of rotation strategy.

        Args:
            now: Current wall-clock epoch seconds.
            window_seconds: Attempts older than this are discarded — pass
                the same value LoginAttemptTracker uses as its sliding
                window (LOGIN_LOCKOUT_DURATION).
        """
        cutoff = now - window_seconds
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM login_lockouts WHERE locked_until < ?",
                (now,),
            )
            conn.execute(
                "DELETE FROM login_attempts WHERE attempted_at < ?",
                (cutoff,),
            )
            conn.commit()

    def save_feedback(
        self,
        search_id: int,
        rating: str,
        comment: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Save or update feedback for a search result.

        Uses upsert: one feedback per search_id. Clicking the other thumb
        replaces the previous vote.

        Args:
            search_id: The search_queries record ID.
            rating: 'up' or 'down'.
            comment: Optional user comment (typically on thumbs down).
            session_id: Conversation session ID.

        Returns:
            The feedback record ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO search_feedback (search_id, session_id, rating, comment)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(search_id) DO UPDATE SET
                    rating = excluded.rating,
                    comment = excluded.comment,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (search_id, session_id, rating, comment),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_feedback_stats(self, days: int = 30) -> dict[str, Any]:
        """Get aggregated feedback statistics.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with total, thumbs_up, thumbs_down, satisfaction_rate, feedback_rate.
        """
        since = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) as thumbs_down
                FROM search_feedback
                WHERE created_at >= ?
                """,
                (since.isoformat(),),
            )
            row = cursor.fetchone()
            total = row["total"] or 0
            thumbs_up = row["thumbs_up"] or 0
            thumbs_down = row["thumbs_down"] or 0

            # Total searches in same period for feedback rate
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM search_queries WHERE created_at >= ?",
                (since.isoformat(),),
            )
            total_searches = cursor.fetchone()["cnt"] or 0

            return {
                "total": total,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "satisfaction_rate": round(thumbs_up / total * 100, 1) if total else 0.0,
                "feedback_rate": round(total / total_searches * 100, 1) if total_searches else 0.0,
                "total_searches": total_searches,
            }

    def get_recent_feedback(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent feedback joined with search queries.

        Args:
            limit: Maximum number of results.

        Returns:
            List of feedback records with query text, rating, comment, timestamp.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    f.id,
                    f.search_id,
                    f.rating,
                    f.comment,
                    f.created_at as feedback_at,
                    f.updated_at,
                    sq.query,
                    sq.source,
                    sq.results_count,
                    sq.session_id
                FROM search_feedback f
                JOIN search_queries sq ON sq.id = f.search_id
                ORDER BY f.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
