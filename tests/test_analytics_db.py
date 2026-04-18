"""Tests for analytics DB preview extraction and source insights."""

from __future__ import annotations

import tempfile
from pathlib import Path

from clorag.core.analytics_db import AnalyticsDatabase, _extract_preview


class TestExtractPreview:
    def test_empty_returns_none(self) -> None:
        assert _extract_preview(None) is None
        assert _extract_preview("") is None
        assert _extract_preview("   \n\n  ") is None

    def test_strips_markdown_heading(self) -> None:
        assert _extract_preview("# Title\n\nAnswer body.") == "Answer body."
        assert _extract_preview("## Sub\n\nHello.") == "Hello."

    def test_flattens_numbered_list(self) -> None:
        text = "# Steps\n\n1. First.\n2. Second.\n3. Third."
        assert _extract_preview(text) == "1. First. 2. Second. 3. Third."

    def test_truncates_long_text(self) -> None:
        out = _extract_preview("A" * 500)
        assert out is not None
        assert len(out) <= 241  # 240 chars + ellipsis
        assert out.endswith("…")

    def test_prefers_sentence_boundary_for_truncation(self) -> None:
        first = "This is the first sentence. " * 5  # ~140 chars
        filler = (
            "Also some extra words that keep going on and on and on and "
            "will get cut off mid-thought."
        )
        long = first + filler
        out = _extract_preview(long)
        assert out is not None
        # Should cut after a period, not mid-word
        assert ". …" in out or out.rstrip(" …").endswith(".")


class TestSourceInsights:
    def _db(self) -> AnalyticsDatabase:
        tmp = tempfile.mkdtemp()
        return AnalyticsDatabase(db_path=str(Path(tmp) / "a.db"))

    def test_empty_db_returns_zero(self) -> None:
        db = self._db()
        out = db.get_source_insights(days=30)
        assert out["total"] == 0
        assert out["rerank_coverage"] == 0.0
        assert out["sources"] == {}

    def test_win_rate_and_positions(self) -> None:
        db = self._db()
        # Two searches: doc wins once, gmail wins once
        db.log_search(
            query="q1",
            scores=[0.9, 0.7, 0.5],
            source_types=["documentation", "gmail_case", "documentation"],
            reranked=True,
        )
        db.log_search(
            query="q2",
            scores=[0.8, 0.6],
            source_types=["gmail_case", "documentation"],
            reranked=True,
        )
        out = db.get_source_insights(days=30)
        assert out["total"] == 2
        assert out["rerank_coverage"] == 100.0
        docs = out["sources"]["documentation"]
        cases = out["sources"]["gmail_case"]
        assert docs["win_rate"] == 50.0
        assert cases["win_rate"] == 50.0
        # documentation appeared at positions 1 & 3 (first query) + position 2 (second)
        assert docs["positions"][0] == 1
        assert docs["positions"][1] == 1
        assert docs["positions"][2] == 1
        assert docs["appearances"] == 3
        # avg score rounded
        assert docs["avg_top5_score"] is not None
        assert 0.0 < docs["avg_top5_score"] < 1.0

    def test_skips_rows_without_source_types(self) -> None:
        db = self._db()
        db.log_search(query="no-source-data")  # no source_types logged
        out = db.get_source_insights(days=30)
        assert out["total"] == 0


class TestLogSearchNewFields:
    def _db(self) -> AnalyticsDatabase:
        tmp = tempfile.mkdtemp()
        return AnalyticsDatabase(db_path=str(Path(tmp) / "b.db"))

    def test_round_trips_new_columns(self) -> None:
        db = self._db()
        sid = db.log_search(
            query="raw",
            normalized_query="raw with RIO +LAN",
            rewritten_query="connect FX6 to RIO",
            pipeline="chat",
            tool_calls=[{"tool": "search_docs", "query": "FX6", "result_count": 3}],
        )
        detail = db.get_search_by_id(sid)
        assert detail is not None
        assert detail["normalized_query"] == "raw with RIO +LAN"
        assert detail["rewritten_query"] == "connect FX6 to RIO"
        assert detail["pipeline"] == "chat"
        assert detail["tool_calls"][0]["tool"] == "search_docs"

    def test_conversation_preview_included(self) -> None:
        db = self._db()
        db.log_search(
            query="q",
            response="# Heading\n\nFirst sentence here. Second sentence.",
            session_id="sess-x",
        )
        convs = db.get_recent_conversations(limit=5)
        assert convs
        assert convs[0]["queries"][0]["response_preview"] == "First sentence here. Second sentence."
