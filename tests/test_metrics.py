"""Tests for the performance metrics collection module."""

import time
from unittest.mock import patch

import pytest

from clorag.core.metrics import (
    MetricsCollector,
    get_metrics_collector,
    measure_embedding_generation,
    measure_total_search,
    measure_vector_search,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_record_metric(self):
        """Test recording a single metric."""
        collector = MetricsCollector()
        collector.record("test_metric", 100.5, {"key": "value"})

        stats = collector.get_stats("test_metric")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["avg_ms"] == 100.5
        assert stats["min_ms"] == 100.5
        assert stats["max_ms"] == 100.5

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics."""
        collector = MetricsCollector()
        collector.record("metric1", 100.0)
        collector.record("metric1", 200.0)
        collector.record("metric1", 300.0)

        stats = collector.get_stats("metric1")
        assert stats["count"] == 3
        assert stats["avg_ms"] == 200.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 300.0

    def test_percentile_calculation(self):
        """Test percentile calculations."""
        collector = MetricsCollector()
        # Record 100 values from 1 to 100
        for i in range(1, 101):
            collector.record("percentile_test", float(i))

        stats = collector.get_stats("percentile_test")
        # Allow for minor variations in percentile calculation (index-based)
        assert 49.0 <= stats["p50_ms"] <= 51.0
        assert 89.0 <= stats["p90_ms"] <= 91.0
        assert 94.0 <= stats["p95_ms"] <= 96.0
        assert 98.0 <= stats["p99_ms"] <= 100.0

    def test_sliding_window(self):
        """Test that sliding window evicts old entries."""
        collector = MetricsCollector(window_size=10)

        # Record 15 values
        for i in range(15):
            collector.record("window_test", float(i))

        stats = collector.get_stats("window_test")
        assert stats["count"] == 10  # Only last 10 retained
        assert stats["min_ms"] == 5.0  # First 5 evicted
        assert stats["max_ms"] == 14.0

    def test_query_counter(self):
        """Test query counter increments."""
        collector = MetricsCollector()
        collector.record_query()
        collector.record_query()
        collector.record_query()

        all_stats = collector.get_all_stats()
        assert all_stats["total_queries"] == 3

    def test_error_counter(self):
        """Test error counter and error rate calculation."""
        collector = MetricsCollector()
        collector.record_query()
        collector.record_query()
        collector.record_query()
        collector.record_query()
        collector.record_error()

        all_stats = collector.get_all_stats()
        assert all_stats["error_count"] == 1
        assert all_stats["error_rate_percent"] == 25.0

    def test_get_stats_nonexistent(self):
        """Test getting stats for nonexistent metric returns None."""
        collector = MetricsCollector()
        stats = collector.get_stats("nonexistent")
        assert stats is None

    def test_get_recent(self):
        """Test getting recent measurements."""
        collector = MetricsCollector()
        for i in range(5):
            collector.record("recent_test", float(i * 10))

        recent = collector.get_recent("recent_test", count=3)
        assert len(recent) == 3
        # Most recent should be last
        assert recent[-1]["duration_ms"] == 40.0
        assert recent[0]["duration_ms"] == 20.0

    def test_get_recent_nonexistent(self):
        """Test getting recent for nonexistent metric returns empty list."""
        collector = MetricsCollector()
        recent = collector.get_recent("nonexistent")
        assert recent == []

    def test_clear(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        collector.record("test", 100.0)
        collector.record_query()
        collector.record_error()

        collector.clear()

        all_stats = collector.get_all_stats()
        assert all_stats["total_queries"] == 0
        assert all_stats["error_count"] == 0
        assert all_stats["metrics"] == {}


class TestMeasureContextManager:
    """Tests for the measure context manager."""

    def test_measure_records_duration(self):
        """Test that measure context manager records duration."""
        collector = MetricsCollector()

        with collector.measure("timed_op"):
            time.sleep(0.01)  # 10ms

        stats = collector.get_stats("timed_op")
        assert stats is not None
        assert stats["count"] == 1
        # Should be at least 10ms
        assert stats["avg_ms"] >= 10.0

    def test_measure_with_metadata(self):
        """Test measure records metadata."""
        collector = MetricsCollector()

        with collector.measure("with_meta", metadata={"query_len": 50}):
            pass

        recent = collector.get_recent("with_meta", count=1)
        assert len(recent) == 1
        assert recent[0]["metadata"]["query_len"] == 50

    def test_measure_logs_slow_operations(self):
        """Test that slow operations are logged as warnings."""
        collector = MetricsCollector()

        with patch("clorag.core.metrics.logger") as mock_logger:
            with collector.measure("slow_op", log_slow_threshold_ms=5):
                time.sleep(0.02)  # 20ms, exceeds 5ms threshold

            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args[1]
            assert call_kwargs["metric"] == "slow_op"
            assert call_kwargs["threshold_ms"] == 5

    def test_measure_no_log_under_threshold(self):
        """Test that operations under threshold are not logged as warnings."""
        collector = MetricsCollector()

        with patch("clorag.core.metrics.logger") as mock_logger:
            with collector.measure("fast_op", log_slow_threshold_ms=1000):
                pass  # Nearly instant

            mock_logger.warning.assert_not_called()

    def test_measure_log_always(self):
        """Test log_always flag logs debug messages."""
        collector = MetricsCollector()

        with patch("clorag.core.metrics.logger") as mock_logger:
            with collector.measure("debug_op", log_always=True):
                pass

            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["metric"] == "debug_op"


class TestConvenienceFunctions:
    """Tests for convenience measurement functions."""

    def test_measure_embedding_generation(self):
        """Test embedding generation measurement helper."""
        collector = get_metrics_collector()
        collector.clear()

        with measure_embedding_generation(metadata={"query_length": 100}):
            time.sleep(0.001)

        stats = collector.get_stats("embedding_generation")
        assert stats is not None
        assert stats["count"] == 1

    def test_measure_vector_search(self):
        """Test vector search measurement helper."""
        collector = get_metrics_collector()
        collector.clear()

        with measure_vector_search(metadata={"collection": "docs"}):
            time.sleep(0.001)

        stats = collector.get_stats("vector_search")
        assert stats is not None
        assert stats["count"] == 1

    def test_measure_total_search(self):
        """Test total search measurement helper."""
        collector = get_metrics_collector()
        collector.clear()

        with measure_total_search(metadata={"source": "all"}):
            time.sleep(0.001)

        stats = collector.get_stats("total_search")
        assert stats is not None
        assert stats["count"] == 1


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_metrics_collector_returns_same_instance(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test that concurrent recording doesn't lose data."""
        import threading

        collector = MetricsCollector()
        num_threads = 10
        records_per_thread = 100

        def record_metrics():
            for i in range(records_per_thread):
                collector.record("concurrent_test", float(i))

        threads = [threading.Thread(target=record_metrics) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = collector.get_stats("concurrent_test")
        # With window_size=1000, we might not have all 1000 if eviction happened
        # But we should have at least the window size worth of records
        assert stats["count"] >= min(num_threads * records_per_thread, 1000)
