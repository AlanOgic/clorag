"""Performance metrics collection for RAG pipeline observability."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Metrics retention settings
METRICS_WINDOW_SIZE = 1000  # Keep last 1000 measurements per metric
PERCENTILES = [50, 90, 95, 99]  # Percentiles to calculate


@dataclass
class TimingMetric:
    """A single timing measurement."""

    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector with sliding window statistics.

    Collects timing measurements for different pipeline stages and provides
    aggregated statistics including percentiles, averages, and throughput.
    """

    def __init__(self, window_size: int = METRICS_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._metrics: dict[str, deque[TimingMetric]] = {}
        self._lock = Lock()
        self._total_queries = 0
        self._error_count = 0

    def record(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a timing measurement.

        Args:
            name: Metric name (e.g., 'embedding_generation', 'vector_search').
            duration_ms: Duration in milliseconds.
            metadata: Optional metadata (query length, result count, etc.).
        """
        metric = TimingMetric(
            name=name,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self._window_size)
            self._metrics[name].append(metric)

    def record_query(self) -> None:
        """Increment total query counter."""
        with self._lock:
            self._total_queries += 1

    def record_error(self) -> None:
        """Increment error counter."""
        with self._lock:
            self._error_count += 1

    @contextmanager
    def measure(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        log_slow_threshold_ms: float | None = None,
        log_always: bool = False,
    ) -> Generator[None, None, None]:
        """Context manager to measure and record execution time.

        Args:
            name: Metric name.
            metadata: Optional metadata to attach.
            log_slow_threshold_ms: If set, log warning when duration exceeds threshold.
            log_always: If True, log debug message for every measurement.

        Yields:
            None - timing is recorded on context exit.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(name, duration_ms, metadata)

            if log_slow_threshold_ms and duration_ms > log_slow_threshold_ms:
                logger.warning(
                    "Slow operation detected",
                    metric=name,
                    duration_ms=round(duration_ms, 2),
                    threshold_ms=log_slow_threshold_ms,
                    metadata=metadata,
                )
            elif log_always:
                logger.debug(
                    "Performance measurement",
                    metric=name,
                    duration_ms=round(duration_ms, 2),
                    metadata=metadata,
                )

    def get_stats(self, name: str) -> dict[str, Any] | None:
        """Get statistics for a specific metric.

        Args:
            name: Metric name.

        Returns:
            Dict with count, avg, min, max, and percentiles, or None if no data.
        """
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return None

            durations = [m.duration_ms for m in self._metrics[name]]

        count = len(durations)
        sorted_durations = sorted(durations)

        stats = {
            "count": count,
            "avg_ms": round(sum(durations) / count, 2),
            "min_ms": round(min(durations), 2),
            "max_ms": round(max(durations), 2),
        }

        # Calculate percentiles
        for p in PERCENTILES:
            idx = int(count * p / 100)
            idx = min(idx, count - 1)
            stats[f"p{p}_ms"] = round(sorted_durations[idx], 2)

        return stats

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all metrics.

        Returns:
            Dict mapping metric names to their statistics.
        """
        with self._lock:
            metric_names = list(self._metrics.keys())
            total_queries = self._total_queries
            error_count = self._error_count

        result: dict[str, Any] = {
            "total_queries": total_queries,
            "error_count": error_count,
            "error_rate_percent": (
                round(error_count / total_queries * 100, 2) if total_queries > 0 else 0
            ),
            "metrics": {},
        }

        for name in metric_names:
            stats = self.get_stats(name)
            if stats:
                result["metrics"][name] = stats

        return result

    def get_recent(self, name: str, count: int = 10) -> list[dict[str, Any]]:
        """Get recent measurements for a metric.

        Args:
            name: Metric name.
            count: Number of recent measurements to return.

        Returns:
            List of recent measurements as dicts.
        """
        with self._lock:
            if name not in self._metrics:
                return []

            recent = list(self._metrics[name])[-count:]

        return [
            {
                "duration_ms": round(m.duration_ms, 2),
                "timestamp": m.timestamp,
                "metadata": m.metadata,
            }
            for m in recent
        ]

    def clear(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._total_queries = 0
            self._error_count = 0


# Singleton instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the singleton MetricsCollector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions for common metrics
def measure_embedding_generation(
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    """Measure embedding generation time (target: <200ms)."""
    return get_metrics_collector().measure(
        "embedding_generation",
        metadata=metadata,
        log_slow_threshold_ms=500,
    )


def measure_vector_search(metadata: dict[str, Any] | None = None) -> AbstractContextManager[None]:
    """Measure vector search time (target: <100ms)."""
    return get_metrics_collector().measure(
        "vector_search",
        metadata=metadata,
        log_slow_threshold_ms=300,
    )


def measure_graph_enrichment(
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    """Measure graph enrichment time (target: <50ms)."""
    return get_metrics_collector().measure(
        "graph_enrichment",
        metadata=metadata,
        log_slow_threshold_ms=200,
    )


def measure_total_search(metadata: dict[str, Any] | None = None) -> AbstractContextManager[None]:
    """Measure total search pipeline time (target: <500ms)."""
    return get_metrics_collector().measure(
        "total_search",
        metadata=metadata,
        log_slow_threshold_ms=1000,
    )


def measure_llm_synthesis(metadata: dict[str, Any] | None = None) -> AbstractContextManager[None]:
    """Measure LLM synthesis time (TTFB target: <2s)."""
    return get_metrics_collector().measure(
        "llm_synthesis",
        metadata=metadata,
        log_slow_threshold_ms=5000,
    )
