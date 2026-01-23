"""Thread-safe LRU cache with optional TTL support.

This module provides a generic cache implementation that can be used across
the codebase to reduce code duplication and ensure consistent caching behavior.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with optional TTL expiration.

    This cache is designed to be used for caching embeddings, rerank results,
    and other expensive computations. It supports:
    - LRU eviction when at capacity
    - Optional TTL-based expiration
    - Thread-safe operations
    - Hit/miss statistics
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of items to store.
            ttl_seconds: Optional time-to-live for cached items in seconds.
                        If None, items never expire (pure LRU).
        """
        self._cache: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """Get a value from cache if exists and not expired.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            timestamp, value = self._cache[key]

            # Check TTL if enabled
            if self._ttl is not None and time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        """Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                del self._cache[key]
            elif len(self._cache) >= self._max_size:
                # Evict oldest (first) item
                self._cache.popitem(last=False)

            self._cache[key] = (time.time(), value)

    def invalidate(self, key: str | None = None, pattern: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate.
            pattern: If provided, invalidate keys containing this pattern.
                    If both key and pattern are None, invalidate all entries.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            if key is not None:
                if key in self._cache:
                    del self._cache[key]
                    return 1
                return 0

            if pattern is not None:
                keys_to_delete = [k for k in self._cache if pattern in k]
                for k in keys_to_delete:
                    del self._cache[k]
                return len(keys_to_delete)

            # Invalidate all
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dict with size, hits, misses, and hit_rate_percent.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
            }

    def clear_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)


def make_cache_key(*args: str | int | float | None) -> str:
    """Create a hash-based cache key from multiple arguments.

    This is useful for creating cache keys from query parameters.

    Args:
        *args: Values to include in the key.

    Returns:
        32-character hex string hash.

    Example:
        key = make_cache_key(query, model, dimensions)
    """
    key_str = ":".join(str(arg) for arg in args if arg is not None)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]
