"""Settings Manager service with caching and fallback support.

Provides a unified interface for retrieving RAG tuning settings, with:
- Database storage for admin-editable settings
- In-memory caching with TTL for performance
- Fallback to hardcoded defaults when DB settings not found
- Type-safe convenience getters
"""

from __future__ import annotations

import time
from functools import lru_cache
from threading import Lock
from typing import Any

import structlog

from clorag.config import get_settings
from clorag.core.settings_db import Setting, SettingsDatabase, SettingVersion, get_settings_database
from clorag.services.default_settings import (
    DEFAULT_SETTINGS,
    get_default_setting,
)

logger = structlog.get_logger(__name__)


# Singleton instance
_settings_manager: SettingsManager | None = None
_manager_lock = Lock()


def get_settings_manager() -> SettingsManager:
    """Get or create the singleton SettingsManager instance."""
    global _settings_manager
    if _settings_manager is None:
        with _manager_lock:
            if _settings_manager is None:
                _settings_manager = SettingsManager()
    return _settings_manager


class CacheEntry:
    """Cache entry with TTL support."""

    def __init__(self, value: int | float | bool, expires_at: float) -> None:
        """Initialize cache entry."""
        self.value = value
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class SettingsManager:
    """Manages RAG tuning settings with caching and fallback support.

    Usage:
        sm = get_settings_manager()

        # Get a typed setting value
        threshold = sm.get_float("retrieval.short_query_threshold")

        # Hot reload after admin edit
        sm.reload_cache("retrieval.short_query_threshold")

        # Reload all caches
        sm.reload_all()
    """

    def __init__(
        self,
        db: SettingsDatabase | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the settings manager.

        Args:
            db: SettingsDatabase instance (uses singleton if None).
            cache_ttl_seconds: Cache TTL in seconds (uses settings if None).
        """
        self._db = db or get_settings_database()
        settings = get_settings()
        self._cache_ttl = cache_ttl_seconds or settings.prompts_cache_ttl

        # In-memory cache: key -> CacheEntry
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = Lock()

        # Track cache stats
        self._hits = 0
        self._misses = 0

        logger.info(
            "SettingsManager initialized",
            cache_ttl=self._cache_ttl,
            default_settings=len(DEFAULT_SETTINGS),
        )

    def get(self, key: str) -> int | float | bool:
        """Get a setting value by key.

        Retrieval order:
        1. Check in-memory cache (if not expired)
        2. Query database (typed_value)
        3. Fall back to hardcoded default (parse default_value)
        4. Raise KeyError if not found

        Args:
            key: The setting key (e.g., "retrieval.short_query_threshold").

        Returns:
            The typed setting value (int, float, or bool).

        Raises:
            KeyError: If setting not found in DB or defaults.
        """
        # Try cache first
        cached = self._get_from_cache(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1

        # Try database
        db_setting = self._db.get_by_key(key)
        if db_setting:
            value = db_setting.typed_value
            self._set_cache(key, value)
            return value

        # Fall back to defaults
        default = get_default_setting(key)
        if default:
            value = self._parse_value(default.default_value, default.value_type)
            self._set_cache(key, value)
            logger.debug("Using default setting", key=key)
            return value

        # Not found anywhere
        raise KeyError(f"Setting not found: {key}")

    def get_int(self, key: str) -> int:
        """Get a setting value as int.

        Args:
            key: The setting key.

        Returns:
            The setting value as int.

        Raises:
            KeyError: If setting not found.
        """
        return int(self.get(key))

    def get_float(self, key: str) -> float:
        """Get a setting value as float.

        Args:
            key: The setting key.

        Returns:
            The setting value as float.

        Raises:
            KeyError: If setting not found.
        """
        return float(self.get(key))

    def get_bool(self, key: str) -> bool:
        """Get a setting value as bool.

        Args:
            key: The setting key.

        Returns:
            The setting value as bool.

        Raises:
            KeyError: If setting not found.
        """
        return bool(self.get(key))

    def get_all(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all settings from both database and defaults.

        Args:
            category: Optional category filter.

        Returns:
            List of setting info dicts with source indicator.
        """
        result: dict[str, dict[str, Any]] = {}

        # Add defaults first (will be overwritten by DB entries)
        for default in DEFAULT_SETTINGS:
            if category and default.category != category:
                continue
            result[default.key] = {
                "key": default.key,
                "name": default.name,
                "description": default.description,
                "category": default.category,
                "value_type": default.value_type,
                "value": default.default_value,
                "default_value": default.default_value,
                "min_value": default.min_value,
                "max_value": default.max_value,
                "requires_restart": default.requires_restart,
                "source": "default",
                "is_customized": False,
            }

        # Override with DB entries
        db_settings = self._db.list_settings(category=category)
        for setting in db_settings:
            result[setting.key] = {
                "id": setting.id,
                "key": setting.key,
                "name": setting.name,
                "description": setting.description,
                "category": setting.category,
                "value_type": setting.value_type,
                "value": setting.value,
                "typed_value": setting.typed_value,
                "default_value": setting.default_value,
                "min_value": setting.min_value,
                "max_value": setting.max_value,
                "requires_restart": setting.requires_restart,
                "is_active": setting.is_active,
                "created_at": setting.created_at.isoformat() if setting.created_at else None,
                "updated_at": setting.updated_at.isoformat() if setting.updated_at else None,
                "updated_by": setting.updated_by,
                "source": "database" if setting.value != setting.default_value else "default",
                "is_customized": setting.value != setting.default_value,
            }

        return list(result.values())

    def get_setting_by_id(self, setting_id: str) -> Setting | None:
        """Get a database setting by ID.

        Args:
            setting_id: The setting UUID.

        Returns:
            Setting object or None.
        """
        return self._db.get_by_id(setting_id)

    def get_setting_versions(self, setting_id: str) -> list[SettingVersion]:
        """Get version history for a setting.

        Args:
            setting_id: The setting UUID.

        Returns:
            List of versions, newest first.
        """
        return self._db.get_versions(setting_id)

    def update(
        self,
        setting_id: str,
        value: str,
        change_note: str | None = None,
        updated_by: str | None = None,
    ) -> Setting | None:
        """Update a setting value and invalidate cache.

        Args:
            setting_id: The setting UUID.
            value: New value (as string).
            change_note: Note for version history.
            updated_by: Who made the update.

        Returns:
            Updated Setting or None if not found.

        Raises:
            ValueError: If value fails type or bounds validation.
        """
        setting = self._db.update_setting(
            setting_id=setting_id,
            value=value,
            change_note=change_note,
            updated_by=updated_by,
        )

        if setting:
            # Invalidate cache for this key
            self._invalidate_cache(setting.key)
            logger.info("Setting updated and cache invalidated", key=setting.key, id=setting_id)

        return setting

    def rollback_setting(
        self,
        setting_id: str,
        version: int,
        rolled_back_by: str | None = None,
    ) -> Setting | None:
        """Rollback a setting to a previous version.

        Args:
            setting_id: The setting UUID.
            version: Version number to rollback to.
            rolled_back_by: Who performed the rollback.

        Returns:
            Updated Setting or None if version not found.
        """
        setting = self._db.rollback_to_version(setting_id, version, rolled_back_by)

        if setting:
            self._invalidate_cache(setting.key)
            logger.info("Setting rolled back", key=setting.key, version=version)

        return setting

    def initialize_defaults(self, force: bool = False) -> dict[str, int]:
        """Initialize database with default settings.

        Args:
            force: If True, reset all settings to defaults (loses customizations).

        Returns:
            Dict with counts: created, updated, skipped.
        """
        created = 0
        updated = 0
        skipped = 0

        for default in DEFAULT_SETTINGS:
            existing = self._db.get_by_key(default.key)

            if existing and not force:
                # Skip if already exists and not forcing
                skipped += 1
                continue

            if existing and force:
                # Reset value AND sync metadata (name, description, bounds…)
                self._db.upsert_setting(
                    key=default.key,
                    name=default.name,
                    description=default.description,
                    category=default.category,
                    value_type=default.value_type,
                    value=default.default_value,
                    default_value=default.default_value,
                    min_value=default.min_value,
                    max_value=default.max_value,
                    requires_restart=default.requires_restart,
                    created_by="system",
                )
                updated += 1
            else:
                # Create new setting via upsert
                self._db.upsert_setting(
                    key=default.key,
                    name=default.name,
                    description=default.description,
                    category=default.category,
                    value_type=default.value_type,
                    value=default.default_value,
                    default_value=default.default_value,
                    min_value=default.min_value,
                    max_value=default.max_value,
                    requires_restart=default.requires_restart,
                    created_by="system",
                )
                created += 1

        # Clear entire cache after initialization
        self.reload_all()

        logger.info(
            "Defaults initialized",
            created=created,
            updated=updated,
            skipped=skipped,
            force=force,
        )

        return {"created": created, "updated": updated, "skipped": skipped}

    def reload_cache(self, key: str) -> None:
        """Reload a specific setting in the cache.

        Args:
            key: The setting key to reload.
        """
        self._invalidate_cache(key)
        # Pre-warm the cache by fetching it
        try:
            self.get(key)
            logger.debug("Cache reloaded", key=key)
        except KeyError:
            logger.warning("Setting not found during cache reload", key=key)

    def reload_all(self) -> None:
        """Clear and reload all cached settings."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("All settings caches cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            # Count non-expired entries
            active_entries = sum(
                1 for entry in self._cache.values() if not entry.is_expired()
            )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_rate": round(hit_rate, 3),
                "cache_size": len(self._cache),
                "active_entries": active_entries,
                "ttl_seconds": self._cache_ttl,
            }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_from_cache(self, key: str) -> int | float | bool | None:
        """Get setting from cache if valid.

        Args:
            key: The setting key.

        Returns:
            Cached value or None if not cached/expired.
        """
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.value
        return None

    def _set_cache(self, key: str, value: int | float | bool) -> None:
        """Add setting to cache.

        Args:
            key: The setting key.
            value: The typed setting value.
        """
        expires_at = time.time() + self._cache_ttl
        with self._cache_lock:
            self._cache[key] = CacheEntry(value, expires_at)

    def _invalidate_cache(self, key: str) -> None:
        """Remove setting from cache.

        Args:
            key: The setting key.
        """
        with self._cache_lock:
            self._cache.pop(key, None)

    @staticmethod
    def _parse_value(value_str: str, value_type: str) -> int | float | bool:
        """Convert a string value to its declared type.

        Args:
            value_str: The string representation of the value.
            value_type: One of "int", "float", "bool".

        Returns:
            The parsed typed value.

        Raises:
            ValueError: If the value cannot be parsed.
        """
        if value_type == "int":
            return int(value_str)
        elif value_type == "float":
            return float(value_str)
        elif value_type == "bool":
            return value_str.lower() in ("true", "1", "yes")
        else:
            raise ValueError(f"Unknown value_type: {value_type}")


# =============================================================================
# Convenience functions
# =============================================================================


@lru_cache(maxsize=1)
def _get_manager() -> SettingsManager:
    """Cached manager getter for module-level functions."""
    return get_settings_manager()


def get_setting(key: str) -> int | float | bool:
    """Convenience function to get a setting value.

    Args:
        key: The setting key.

    Returns:
        The typed setting value.
    """
    return _get_manager().get(key)
