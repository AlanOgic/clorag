"""Prompt Manager service with caching and fallback support.

Provides a unified interface for retrieving LLM prompts, with:
- Database storage for admin-editable prompts
- In-memory caching with TTL for performance
- Fallback to hardcoded defaults when DB prompts not found
- Variable substitution for dynamic prompts
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from threading import Lock
from typing import Any

import structlog

from clorag.config import get_settings
from clorag.core.prompt_db import Prompt, PromptDatabase, PromptVersion, get_prompt_database
from clorag.services.default_prompts import (
    DEFAULT_PROMPTS,
    get_default_prompt,
)

logger = structlog.get_logger(__name__)


# Singleton instance
_prompt_manager: PromptManager | None = None
_manager_lock = Lock()


def get_prompt_manager() -> PromptManager:
    """Get or create the singleton PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        with _manager_lock:
            if _prompt_manager is None:
                _prompt_manager = PromptManager()
    return _prompt_manager


class CacheEntry:
    """Cache entry with TTL support."""

    def __init__(self, content: str, expires_at: float) -> None:
        """Initialize cache entry."""
        self.content = content
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class PromptManager:
    """Manages LLM prompts with caching and fallback support.

    Usage:
        pm = get_prompt_manager()

        # Get prompt with variable substitution
        prompt = pm.get_prompt("analysis.thread_analyzer", thread_content="...")

        # Hot reload after admin edit
        pm.reload_cache("analysis.thread_analyzer")

        # Reload all caches
        pm.reload_all()
    """

    def __init__(
        self,
        db: PromptDatabase | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the prompt manager.

        Args:
            db: PromptDatabase instance (uses singleton if None).
            cache_ttl_seconds: Cache TTL in seconds (uses settings if None).
        """
        self._db = db or get_prompt_database()
        settings = get_settings()
        self._cache_ttl = cache_ttl_seconds or settings.prompts_cache_ttl

        # In-memory cache: key -> CacheEntry
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = Lock()

        # Track cache stats
        self._hits = 0
        self._misses = 0

        logger.info(
            "PromptManager initialized",
            cache_ttl=self._cache_ttl,
            default_prompts=len(DEFAULT_PROMPTS),
        )

    def get_prompt(self, key: str, **variables: Any) -> str:
        """Get a prompt by key with variable substitution.

        Retrieval order:
        1. Check in-memory cache (if not expired)
        2. Query database
        3. Fall back to hardcoded default
        4. Raise KeyError if not found

        Args:
            key: The prompt key (e.g., "analysis.thread_analyzer").
            **variables: Variables to substitute in the prompt.

        Returns:
            The prompt text with variables substituted.

        Raises:
            KeyError: If prompt not found in DB or defaults.
        """
        # Try cache first
        content = self._get_from_cache(key)
        if content is not None:
            self._hits += 1
            return self._substitute_variables(content, variables)

        self._misses += 1

        # Try database
        db_prompt = self._db.get_prompt_by_key(key)
        if db_prompt:
            self._set_cache(key, db_prompt.content)
            return self._substitute_variables(db_prompt.content, variables)

        # Fall back to defaults
        default = get_default_prompt(key)
        if default:
            # Cache the default too
            self._set_cache(key, default.content)
            logger.debug("Using default prompt", key=key)
            return self._substitute_variables(default.content, variables)

        # Not found anywhere
        raise KeyError(f"Prompt not found: {key}")

    def get_prompt_content(self, key: str) -> str:
        """Get raw prompt content without variable substitution.

        Args:
            key: The prompt key.

        Returns:
            The raw prompt text.

        Raises:
            KeyError: If prompt not found.
        """
        return self.get_prompt(key)

    def get_prompt_with_metadata(self, key: str) -> dict[str, Any]:
        """Get prompt with full metadata.

        Args:
            key: The prompt key.

        Returns:
            Dict with prompt data and source info.

        Raises:
            KeyError: If prompt not found.
        """
        # Try database first
        db_prompt = self._db.get_prompt_by_key(key)
        if db_prompt:
            return {
                "source": "database",
                "prompt": db_prompt.to_dict(),
            }

        # Fall back to defaults
        default = get_default_prompt(key)
        if default:
            return {
                "source": "default",
                "prompt": {
                    "key": default.key,
                    "name": default.name,
                    "description": default.description,
                    "model": default.model,
                    "category": default.category,
                    "content": default.content,
                    "variables": default.variables,
                },
            }

        raise KeyError(f"Prompt not found: {key}")

    def list_all_prompts(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all prompts from both database and defaults.

        Args:
            category: Optional category filter.

        Returns:
            List of prompt info dicts with source indicator.
        """
        result: dict[str, dict[str, Any]] = {}

        # Add defaults first (will be overwritten by DB entries)
        for default in DEFAULT_PROMPTS:
            if category and default.category != category:
                continue
            result[default.key] = {
                "key": default.key,
                "name": default.name,
                "description": default.description,
                "model": default.model,
                "category": default.category,
                "variables": default.variables,
                "source": "default",
                "is_customized": False,
            }

        # Override with DB entries
        db_prompts = self._db.list_prompts(category=category)
        for prompt in db_prompts:
            result[prompt.key] = {
                "id": prompt.id,
                "key": prompt.key,
                "name": prompt.name,
                "description": prompt.description,
                "model": prompt.model,
                "category": prompt.category,
                "variables": prompt.variables,
                "is_active": prompt.is_active,
                "created_at": prompt.created_at.isoformat() if prompt.created_at else None,
                "updated_at": prompt.updated_at.isoformat() if prompt.updated_at else None,
                "source": "database",
                "is_customized": True,
            }

        return list(result.values())

    def get_prompt_by_id(self, prompt_id: str) -> Prompt | None:
        """Get a database prompt by ID.

        Args:
            prompt_id: The prompt UUID.

        Returns:
            Prompt object or None.
        """
        return self._db.get_prompt_by_id(prompt_id)

    def get_prompt_versions(self, prompt_id: str) -> list[PromptVersion]:
        """Get version history for a prompt.

        Args:
            prompt_id: The prompt UUID.

        Returns:
            List of versions, newest first.
        """
        return self._db.get_prompt_versions(prompt_id)

    def update_prompt(
        self,
        prompt_id: str,
        content: str | None = None,
        name: str | None = None,
        description: str | None = None,
        model: str | None = None,
        variables: list[str] | None = None,
        change_note: str | None = None,
        updated_by: str | None = None,
    ) -> Prompt | None:
        """Update a prompt and invalidate cache.

        Args:
            prompt_id: The prompt UUID.
            content: New content (creates version if changed).
            name: New name.
            description: New description.
            model: New target model.
            variables: New variables list.
            change_note: Note for version history.
            updated_by: Who made the update.

        Returns:
            Updated Prompt or None if not found.
        """
        prompt = self._db.update_prompt(
            prompt_id=prompt_id,
            content=content,
            name=name,
            description=description,
            model=model,
            variables=variables,
            change_note=change_note,
            updated_by=updated_by,
        )

        if prompt:
            # Invalidate cache for this key
            self._invalidate_cache(prompt.key)
            logger.info("Prompt updated and cache invalidated", key=prompt.key, id=prompt_id)

        return prompt

    def rollback_prompt(
        self,
        prompt_id: str,
        version: int,
        rolled_back_by: str | None = None,
    ) -> Prompt | None:
        """Rollback a prompt to a previous version.

        Args:
            prompt_id: The prompt UUID.
            version: Version number to rollback to.
            rolled_back_by: Who performed the rollback.

        Returns:
            Updated Prompt or None if version not found.
        """
        prompt = self._db.rollback_to_version(prompt_id, version, rolled_back_by)

        if prompt:
            self._invalidate_cache(prompt.key)
            logger.info("Prompt rolled back", key=prompt.key, version=version)

        return prompt

    def initialize_defaults(self, force: bool = False) -> dict[str, int]:
        """Initialize database with default prompts.

        Args:
            force: If True, reset all prompts to defaults (loses customizations).

        Returns:
            Dict with counts: created, updated, skipped.
        """
        created = 0
        updated = 0
        skipped = 0

        for default in DEFAULT_PROMPTS:
            existing = self._db.get_prompt_by_key(default.key)

            if existing and not force:
                # Skip if already exists and not forcing
                skipped += 1
                continue

            if existing and force:
                # Update existing prompt with default content
                self._db.update_prompt(
                    prompt_id=existing.id,
                    content=default.content,
                    name=default.name,
                    description=default.description,
                    model=default.model,
                    variables=default.variables,
                    change_note="Reset to default",
                    updated_by="system",
                )
                updated += 1
            else:
                # Create new prompt
                self._db.create_prompt(
                    key=default.key,
                    name=default.name,
                    content=default.content,
                    category=default.category,
                    description=default.description,
                    model=default.model,
                    variables=default.variables,
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
        """Reload a specific prompt in the cache.

        Args:
            key: The prompt key to reload.
        """
        self._invalidate_cache(key)
        # Pre-warm the cache by fetching it
        try:
            self.get_prompt(key)
            logger.debug("Cache reloaded", key=key)
        except KeyError:
            logger.warning("Prompt not found during cache reload", key=key)

    def reload_all(self) -> None:
        """Clear and reload all cached prompts."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("All prompt caches cleared")

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

    def detect_variables(self, content: str) -> list[str]:
        """Detect variable placeholders in prompt content.

        Args:
            content: Prompt content to analyze.

        Returns:
            List of variable names found.
        """
        # Match {variable_name} patterns (not {{escaped}})
        pattern = r"(?<!\{)\{(\w+)\}(?!\})"
        matches = re.findall(pattern, content)
        return sorted(set(matches))

    def _get_from_cache(self, key: str) -> str | None:
        """Get prompt from cache if valid.

        Args:
            key: The prompt key.

        Returns:
            Cached content or None if not cached/expired.
        """
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.content
        return None

    def _set_cache(self, key: str, content: str) -> None:
        """Add prompt to cache.

        Args:
            key: The prompt key.
            content: The prompt content.
        """
        expires_at = time.time() + self._cache_ttl
        with self._cache_lock:
            self._cache[key] = CacheEntry(content, expires_at)

    def _invalidate_cache(self, key: str) -> None:
        """Remove prompt from cache.

        Args:
            key: The prompt key.
        """
        with self._cache_lock:
            self._cache.pop(key, None)

    def _substitute_variables(self, content: str, variables: dict[str, Any]) -> str:
        """Substitute variables in prompt content.

        Uses str.format() style substitution.

        Args:
            content: The prompt content with {variable} placeholders.
            variables: Dict of variable name -> value.

        Returns:
            Content with variables substituted.
        """
        if not variables:
            return content

        try:
            return content.format(**variables)
        except KeyError as e:
            logger.warning("Missing variable in prompt", missing=str(e))
            # Return content with unsubstituted variables
            return content


# =============================================================================
# Convenience functions for backward compatibility
# =============================================================================


@lru_cache(maxsize=1)
def _get_manager() -> PromptManager:
    """Cached manager getter for module-level functions."""
    return get_prompt_manager()


def get_prompt(key: str, **variables: Any) -> str:
    """Convenience function to get a prompt.

    Args:
        key: The prompt key.
        **variables: Variables to substitute.

    Returns:
        The prompt text.
    """
    return _get_manager().get_prompt(key, **variables)
