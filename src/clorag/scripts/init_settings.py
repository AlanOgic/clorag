#!/usr/bin/env python3
"""Initialize the settings database with default RAG tuning parameters.

Usage:
    uv run init-settings           # Initialize (skip existing)
    uv run init-settings --force   # Reset all to defaults
    uv run init-settings --list    # List all settings
    uv run init-settings --stats   # Show statistics
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from clorag.services.default_settings import DEFAULT_SETTINGS
from clorag.services.settings_manager import get_settings_manager
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


def list_settings(category: str | None = None) -> None:
    """List all settings from database and defaults."""
    sm = get_settings_manager()
    settings = sm.get_all(category=category)

    if not settings:
        print("No settings found.")
        return

    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for s in settings:
        cat = s["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(s)

    for cat in sorted(by_category.keys()):
        print(f"\n=== {cat.upper()} ===")
        for s in by_category[cat]:
            source = "DB" if s["source"] == "database" else "DEF"
            restart = " [RESTART]" if s.get("requires_restart") else ""
            value = s.get("value", s.get("default_value", "?"))
            print(f"  [{source}] {s['key']:45} = {value:>8}  ({s['value_type']:5}){restart}")


def show_stats() -> None:
    """Show settings database statistics."""
    sm = get_settings_manager()

    from clorag.core.settings_db import get_settings_database
    db = get_settings_database()
    db_stats = db.get_stats()

    cache_stats = sm.get_cache_stats()

    print("\n=== SETTINGS DATABASE STATS ===")
    print(f"Total settings in DB: {db_stats['total']}")
    print(f"Total versions:       {db_stats['versions_count']}")
    print("\nBy category:")
    by_category = db_stats.get("by_category", {})
    if isinstance(by_category, dict):
        for cat, count in by_category.items():
            print(f"  {cat}: {count}")

    print("\n=== DEFAULT SETTINGS ===")
    print(f"Total defaults:       {len(DEFAULT_SETTINGS)}")
    by_cat: dict[str, int] = {}
    for s in DEFAULT_SETTINGS:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    print("\n=== CACHE STATS ===")
    print(f"Hits:       {cache_stats['hits']}")
    print(f"Misses:     {cache_stats['misses']}")
    print(f"Hit rate:   {cache_stats['hit_rate']:.1%}")
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"TTL:        {cache_stats['ttl_seconds']}s")


def initialize_settings(force: bool = False) -> None:
    """Initialize database with default settings."""
    sm = get_settings_manager()

    print(f"Initializing settings (force={force})...")
    result = sm.initialize_defaults(force=force)

    print("\nResults:")
    print(f"  Created: {result['created']}")
    print(f"  Updated: {result['updated']}")
    print(f"  Skipped: {result['skipped']}")

    if result['created'] > 0 or result['updated'] > 0:
        print("\nSettings initialized successfully!")
    elif result['skipped'] > 0:
        print("\nAll settings already exist. Use --force to reset to defaults.")


def export_setting(key: str) -> None:
    """Export a specific setting to stdout."""
    sm = get_settings_manager()
    all_settings = sm.get_all()

    setting = next((s for s in all_settings if s["key"] == key), None)
    if not setting:
        print(f"Error: Setting '{key}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"# {key}")
    print(f"# Source: {setting['source']}")
    print(f"# Type: {setting['value_type']}")
    print(f"# Default: {setting.get('default_value', '?')}")
    if setting.get('min_value') is not None:
        print(f"# Min: {setting['min_value']}")
    if setting.get('max_value') is not None:
        print(f"# Max: {setting['max_value']}")
    print(f"# Requires Restart: {setting.get('requires_restart', False)}")
    print("# " + "=" * 60)
    print(setting.get("value", setting.get("default_value", "")))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize and manage RAG tuning settings for CLORAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run init-settings                    # Initialize database with defaults
  uv run init-settings --force            # Reset all settings to defaults
  uv run init-settings --list             # List all settings
  uv run init-settings --category caches  # List settings in category
  uv run init-settings --stats            # Show statistics
  uv run init-settings --export retrieval.short_query_threshold
        """,
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reset all settings to defaults (WARNING: loses customizations)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all settings",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["retrieval", "reranking", "synthesis", "caches", "prefetch"],
        help="Filter by category (with --list)",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show database and cache statistics",
    )
    parser.add_argument(
        "--export", "-e",
        metavar="KEY",
        help="Export a specific setting by key",
    )

    args = parser.parse_args()

    if args.list:
        list_settings(category=args.category)
    elif args.stats:
        show_stats()
    elif args.export:
        export_setting(args.export)
    else:
        initialize_settings(force=args.force)


if __name__ == "__main__":
    main()
