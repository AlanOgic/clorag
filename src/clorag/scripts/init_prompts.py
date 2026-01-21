#!/usr/bin/env python3
"""Initialize the prompt database with default prompts.

Usage:
    uv run init-prompts           # Initialize (skip existing)
    uv run init-prompts --force   # Reset all to defaults
    uv run init-prompts --list    # List all prompts
    uv run init-prompts --stats   # Show statistics
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from clorag.services.default_prompts import DEFAULT_PROMPTS
from clorag.services.prompt_manager import get_prompt_manager
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


def list_prompts(category: str | None = None) -> None:
    """List all prompts from database and defaults."""
    pm = get_prompt_manager()
    prompts = pm.list_all_prompts(category=category)

    if not prompts:
        print("No prompts found.")
        return

    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for p in prompts:
        cat = p["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    for cat in sorted(by_category.keys()):
        print(f"\n=== {cat.upper()} ===")
        for p in by_category[cat]:
            source = "DB" if p["source"] == "database" else "DEF"
            model = p.get("model", "?")
            vars_count = len(p.get("variables", []))
            print(f"  [{source}] {p['key']:40} ({model:6}) {vars_count} vars")


def show_stats() -> None:
    """Show prompt database statistics."""
    pm = get_prompt_manager()

    # Database stats
    from clorag.core.prompt_db import get_prompt_database
    db = get_prompt_database()
    db_stats = db.get_stats()

    # Cache stats
    cache_stats = pm.get_cache_stats()

    print("\n=== PROMPT DATABASE STATS ===")
    print(f"Total prompts in DB: {db_stats['total']}")
    print(f"Total versions:      {db_stats['versions_count']}")
    print("\nBy category:")
    by_category = db_stats.get("by_category", {})
    if isinstance(by_category, dict):
        for cat, count in by_category.items():
            print(f"  {cat}: {count}")

    print("\n=== DEFAULT PROMPTS ===")
    print(f"Total defaults:      {len(DEFAULT_PROMPTS)}")
    by_cat: dict[str, int] = {}
    for p in DEFAULT_PROMPTS:
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    print("\n=== CACHE STATS ===")
    print(f"Hits:       {cache_stats['hits']}")
    print(f"Misses:     {cache_stats['misses']}")
    print(f"Hit rate:   {cache_stats['hit_rate']:.1%}")
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"TTL:        {cache_stats['ttl_seconds']}s")


def initialize_prompts(force: bool = False) -> None:
    """Initialize database with default prompts."""
    pm = get_prompt_manager()

    print(f"Initializing prompts (force={force})...")
    result = pm.initialize_defaults(force=force)

    print("\nResults:")
    print(f"  Created: {result['created']}")
    print(f"  Updated: {result['updated']}")
    print(f"  Skipped: {result['skipped']}")

    if result['created'] > 0 or result['updated'] > 0:
        print("\nPrompts initialized successfully!")
    elif result['skipped'] > 0:
        print("\nAll prompts already exist. Use --force to reset to defaults.")


def export_prompt(key: str) -> None:
    """Export a specific prompt to stdout."""
    pm = get_prompt_manager()

    try:
        data = pm.get_prompt_with_metadata(key)
        prompt = data["prompt"]
        source = data["source"]

        print(f"# {key}")
        print(f"# Source: {source}")
        print(f"# Model: {prompt.get('model', 'unknown')}")
        print(f"# Variables: {prompt.get('variables', [])}")
        print("# " + "=" * 60)
        print(prompt["content"])
    except KeyError:
        print(f"Error: Prompt '{key}' not found", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize and manage LLM prompts for CLORAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run init-prompts                 # Initialize database with defaults
  uv run init-prompts --force         # Reset all prompts to defaults
  uv run init-prompts --list          # List all prompts
  uv run init-prompts --category analysis  # List prompts in category
  uv run init-prompts --stats         # Show statistics
  uv run init-prompts --export analysis.thread_analyzer  # Export prompt
        """,
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reset all prompts to defaults (WARNING: loses customizations)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all prompts",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["agent", "analysis", "synthesis", "drafts", "graph", "scripts"],
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
        help="Export a specific prompt by key",
    )

    args = parser.parse_args()

    if args.list:
        list_prompts(category=args.category)
    elif args.stats:
        show_stats()
    elif args.export:
        export_prompt(args.export)
    else:
        initialize_prompts(force=args.force)


if __name__ == "__main__":
    main()
