#!/usr/bin/env python3
"""Initialize the prompt database with default prompts.

Usage:
    uv run init-prompts           # Initialize (skip existing)
    uv run init-prompts --force   # Reset all to defaults (auto-backups first)
    uv run init-prompts --list    # List all prompts
    uv run init-prompts --stats   # Show statistics
    uv run init-prompts --backup  # Export customized prompts to JSON
    uv run init-prompts --restore FILE  # Restore customizations from backup
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from clorag.services.default_prompts import DEFAULT_PROMPTS, get_default_prompt
from clorag.services.prompt_manager import get_prompt_manager
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

# Default backup directory
BACKUP_DIR = Path("data/prompt_backups")


def _get_customized_prompts() -> list[dict[str, Any]]:
    """Find all prompts whose DB content differs from the hardcoded default.

    Returns:
        List of dicts with key, content, name, description, model, category, variables.
    """
    pm = get_prompt_manager()
    customized: list[dict[str, Any]] = []

    for default in DEFAULT_PROMPTS:
        try:
            data = pm.get_prompt_with_metadata(default.key)
        except KeyError:
            continue

        if data["source"] != "database":
            continue

        db_content = data["prompt"]["content"]
        if db_content != default.content:
            customized.append({
                "key": default.key,
                "name": data["prompt"].get("name", default.name),
                "description": data["prompt"].get("description"),
                "model": data["prompt"].get("model"),
                "category": default.category,
                "variables": data["prompt"].get("variables", []),
                "content": db_content,
            })

    return customized


def backup_prompts(output_path: str | None = None) -> Path | None:
    """Export all customized prompts to a JSON file.

    Args:
        output_path: Optional explicit path. Defaults to data/prompt_backups/<timestamp>.json.

    Returns:
        Path to the backup file, or None if nothing to backup.
    """
    customized = _get_customized_prompts()

    if not customized:
        print("No customized prompts found (all match defaults).")
        return None

    if output_path:
        backup_path = Path(output_path)
    else:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"prompts_{timestamp}.json"

    backup_data = {
        "backed_up_at": datetime.now().isoformat(),
        "prompt_count": len(customized),
        "prompts": customized,
    }

    backup_path.write_text(json.dumps(backup_data, indent=2, ensure_ascii=False))
    print(f"\nBacked up {len(customized)} customized prompt(s) to: {backup_path}")

    for p in customized:
        print(f"  {p['key']}")

    return backup_path


def restore_prompts(backup_file: str) -> None:
    """Restore customized prompts from a backup file.

    Applies saved content on top of whatever is currently in the DB.
    Only restores prompts whose keys still exist in the default registry.

    Args:
        backup_file: Path to the JSON backup file.
    """
    path = Path(backup_file)
    if not path.exists():
        print(f"Error: Backup file not found: {backup_file}", file=sys.stderr)
        sys.exit(1)

    backup_data = json.loads(path.read_text())
    prompts_to_restore = backup_data.get("prompts", [])

    if not prompts_to_restore:
        print("Backup file contains no prompts.")
        return

    pm = get_prompt_manager()
    from clorag.core.prompt_db import get_prompt_database
    db = get_prompt_database()

    restored = 0
    skipped = 0
    not_found = 0

    for saved in prompts_to_restore:
        key = saved["key"]

        # Only restore if the key still exists in the default registry
        default = get_default_prompt(key)
        if not default:
            print(f"  SKIP {key} (no longer in default registry)")
            not_found += 1
            continue

        # Skip if saved content is identical to the current default
        if saved["content"] == default.content:
            print(f"  SKIP {key} (matches current default)")
            skipped += 1
            continue

        # Find the DB entry to update
        existing = db.get_prompt_by_key(key)
        if existing:
            db.update_prompt(
                prompt_id=existing.id,
                content=saved["content"],
                change_note="Restored from backup",
                updated_by="system",
            )
            restored += 1
            print(f"  RESTORED {key}")
        else:
            # Prompt not in DB yet — create it
            db.create_prompt(
                key=key,
                name=saved.get("name", default.name),
                content=saved["content"],
                category=saved.get("category", default.category),
                description=saved.get("description", default.description),
                model=saved.get("model", default.model),
                variables=saved.get("variables", default.variables),
                created_by="system",
            )
            restored += 1
            print(f"  CREATED {key}")

    # Clear cache after restore
    pm.reload_all()

    print(
        f"\nRestore complete: {restored} restored,"
        f" {skipped} skipped, {not_found} removed from registry"
    )
    print(f"Source: {backup_file}")
    print(f"Backed up at: {backup_data.get('backed_up_at', 'unknown')}")


def list_prompts(category: str | None = None) -> None:
    """List all prompts from database and defaults."""
    pm = get_prompt_manager()
    prompts = pm.list_all_prompts(category=category)

    if not prompts:
        print("No prompts found.")
        return

    # Also get customized list for marking
    customized_keys = {p["key"] for p in _get_customized_prompts()}

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
            custom_mark = " *" if p["key"] in customized_keys else ""
            print(f"  [{source}] {p['key']:40} ({model:6}) {vars_count} vars{custom_mark}")

    if customized_keys:
        print(f"\n  * = customized ({len(customized_keys)} prompt(s) differ from defaults)")


def show_stats() -> None:
    """Show prompt database statistics."""
    pm = get_prompt_manager()

    # Database stats
    from clorag.core.prompt_db import get_prompt_database
    db = get_prompt_database()
    db_stats = db.get_stats()

    # Cache stats
    cache_stats = pm.get_cache_stats()

    # Customization stats
    customized = _get_customized_prompts()

    print("\n=== PROMPT DATABASE STATS ===")
    print(f"Total prompts in DB: {db_stats['total']}")
    print(f"Total versions:      {db_stats['versions_count']}")
    print(f"Customized:          {len(customized)}")

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

    if customized:
        print("\n=== CUSTOMIZED PROMPTS ===")
        for cp in customized:
            print(f"  {cp['key']}")

    print("\n=== CACHE STATS ===")
    print(f"Hits:       {cache_stats['hits']}")
    print(f"Misses:     {cache_stats['misses']}")
    print(f"Hit rate:   {cache_stats['hit_rate']:.1%}")
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"TTL:        {cache_stats['ttl_seconds']}s")


def initialize_prompts(force: bool = False) -> None:
    """Initialize database with default prompts.

    When force=True, automatically backs up customized prompts first.
    """
    pm = get_prompt_manager()

    # Auto-backup before force reset
    if force:
        customized = _get_customized_prompts()
        if customized:
            print("Auto-backing up customized prompts before reset...")
            backup_path = backup_prompts()
            if backup_path:
                print(f"  Use --restore {backup_path} to re-apply later\n")

    print(f"Initializing prompts (force={force})...")
    result = pm.initialize_defaults(force=force)

    print("\nResults:")
    print(f"  Created: {result['created']}")
    print(f"  Updated: {result['updated']}")
    print(f"  Skipped: {result['skipped']}")
    if result.get('removed', 0) > 0:
        print(f"  Removed: {result['removed']}")

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
  uv run init-prompts --force         # Reset all to defaults (auto-backups)
  uv run init-prompts --list          # List all prompts (* = customized)
  uv run init-prompts --category analysis  # List prompts in category
  uv run init-prompts --stats         # Show statistics
  uv run init-prompts --export analysis.thread_analyzer  # Export prompt
  uv run init-prompts --backup        # Backup customized prompts
  uv run init-prompts --backup -o my_prompts.json  # Backup to specific file
  uv run init-prompts --restore data/prompt_backups/prompts_20260326.json
        """,
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reset all prompts to defaults (auto-backups customizations first)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all prompts (* marks customized)",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["base", "agent", "analysis", "synthesis", "drafts", "graph", "scripts"],
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
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Backup all customized prompts to JSON",
    )
    parser.add_argument(
        "--restore", "-r",
        metavar="FILE",
        help="Restore customized prompts from a backup file",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Output path for --backup (default: data/prompt_backups/<timestamp>.json)",
    )

    args = parser.parse_args()

    if args.backup:
        backup_prompts(output_path=args.output)
    elif args.restore:
        restore_prompts(args.restore)
    elif args.list:
        list_prompts(category=args.category)
    elif args.stats:
        show_stats()
    elif args.export:
        export_prompt(args.export)
    else:
        initialize_prompts(force=args.force)


if __name__ == "__main__":
    main()
