#!/usr/bin/env python3
"""Rebuild the FTS5 full-text search index for the camera database."""

from __future__ import annotations

from clorag.core.database import get_camera_database
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Rebuild the FTS5 index for cameras."""
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild camera FTS5 search index")
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Only check if FTS index exists and is populated (no rebuild)",
    )
    args = parser.parse_args()

    db = get_camera_database()

    if args.check:
        # Check FTS index status
        try:
            stats = db.get_stats()
            total_cameras = stats.get("total_cameras", 0)

            # Try a test search
            results = db.search_cameras("test", use_fts=True)
            print(f"FTS index is operational. Total cameras in DB: {total_cameras}")
            print(f"Test search returned {len(results)} results")
        except Exception as e:
            print(f"FTS index check failed: {e}")
            print("Run 'rebuild-fts' without --check to rebuild the index")
    else:
        print("Rebuilding FTS5 search index...")
        count = db.rebuild_fts_index()
        print(f"Successfully indexed {count} cameras")


if __name__ == "__main__":
    main()
