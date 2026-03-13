"""CLI script for managing Qdrant collection snapshots."""

import argparse
import sys

import anyio
import structlog

from clorag.core.vectorstore import VectorStore
from clorag.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


async def create_snapshot(collection: str) -> None:
    """Create a snapshot of the specified collection."""
    vectorstore = VectorStore()
    logger.info("Creating snapshot", collection=collection)
    name = await vectorstore.create_snapshot(collection)
    logger.info("Snapshot created", collection=collection, snapshot=name)


async def list_snapshots(collection: str) -> None:
    """List all snapshots for the specified collection."""
    vectorstore = VectorStore()
    snapshots = await vectorstore.list_snapshots(collection)
    if not snapshots:
        logger.info("No snapshots found", collection=collection)
        return
    logger.info("Snapshots found", collection=collection, count=len(snapshots))
    for s in snapshots:
        size_mb = s["size"] / (1024 * 1024) if s["size"] else 0
        logger.info(
            "Snapshot",
            name=s["name"],
            created=s["creation_time"],
            size_mb=f"{size_mb:.1f}",
        )


async def recover_snapshot(collection: str, snapshot_name: str) -> None:
    """Recover a collection from a snapshot."""
    vectorstore = VectorStore()
    logger.info(
        "Recovering from snapshot",
        collection=collection,
        snapshot=snapshot_name,
    )
    await vectorstore.recover_snapshot(collection, snapshot_name)
    logger.info("Recovery complete", collection=collection, snapshot=snapshot_name)


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Manage Qdrant collection snapshots (archive/recover)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a snapshot")
    create_parser.add_argument(
        "collection",
        nargs="?",
        default="gmail_cases",
        help="Collection name (default: gmail_cases)",
    )

    # list
    list_parser = subparsers.add_parser("list", help="List existing snapshots")
    list_parser.add_argument(
        "collection",
        nargs="?",
        default="gmail_cases",
        help="Collection name (default: gmail_cases)",
    )

    # recover
    recover_parser = subparsers.add_parser("recover", help="Recover from a snapshot")
    recover_parser.add_argument(
        "collection",
        nargs="?",
        default="gmail_cases",
        help="Collection name (default: gmail_cases)",
    )
    recover_parser.add_argument(
        "snapshot_name",
        help="Name of the snapshot to recover from",
    )

    args = parser.parse_args()

    try:
        if args.command == "create":
            anyio.run(create_snapshot, args.collection)
        elif args.command == "list":
            anyio.run(list_snapshots, args.collection)
        elif args.command == "recover":
            anyio.run(recover_snapshot, args.collection, args.snapshot_name)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error("Operation failed", command=args.command, error=str(e))
        raise


if __name__ == "__main__":
    main()
