"""CLI script for curated Gmail ingestion."""

import argparse

import anyio
import structlog

from clorag.core.vectorstore import VectorStore
from clorag.ingestion.curated_gmail import run_curated_ingestion
from clorag.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


async def run_ingestion(
    max_threads: int | None,
    offset: int,
    min_confidence: float,
    fresh: bool = False,
) -> int:
    """Run the curated ingestion.

    Args:
        max_threads: Maximum threads to fetch.
        offset: Number of threads to skip.
        min_confidence: Minimum confidence for resolved cases.
        fresh: If True, delete the collection before re-ingesting.

    Returns:
        Number of cases ingested.
    """
    if fresh:
        vectorstore = VectorStore()
        logger.info("Fresh ingestion requested - deleting existing collection")
        try:
            await vectorstore.delete_collection(vectorstore.cases_collection)
            logger.info("Deleted collection", collection=vectorstore.cases_collection)
        except Exception as e:
            logger.warning("Could not delete collection (may not exist)", error=str(e))

    return await run_curated_ingestion(
        max_threads=max_threads,
        offset=offset,
        min_confidence=min_confidence,
    )


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Curated Gmail ingestion with LLM analysis and QC"
    )
    parser.add_argument(
        "--max-threads", "-m",
        type=int,
        help="Maximum number of threads to fetch",
    )
    parser.add_argument(
        "--offset", "-o",
        type=int,
        default=0,
        help="Skip first N threads (for incremental ingestion)",
    )
    parser.add_argument(
        "--min-confidence", "-c",
        type=float,
        default=0.7,
        help="Minimum confidence for resolved classification (default: 0.7)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection before re-ingesting (complete refresh)",
    )

    args = parser.parse_args()

    logger.info(
        "Starting curated Gmail ingestion",
        max_threads=args.max_threads or "all",
        offset=args.offset,
        min_confidence=args.min_confidence,
        fresh=args.fresh,
    )

    try:
        count = anyio.run(
            run_ingestion, args.max_threads, args.offset, args.min_confidence, args.fresh
        )
        logger.info("Curated ingestion completed", cases=count)
    except Exception as e:
        logger.error("Curated ingestion failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
