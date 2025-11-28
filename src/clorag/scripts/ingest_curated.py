"""CLI script for curated Gmail ingestion."""

import argparse

import anyio
import structlog

from clorag.ingestion.curated_gmail import run_curated_ingestion
from clorag.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


async def run_ingestion(
    max_threads: int | None,
    min_confidence: float,
) -> int:
    """Run the curated ingestion."""
    return await run_curated_ingestion(
        max_threads=max_threads,
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
        "--min-confidence", "-c",
        type=float,
        default=0.7,
        help="Minimum confidence for resolved classification (default: 0.7)",
    )

    args = parser.parse_args()

    logger.info(
        "Starting curated Gmail ingestion",
        max_threads=args.max_threads or "all",
        min_confidence=args.min_confidence,
    )

    try:
        count = anyio.run(run_ingestion, args.max_threads, args.min_confidence)
        logger.info("Curated ingestion completed", cases=count)
    except Exception as e:
        logger.error("Curated ingestion failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
