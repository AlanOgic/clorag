"""Script to ingest Gmail threads into Qdrant."""

import argparse

import anyio

from clorag.ingestion.gmail import GmailIngestionPipeline
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def run_ingestion(label: str | None = None, max_threads: int | None = None) -> int:
    """Run the Gmail ingestion pipeline.

    Args:
        label: Optional label override.
        max_threads: Maximum number of threads to fetch.

    Returns:
        Number of documents ingested.
    """
    pipeline = GmailIngestionPipeline(label=label, max_threads=max_threads)
    return await pipeline.run()


def main() -> None:
    """Main entry point for Gmail ingestion."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Ingest Gmail threads into Qdrant")
    parser.add_argument("label", nargs="?", help="Gmail label to filter threads")
    parser.add_argument(
        "--max-threads", "-m", type=int, help="Maximum number of threads to fetch"
    )
    args = parser.parse_args()

    logger.info(
        "Starting Gmail ingestion",
        label=args.label or "from config",
        max_threads=args.max_threads or "unlimited",
    )

    try:
        count = anyio.run(run_ingestion, args.label, args.max_threads)
        logger.info("Ingestion completed", documents=count)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
