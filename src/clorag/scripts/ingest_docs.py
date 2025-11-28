"""Script to ingest Docusaurus documentation into Qdrant."""

import sys

import anyio

from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def run_ingestion(url: str | None = None) -> int:
    """Run the Docusaurus ingestion pipeline.

    Args:
        url: Optional URL override.

    Returns:
        Number of documents ingested.
    """
    pipeline = DocusaurusIngestionPipeline(base_url=url)
    return await pipeline.run()


def main() -> None:
    """Main entry point for docs ingestion."""
    setup_logging()

    url = None
    if len(sys.argv) > 1:
        url = sys.argv[1]

    logger.info("Starting Docusaurus ingestion", url=url or "from config")

    try:
        count = anyio.run(run_ingestion, url)
        logger.info("Ingestion completed", documents=count)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
