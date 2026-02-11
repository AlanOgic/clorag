"""Script to ingest Docusaurus documentation into Qdrant."""

import argparse

import anyio

from clorag.core.vectorstore import VectorStore
from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def run_ingestion(
    url: str | None = None, fresh: bool = False, extract_cameras: bool = True
) -> int:
    """Run the Docusaurus ingestion pipeline.

    Args:
        url: Optional URL override.
        fresh: If True, delete the collection before re-ingesting.
        extract_cameras: Whether to extract camera compatibility info.

    Returns:
        Number of documents ingested.
    """
    vectorstore = VectorStore()

    if fresh:
        logger.info("Fresh ingestion requested - deleting existing collection")
        try:
            await vectorstore.delete_collection(vectorstore.docs_collection)
            logger.info("Deleted collection", collection=vectorstore.docs_collection)
        except Exception as e:
            logger.warning("Could not delete collection (may not exist)", error=str(e))

    pipeline = DocusaurusIngestionPipeline(
        base_url=url, vector_store=vectorstore, extract_cameras=extract_cameras
    )
    return await pipeline.run()


def main() -> None:
    """Main entry point for docs ingestion."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Ingest Docusaurus documentation into Qdrant")
    parser.add_argument("url", nargs="?", help="Optional URL override")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection before re-ingesting (complete refresh)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip camera compatibility extraction (faster ingestion)",
    )
    args = parser.parse_args()

    logger.info(
        "Starting Docusaurus ingestion",
        url=args.url or "from config",
        fresh=args.fresh,
        extract_cameras=not args.no_cameras,
    )

    try:
        count = anyio.run(run_ingestion, args.url, args.fresh, not args.no_cameras)
        logger.info("Ingestion completed", documents=count)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
