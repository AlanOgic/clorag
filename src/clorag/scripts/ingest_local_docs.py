"""Script to ingest local markdown documentation into Qdrant.

Reads .md/.mdx files from a local directory and ingests them into the
docusaurus_docs collection using the same pipeline as remote ingestion.
"""

import argparse

import anyio

from clorag.core.vectorstore import VectorStore
from clorag.ingestion.local_docs import LocalDocsIngestionPipeline
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def run_ingestion(
    docs_dir: str,
    base_url: str | None = None,
    fresh: bool = False,
    extract_cameras: bool = True,
) -> int:
    """Run the local docs ingestion pipeline.

    Args:
        docs_dir: Path to the local docs directory.
        base_url: Base URL for constructing document URLs.
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

    pipeline = LocalDocsIngestionPipeline(
        docs_dir=docs_dir,
        base_url=base_url,
        vector_store=vectorstore,
        extract_cameras=extract_cameras,
    )
    return await pipeline.run()


def main() -> None:
    """Main entry point for local docs ingestion."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Ingest local markdown documentation into Qdrant (docusaurus_docs collection)"
    )
    parser.add_argument(
        "docs_dir",
        help="Path to the local docs directory (e.g., ../cyanview-support/docs/)",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for document URLs (defaults to DOCUSAURUS_URL env var)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection before re-ingesting (complete refresh)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip camera compatibility extraction",
    )
    args = parser.parse_args()

    logger.info(
        "Starting local docs ingestion",
        docs_dir=args.docs_dir,
        base_url=args.base_url or "from config",
        fresh=args.fresh,
        extract_cameras=not args.no_cameras,
    )

    try:
        count = anyio.run(
            run_ingestion,
            args.docs_dir,
            args.base_url,
            args.fresh,
            not args.no_cameras,
        )
        logger.info("Ingestion completed", documents=count)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
