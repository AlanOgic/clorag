"""Script to populate Neo4j knowledge graph from existing Qdrant chunks."""

import argparse
import sys

import anyio
import structlog

from clorag.core.entity_extractor import EntityExtractor
from clorag.core.graph_store import close_graph_driver, get_graph_store
from clorag.core.vectorstore import VectorStore
from clorag.graph.schema import GraphChunk
from clorag.utils.logger import setup_logging

logger = structlog.get_logger()


async def populate_from_collection(
    collection: str,
    batch_size: int = 50,
    max_chunks: int | None = None,
    concurrency: int = 5,
) -> dict[str, int]:
    """Populate graph from a Qdrant collection.

    Args:
        collection: Qdrant collection name.
        batch_size: Chunks to process per batch.
        max_chunks: Maximum chunks to process (None for all).
        concurrency: Concurrent LLM extractions.

    Returns:
        Dict with entity counts.
    """
    vector_store = VectorStore()
    extractor = EntityExtractor()
    graph_store = await get_graph_store()

    total_counts: dict[str, int] = {
        "cameras": 0,
        "products": 0,
        "protocols": 0,
        "ports": 0,
        "controls": 0,
        "issues": 0,
        "solutions": 0,
        "firmware": 0,
        "relationships": 0,
        "chunks_processed": 0,
    }

    offset = None
    processed = 0

    while True:
        # Scroll through collection
        chunks, next_offset = await vector_store.scroll_chunks(
            collection=collection,
            limit=batch_size,
            offset=offset,
        )

        if not chunks:
            break

        logger.info(
            "processing_batch",
            collection=collection,
            batch_size=len(chunks),
            processed=processed,
        )

        # Prepare chunks for extraction
        chunk_data = []
        for chunk in chunks:
            chunk_id = str(chunk["id"])
            payload = chunk.get("payload", {})
            text = payload.get("text", "")
            title = payload.get("title") or payload.get("subject", "")

            if text and len(text) > 50:
                chunk_data.append((chunk_id, text, title))

                # Create chunk node in graph
                await graph_store.upsert_chunk(GraphChunk(
                    chunk_id=chunk_id,
                    collection=collection,
                    title=title,
                    source_url=payload.get("url"),
                ))

        # Extract entities from batch
        if chunk_data:
            result = await extractor.extract_from_batch(chunk_data, concurrency=concurrency)

            # Ingest to graph
            counts = await graph_store.ingest_extraction_result(result)

            # Accumulate counts
            for key, value in counts.items():
                if key in total_counts:
                    total_counts[key] += value

        processed += len(chunks)
        total_counts["chunks_processed"] = processed

        # Check limit
        if max_chunks and processed >= max_chunks:
            logger.info("max_chunks_reached", limit=max_chunks)
            break

        # Continue to next page
        offset = next_offset
        if offset is None:
            break

    return total_counts


async def run_population(
    collections: list[str],
    batch_size: int,
    max_chunks: int | None,
    concurrency: int,
) -> dict[str, int]:
    """Run graph population for multiple collections.

    Args:
        collections: List of Qdrant collection names.
        batch_size: Chunks per batch.
        max_chunks: Max chunks per collection.
        concurrency: Concurrent extractions.

    Returns:
        Aggregated entity counts.
    """
    total_counts: dict[str, int] = {}

    try:
        for collection in collections:
            logger.info("populating_from_collection", collection=collection)

            counts = await populate_from_collection(
                collection=collection,
                batch_size=batch_size,
                max_chunks=max_chunks,
                concurrency=concurrency,
            )

            # Aggregate counts
            for key, value in counts.items():
                total_counts[key] = total_counts.get(key, 0) + value

            logger.info(
                "collection_complete",
                collection=collection,
                counts=counts,
            )

        # Get final graph stats
        graph_store = await get_graph_store()
        graph_stats = await graph_store.get_stats()
        logger.info("graph_stats", stats=graph_stats)

    finally:
        await close_graph_driver()

    return total_counts


def main() -> None:
    """Main entry point for graph population."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Populate Neo4j knowledge graph from Qdrant chunks"
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["docusaurus_docs", "gmail_cases", "custom_docs"],
        help="Qdrant collections to process (default: all 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Chunks per batch (default: 50)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Max chunks per collection (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent LLM extractions (default: 5)",
    )

    args = parser.parse_args()

    logger.info(
        "starting_graph_population",
        collections=args.collections,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        concurrency=args.concurrency,
    )

    try:
        counts = anyio.run(
            run_population,
            args.collections,
            args.batch_size,
            args.max_chunks,
            args.concurrency,
        )
        logger.info("population_complete", total_counts=counts)
    except Exception as e:
        logger.error("population_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
