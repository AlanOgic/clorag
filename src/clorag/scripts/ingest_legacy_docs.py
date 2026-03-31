"""Script to ingest local Docusaurus markdown sources into the legacy collection.

Reads .md/.mdx files from a local Docusaurus docs/ directory, reconstructs
URLs from file paths, and ingests into docusaurus_docs_legacy — completely
independent from the main CLORAG collections.
"""

import argparse
import re
from pathlib import Path
from uuid import uuid4

import anyio

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.base import Document
from clorag.ingestion.chunker import ContentType, SemanticChunker
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Default path to the local Docusaurus sources
DEFAULT_DOCS_PATH = "cyanview-web/instructions-master/docs"
DEFAULT_SITE_URL = "https://support.cyanview.com"


def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter dict, content without frontmatter).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    fm_block = text[3:end].strip()
    body = text[end + 3:].strip()

    frontmatter: dict[str, str] = {}
    for line in fm_block.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    return frontmatter, body


def _path_to_url(file_path: Path, docs_root: Path, site_url: str) -> str:
    """Convert a file path to its URL on the live site.

    docs/Configuration/REMI.md → https://support.cyanview.com/docs/Configuration/REMI
    """
    relative = file_path.relative_to(docs_root.parent)  # includes "docs/" prefix
    # Remove .md/.mdx extension
    url_path = str(relative).removesuffix(".md").removesuffix(".mdx")
    return f"{site_url.rstrip('/')}/{url_path}"


def _extract_title(frontmatter: dict[str, str], content: str) -> str:
    """Extract title from frontmatter or first heading."""
    if frontmatter.get("title"):
        return frontmatter["title"]

    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return "Untitled"


def load_documents(docs_path: Path, site_url: str) -> list[Document]:
    """Load all markdown files as Documents with reconstructed URLs."""
    documents: list[Document] = []

    md_files = sorted(docs_path.rglob("*.md")) + sorted(docs_path.rglob("*.mdx"))

    for file_path in md_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read file", path=str(file_path), error=str(e))
            continue

        frontmatter, content = _extract_frontmatter(raw_text)

        if len(content) < 100:
            logger.debug("Skipping short file", path=str(file_path), length=len(content))
            continue

        # Keep original text — no product name transforms for legacy
        # so David sees what's actually on the site (e.g. "RIO-Live" not "RIO +LAN")
        title = _extract_title(frontmatter, content)

        url = _path_to_url(file_path, docs_path, site_url)

        # Use frontmatter id for the URL if it differs from filename
        # (some pages use custom slugs via the id field)
        if frontmatter.get("id"):
            # Reconstruct URL using the parent path + frontmatter id
            parent_url = "/".join(url.split("/")[:-1])
            url = f"{parent_url}/{frontmatter['id']}"

        metadata = {
            "source": "docusaurus",
            "url": url,
            "title": title,
            "extractor": "local_markdown",
        }

        if frontmatter.get("description"):
            metadata["description"] = frontmatter["description"]

        documents.append(Document(
            id=str(uuid4()),
            text=content,
            metadata=metadata,
        ))

    logger.info("Loaded local markdown documents", count=len(documents), path=str(docs_path))
    return documents


async def run_legacy_ingestion(
    docs_path: Path,
    site_url: str,
    fresh: bool = False,
) -> int:
    """Ingest local markdown files into the legacy Qdrant collection.

    Args:
        docs_path: Path to the Docusaurus docs/ directory.
        site_url: Base URL of the live site (for URL reconstruction).
        fresh: Delete the collection before re-ingesting.

    Returns:
        Number of chunks ingested.
    """
    settings = get_settings()
    legacy_collection = settings.qdrant_legacy_docs_collection

    vs = VectorStore()

    if fresh:
        logger.info("Fresh ingestion — deleting legacy collection", collection=legacy_collection)
        try:
            await vs.delete_collection(legacy_collection)
        except Exception as e:
            logger.warning("Could not delete legacy collection", error=str(e))

    # Override docs_collection so upsert goes to legacy
    vs._docs_collection = legacy_collection
    await vs._ensure_collection_hybrid(legacy_collection)

    # Load documents from local files
    documents = load_documents(docs_path, site_url)
    if not documents:
        logger.error("No documents found", path=str(docs_path))
        return 0

    # Chunk
    chunker = SemanticChunker.from_settings(ContentType.DOCUMENTATION)
    doc_chunks: list[tuple[Document, list[Document]]] = []

    for doc in documents:
        chunks = chunker.chunk_text(doc.text, content_type=ContentType.DOCUMENTATION)
        chunk_docs = []
        for chunk in chunks:
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": chunk.chunk_index,
                "parent_id": doc.id,
            }
            for key, value in chunk.metadata.items():
                if key not in chunk_metadata:
                    chunk_metadata[key] = value

            chunk_docs.append(Document(
                id=str(uuid4()),
                text=chunk.text,
                metadata=chunk_metadata,
            ))

        if chunk_docs:
            doc_chunks.append((doc, chunk_docs))

    total_chunks = sum(len(cds) for _, cds in doc_chunks)
    logger.info("Chunked documents", documents=len(doc_chunks), total_chunks=total_chunks)

    # Embed with contextualized embeddings
    embeddings = EmbeddingsClient()
    sparse_embeddings = SparseEmbeddingsClient()

    documents_for_embed: list[list[str]] = []
    all_chunk_docs: list[Document] = []

    for parent_doc, cds in doc_chunks:
        documents_for_embed.append([c.text for c in cds])
        all_chunk_docs.extend(cds)

    logger.info("Generating contextualized embeddings", chunks=len(all_chunk_docs))
    all_embeddings = await embeddings.embed_contextualized_batch(
        documents_for_embed, batch_size=10, input_type="document",
    )

    flat_dense: list[list[float]] = []
    for doc_embs in all_embeddings:
        flat_dense.extend(doc_embs)

    # Sparse BM25 embeddings
    texts = [doc.text for doc in all_chunk_docs]
    logger.info("Generating sparse BM25 embeddings", count=len(texts))
    sparse_vectors = sparse_embeddings.embed_batch(texts)

    # Store
    ids = [doc.id for doc in all_chunk_docs]
    metadata = [doc.metadata for doc in all_chunk_docs]

    await vs.upsert_documents_hybrid(
        collection=legacy_collection,
        texts=texts,
        dense_vectors=flat_dense,
        sparse_vectors=sparse_vectors,
        metadata=metadata,
        ids=ids,
    )

    logger.info(
        "Legacy ingestion complete",
        collection=legacy_collection,
        documents=len(doc_chunks),
        chunks=len(all_chunk_docs),
    )
    return len(all_chunk_docs)


def main() -> None:
    """CLI entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Ingest local Docusaurus markdown into the legacy Qdrant collection"
    )
    parser.add_argument(
        "docs_path",
        nargs="?",
        default=DEFAULT_DOCS_PATH,
        help=f"Path to the Docusaurus docs/ directory (default: {DEFAULT_DOCS_PATH})",
    )
    parser.add_argument(
        "--site-url",
        default=DEFAULT_SITE_URL,
        help=f"Base URL of the live site for URL reconstruction (default: {DEFAULT_SITE_URL})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete legacy collection before re-ingesting",
    )
    args = parser.parse_args()

    docs_path = Path(args.docs_path)
    if not docs_path.is_dir():
        logger.error("Docs path does not exist or is not a directory", path=str(docs_path))
        raise SystemExit(1)

    settings = get_settings()
    logger.info(
        "Starting legacy markdown ingestion",
        docs_path=str(docs_path),
        site_url=args.site_url,
        collection=settings.qdrant_legacy_docs_collection,
        fresh=args.fresh,
    )

    try:
        count = anyio.run(run_legacy_ingestion, docs_path, args.site_url, args.fresh)
        logger.info("Done", total_chunks=count)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
