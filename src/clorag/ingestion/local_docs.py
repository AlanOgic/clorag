"""Local markdown documentation ingestion pipeline.

Reads .md/.mdx files from a local directory (e.g., Docusaurus docs/) and ingests
them into the docusaurus_docs Qdrant collection using the same chunking, keyword
extraction, RIO fixes, and embedding pipeline as the remote DocusaurusIngestionPipeline.
"""

import re
from pathlib import Path
from uuid import uuid4

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.base import Document
from clorag.ingestion.chunker import ContentType, SemanticChunker
from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline
from clorag.utils.logger import get_logger
from clorag.utils.text_transforms import apply_product_name_transforms

logger = get_logger(__name__)

# Regex to match YAML frontmatter block
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Raw markdown file content.

    Returns:
        Tuple of (frontmatter dict, body without frontmatter).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    fm_block = match.group(1)
    body = content[match.end() :]

    # Simple key: value parser (handles quoted and unquoted values)
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip array lines like [keyword1, keyword2]
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            fm[key] = value

    return fm, body


class LocalDocsIngestionPipeline(DocusaurusIngestionPipeline):
    """Pipeline for ingesting local markdown documentation files.

    Overrides fetch() to read from the filesystem instead of a remote sitemap.
    Reuses process() and ingest() from DocusaurusIngestionPipeline for chunking,
    keyword extraction, RIO terminology fixes, and hybrid embedding.
    """

    def __init__(
        self,
        docs_dir: str | Path,
        base_url: str | None = None,
        embeddings_client: EmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
        extract_cameras: bool = True,
    ) -> None:
        """Initialize the local docs pipeline.

        Args:
            docs_dir: Path to the local docs directory (e.g., cyanview-support/docs/).
            base_url: Base URL for constructing document URLs from slugs.
                      Defaults to DOCUSAURUS_URL env var.
            embeddings_client: Client for generating embeddings.
            vector_store: Client for storing vectors.
            extract_cameras: Whether to extract camera compatibility info.
        """
        # Initialize parent (skips Jina config since we won't use it)
        super().__init__(
            base_url=base_url,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
            extract_cameras=extract_cameras,
            use_jina=False,
        )
        self._docs_dir = Path(docs_dir)

    async def fetch(self) -> list[Document]:
        """Fetch documents from local markdown files.

        Walks the docs directory, reads each .md/.mdx file, parses frontmatter
        for title and slug, strips frontmatter from body, and constructs
        Document objects with URL metadata derived from slug.

        Returns:
            List of documents with markdown content and metadata.
        """
        if not self._docs_dir.is_dir():
            logger.error("Docs directory does not exist", path=str(self._docs_dir))
            return []

        md_files = sorted(
            p
            for p in self._docs_dir.rglob("*")
            if p.suffix in {".md", ".mdx"} and not p.name.startswith("_")
        )

        logger.info(
            "Found markdown files",
            directory=str(self._docs_dir),
            count=len(md_files),
        )

        documents: list[Document] = []
        skipped = 0

        for path in md_files:
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read file", path=str(path), error=str(e))
                skipped += 1
                continue

            fm, body = _parse_frontmatter(raw)

            # Skip very short documents
            if len(body.strip()) < 100:
                logger.debug("Skipping short file", path=str(path))
                skipped += 1
                continue

            # Extract title from frontmatter or first heading
            title = fm.get("title") or self._extract_title_from_markdown(body)

            # Build URL from slug or relative path
            slug = fm.get("slug")
            if slug:
                url = (self._base_url or "").rstrip("/") + slug
            else:
                # Fallback: derive from file path relative to docs_dir
                rel = path.relative_to(self._docs_dir)
                # Remove extension, convert index to parent path
                url_path = str(rel.with_suffix(""))
                if url_path.endswith("/index"):
                    url_path = url_path[: -len("/index")]
                url = (self._base_url or "").rstrip("/") + "/docs/" + url_path

            # Apply product name transformations
            body = apply_product_name_transforms(body)
            if title:
                title = apply_product_name_transforms(title)

            metadata = {
                "source": "docusaurus",
                "url": url,
                "title": title or path.stem,
                "extractor": "local_markdown",
                "local_path": str(path.relative_to(self._docs_dir)),
            }

            # Carry over frontmatter keywords as initial hint
            fm_keywords = fm.get("keywords", "")
            if fm_keywords.startswith("[") and fm_keywords.endswith("]"):
                try:
                    kw_list = [
                        k.strip().strip('"').strip("'")
                        for k in fm_keywords[1:-1].split(",")
                        if k.strip()
                    ]
                    metadata["frontmatter_keywords"] = kw_list
                except Exception:
                    pass

            documents.append(
                Document(
                    id=str(uuid4()),
                    text=body,
                    metadata=metadata,
                )
            )

        logger.info(
            "Loaded local documents",
            loaded=len(documents),
            skipped=skipped,
            total_files=len(md_files),
        )

        return documents
