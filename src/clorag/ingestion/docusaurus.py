"""Docusaurus documentation ingestion pipeline."""

import asyncio
from urllib.parse import urljoin
from uuid import uuid4

import httpx

from clorag.analysis.camera_extractor import CameraExtractor
from clorag.config import get_settings
from clorag.core.database import get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.base import BaseIngestionPipeline, Document
from clorag.ingestion.chunker import ContentType, SemanticChunker
from clorag.models.camera import CameraSource
from clorag.utils.logger import get_logger
from clorag.utils.text_transforms import apply_product_name_transforms

logger = get_logger(__name__)


class DocusaurusIngestionPipeline(BaseIngestionPipeline):
    """Pipeline for ingesting Docusaurus documentation.

    Uses the sitemap.xml to discover pages and Crawl4AI for scraping.
    """

    def __init__(
        self,
        base_url: str | None = None,
        embeddings_client: EmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        extract_cameras: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            base_url: Docusaurus site URL. Defaults to DOCUSAURUS_URL env var.
            embeddings_client: Client for generating embeddings.
            vector_store: Client for storing vectors.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            extract_cameras: Whether to extract camera compatibility info.
        """
        settings = get_settings()
        self._base_url = base_url or settings.docusaurus_url
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._sparse_embeddings = SparseEmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()
        # Use SemanticChunker for code block and heading preservation
        self._chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            adaptive_threshold=800,  # Short pages stay as single chunk
            preserve_code_blocks=True,
            preserve_tables=True,
            respect_headings=True,
        )
        self._extract_cameras = extract_cameras

    async def fetch(self) -> list[Document]:
        """Fetch pages from Docusaurus using sitemap.xml with parallel requests.

        Returns:
            List of documents with page content.
        """
        if not self._base_url:
            logger.error("No Docusaurus URL configured")
            return []

        # Fetch sitemap
        sitemap_url = urljoin(self._base_url, "/sitemap.xml")
        logger.info("Fetching sitemap", url=sitemap_url)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(sitemap_url, timeout=30.0)
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to fetch sitemap", error=str(e))
                return []

            # Parse sitemap (simple XML parsing)
            sitemap_content = response.text
            url_entries = self._parse_sitemap(sitemap_content)

            logger.info("Found pages in sitemap", count=len(url_entries))

            # Fetch pages in parallel batches for better performance
            documents: list[Document] = []
            batch_size = 10

            for i in range(0, len(url_entries), batch_size):
                batch_entries = url_entries[i : i + batch_size]
                results = await asyncio.gather(
                    *[self._fetch_page(client, url, lastmod) for url, lastmod in batch_entries],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Document):
                        documents.append(result)
                    elif isinstance(result, Exception):
                        logger.warning("Failed to fetch page", error=str(result))

        logger.info("Fetched documents", count=len(documents))
        return documents

    def _parse_sitemap(self, content: str) -> list[tuple[str, str | None]]:
        """Parse URLs and lastmod dates from sitemap XML.

        Args:
            content: Sitemap XML content.

        Returns:
            List of tuples (url, lastmod) filtered to exclude tag pages and search.
            lastmod is ISO date string or None if not available.
        """
        import re

        # Parse <url> blocks to extract both <loc> and <lastmod>
        url_pattern = re.compile(r"<url>(.*?)</url>", re.DOTALL)
        loc_pattern = re.compile(r"<loc>(.*?)</loc>")
        lastmod_pattern = re.compile(r"<lastmod>(.*?)</lastmod>")

        results: list[tuple[str, str | None]] = []
        excluded_patterns = ["/tags", "/search", "/download"]

        for url_block in url_pattern.findall(content):
            loc_match = loc_pattern.search(url_block)
            if not loc_match:
                continue

            url = loc_match.group(1)

            # Skip excluded patterns
            if any(pattern in url for pattern in excluded_patterns):
                continue

            # Extract lastmod if available
            lastmod_match = lastmod_pattern.search(url_block)
            lastmod = lastmod_match.group(1) if lastmod_match else None

            results.append((url, lastmod))

        with_lastmod = sum(1 for _, lm in results if lm)
        logger.info("Filtered sitemap URLs", total=len(results), with_lastmod=with_lastmod)
        return results

    async def _fetch_page(
        self, client: httpx.AsyncClient, url: str, lastmod: str | None = None
    ) -> Document | None:
        """Fetch a single page and extract content.

        Args:
            client: HTTP client.
            url: Page URL.
            lastmod: Last modification date from sitemap (ISO format).

        Returns:
            Document with page content, or None if failed.
        """
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        # Extract text content (basic HTML parsing)
        # In production, use Crawl4AI for better extraction
        html = response.text
        text = self._extract_text_from_html(html)
        title = self._extract_title_from_html(html)

        if not text or len(text) < 100:  # Skip very short pages
            return None

        # Apply product name transformations (RIO-Live -> RIO +LAN, RIO -> RIO +WAN)
        text = self._apply_text_transformations(text)
        title = self._apply_text_transformations(title)

        metadata = {
            "source": "docusaurus",
            "url": url,
            "title": title,
        }
        if lastmod:
            metadata["lastmod"] = lastmod

        return Document(
            id=str(uuid4()),
            text=text,
            metadata=metadata,
        )

    def _extract_text_from_html(self, html: str) -> str:
        """Extract main content from Docusaurus HTML, excluding navigation."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "aside", "header", "footer"]):
            element.decompose()

        # Remove sidebar and menu elements by class
        for element in soup.find_all(
            class_=lambda c: c and any(x in c for x in ["sidebar", "menu", "toc"])
        ):
            element.decompose()

        # Try to find main content area
        content = soup.find("article") or soup.find("main") or soup.body or soup

        # Get text with proper spacing
        text = content.get_text(separator=" ", strip=True)

        return text

    def _extract_title_from_html(self, html: str) -> str:
        """Extract title from HTML using cascading fallback strategy."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # 1. Try h1 tag
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        # 2. Try og:title meta tag
        og_title = soup.find("meta", property="og:title")
        if og_title:
            content = og_title.get("content")
            if content and isinstance(content, str):
                return content.strip()

        # 3. Clean title tag
        title = soup.find("title")
        if title:
            raw_title = title.get_text(strip=True)
            # Split on common separators and take first part
            import re
            cleaned = re.split(r"\s*[|\u2013\u2014-]\s*", raw_title)[0].strip()
            if cleaned:
                return cleaned

        return "Untitled"

    def _apply_text_transformations(self, text: str) -> str:
        """Apply product name transformations to text.

        Transforms:
        - RIO-Live / RIO Live / RIOLive -> RIO +LAN
        - RIO (standalone) -> RIO +WAN

        Args:
            text: Input text to transform.

        Returns:
            Transformed text with product name replacements.
        """
        return apply_product_name_transforms(text)

    async def process(self, documents: list[Document]) -> list[tuple[Document, list[Document]]]:
        """Chunk documents for contextualized embedding.

        Args:
            documents: Raw documents.

        Returns:
            List of (parent_doc, chunk_docs) tuples for contextualized embedding.
        """
        processed: list[tuple[Document, list[Document]]] = []

        for doc in documents:
            # Use semantic chunking with documentation content type
            chunks = self._chunker.chunk_text(doc.text, content_type=ContentType.DOCUMENTATION)
            chunk_docs = []

            for chunk in chunks:
                # Merge parent metadata with chunk-specific semantic metadata
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": chunk.chunk_index,
                    "parent_id": doc.id,
                }
                # Include semantic metadata (section, heading_level, has_code, etc.)
                for key, value in chunk.metadata.items():
                    if key not in chunk_metadata:
                        chunk_metadata[key] = value

                chunk_docs.append(
                    Document(
                        id=str(uuid4()),  # Generate unique UUID for each chunk
                        text=chunk.text,
                        metadata=chunk_metadata,
                    )
                )

            if chunk_docs:
                processed.append((doc, chunk_docs))

        total_chunks = sum(len(chunks) for _, chunks in processed)
        logger.info("Processed documents into chunks", original=len(documents), chunks=total_chunks)
        return processed

    async def ingest(self, doc_chunks: list[tuple[Document, list[Document]]]) -> int:
        """Embed and store documents using contextualized embeddings and hybrid search.

        Uses voyage-context-3 for dense vectors and BM25 for sparse vectors.

        Args:
            doc_chunks: List of (parent_doc, chunk_docs) tuples.

        Returns:
            Number of chunks ingested.
        """
        if not doc_chunks:
            return 0

        # Ensure collection exists with hybrid vector support
        await self._vectorstore.ensure_collections(hybrid=True)

        # Prepare documents for contextualized embedding
        # Each document is a list of chunk texts
        documents_for_embed: list[list[str]] = []
        all_chunk_docs: list[Document] = []

        for parent_doc, chunk_docs in doc_chunks:
            chunk_texts = [c.text for c in chunk_docs]
            documents_for_embed.append(chunk_texts)
            all_chunk_docs.extend(chunk_docs)

        total_chunks = sum(len(doc) for doc in documents_for_embed)
        logger.info(
            "Generating contextualized embeddings",
            documents=len(documents_for_embed),
            total_chunks=total_chunks,
        )

        # Generate contextualized dense embeddings in batches
        all_embeddings = await self._embeddings.embed_contextualized_batch(
            documents_for_embed,
            batch_size=10,
            input_type="document",
        )

        # Flatten dense embeddings to match chunks
        flat_dense_vectors: list[list[float]] = []
        for doc_embeddings in all_embeddings:
            flat_dense_vectors.extend(doc_embeddings)

        logger.info("Generated dense embeddings", count=len(flat_dense_vectors))

        # Generate sparse BM25 embeddings for all chunks
        texts = [doc.text for doc in all_chunk_docs]
        logger.info("Generating sparse BM25 embeddings", count=len(texts))
        sparse_vectors = self._sparse_embeddings.embed_batch(texts)
        logger.info("Generated sparse embeddings", count=len(sparse_vectors))

        # Store in Qdrant with hybrid vectors
        ids = [doc.id for doc in all_chunk_docs]
        metadata = [doc.metadata for doc in all_chunk_docs]

        await self._vectorstore.upsert_documents_hybrid(
            collection=self._vectorstore.docs_collection,
            texts=texts,
            dense_vectors=flat_dense_vectors,
            sparse_vectors=sparse_vectors,
            metadata=metadata,
            ids=ids,
        )

        logger.info("Ingested documents into Qdrant with hybrid vectors", count=len(all_chunk_docs))

        # Extract camera compatibility information from documentation
        if self._extract_cameras:
            await self._extract_cameras_from_docs(doc_chunks)

        return len(all_chunk_docs)

    async def _extract_cameras_from_docs(
        self, doc_chunks: list[tuple[Document, list[Document]]]
    ) -> int:
        """Extract and store camera compatibility info from documentation.

        Args:
            doc_chunks: List of (parent_doc, chunk_docs) tuples.

        Returns:
            Number of cameras extracted/updated.
        """
        extractor = CameraExtractor()
        db = get_camera_database()

        # Prepare contents for batch extraction (use parent docs for full context)
        contents: list[tuple[str, str | None]] = []
        for parent_doc, _ in doc_chunks:
            url = parent_doc.metadata.get("url")
            contents.append((parent_doc.text, url))

        logger.info("Extracting cameras from documentation", pages=len(contents))

        try:
            cameras = await extractor.extract_from_batch(contents, concurrency=5)

            # Upsert cameras into database
            for camera in cameras:
                db.upsert_camera(camera, CameraSource.DOCUMENTATION)

            logger.info(
                "Camera extraction complete",
                extracted=len(cameras),
            )
            return len(cameras)

        except Exception as e:
            logger.error("Camera extraction failed", error=str(e))
            return 0
        finally:
            await extractor.close()
