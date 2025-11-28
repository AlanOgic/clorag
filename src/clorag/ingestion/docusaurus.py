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
from clorag.ingestion.chunker import TextChunker
from clorag.models.camera import CameraSource
from clorag.utils.logger import get_logger

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
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
            urls = self._parse_sitemap(sitemap_content)

            logger.info("Found pages in sitemap", count=len(urls))

            # Fetch pages in parallel batches for better performance
            documents: list[Document] = []
            batch_size = 10

            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i : i + batch_size]
                results = await asyncio.gather(
                    *[self._fetch_page(client, url) for url in batch_urls],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Document):
                        documents.append(result)
                    elif isinstance(result, Exception):
                        logger.warning("Failed to fetch page", error=str(result))

        logger.info("Fetched documents", count=len(documents))
        return documents

    def _parse_sitemap(self, content: str) -> list[str]:
        """Parse URLs from sitemap XML.

        Args:
            content: Sitemap XML content.

        Returns:
            List of URLs (filtered to exclude tag pages and search).
        """
        import re

        # Simple parsing - look for <loc> tags
        loc_pattern = re.compile(r"<loc>(.*?)</loc>")
        matches = loc_pattern.findall(content)

        # Filter out non-documentation pages
        excluded_patterns = ["/tags", "/search", "/download"]
        urls = [
            url for url in matches
            if not any(pattern in url for pattern in excluded_patterns)
        ]

        logger.info("Filtered sitemap URLs", total=len(matches), kept=len(urls))
        return urls

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Document | None:
        """Fetch a single page and extract content.

        Args:
            client: HTTP client.
            url: Page URL.

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

        return Document(
            id=str(uuid4()),
            text=text,
            metadata={
                "source": "docusaurus",
                "url": url,
                "title": title,
            },
        )

    def _extract_text_from_html(self, html: str) -> str:
        """Extract main content from Docusaurus HTML, excluding navigation.

        Args:
            html: HTML content.

        Returns:
            Extracted text from main content only.
        """
        import re

        # Try to extract only the main content area (Docusaurus specific)
        # Priority: <article>, then <main>, then fall back to full page
        content = html

        # Try article tag first (most specific for doc content)
        article_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
        if article_match:
            content = article_match.group(1)
        else:
            # Try main tag
            main_match = re.search(r"<main[^>]*>(.*?)</main>", html, re.DOTALL)
            if main_match:
                content = main_match.group(1)

        # Remove navigation, sidebar, footer, header elements
        content = re.sub(r"<nav[^>]*>.*?</nav>", "", content, flags=re.DOTALL)
        content = re.sub(r"<aside[^>]*>.*?</aside>", "", content, flags=re.DOTALL)
        content = re.sub(r"<header[^>]*>.*?</header>", "", content, flags=re.DOTALL)
        content = re.sub(r"<footer[^>]*>.*?</footer>", "", content, flags=re.DOTALL)
        content = re.sub(r'<div[^>]*class="[^"]*sidebar[^"]*"[^>]*>.*?</div>', "", content, flags=re.DOTALL)
        content = re.sub(r'<div[^>]*class="[^"]*menu[^"]*"[^>]*>.*?</div>', "", content, flags=re.DOTALL)
        content = re.sub(r'<div[^>]*class="[^"]*toc[^"]*"[^>]*>.*?</div>', "", content, flags=re.DOTALL)

        # Remove script and style tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", content)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _extract_title_from_html(self, html: str) -> str:
        """Extract title from HTML using cascading fallback strategy.

        Order of preference:
        1. <h1> tag - most semantic for documentation
        2. <meta property="og:title"> - OpenGraph title
        3. <title> tag (cleaned of site suffix)
        4. Fallback: "Untitled"

        Args:
            html: HTML content.

        Returns:
            Page title.
        """
        import re

        # 1. Try h1 tag (most semantic for documentation pages)
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if h1_match:
            title = h1_match.group(1).strip()
            if title:
                return title

        # 2. Try og:title meta tag
        og_match = re.search(
            r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if og_match:
            title = og_match.group(1).strip()
            if title:
                return title

        # Also check for content first, then property (different order)
        og_match2 = re.search(
            r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:title["\']',
            html,
            re.IGNORECASE,
        )
        if og_match2:
            title = og_match2.group(1).strip()
            if title:
                return title

        # 3. Clean title tag (remove site suffix like "| Cyanview Support")
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if title_match:
            raw_title = title_match.group(1).strip()
            # Split on common separators and take first part
            cleaned = re.split(r"\s*[|\u2013\u2014-]\s*", raw_title)[0].strip()
            if cleaned:
                return cleaned

        # 4. Fallback
        return "Untitled"

    async def process(self, documents: list[Document]) -> list[tuple[Document, list[Document]]]:
        """Chunk documents for contextualized embedding.

        Args:
            documents: Raw documents.

        Returns:
            List of (parent_doc, chunk_docs) tuples for contextualized embedding.
        """
        processed: list[tuple[Document, list[Document]]] = []

        for doc in documents:
            chunks = self._chunker.chunk_text(doc.text)
            chunk_docs = []

            for chunk in chunks:
                chunk_docs.append(
                    Document(
                        id=str(uuid4()),  # Generate unique UUID for each chunk
                        text=chunk.text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk.chunk_index,
                            "parent_id": doc.id,
                        },
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
