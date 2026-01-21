"""Docusaurus documentation ingestion pipeline."""

import asyncio
from urllib.parse import urljoin
from uuid import uuid4

import httpx

from clorag.analysis.camera_extractor import CameraExtractor
from clorag.analysis.rio_analyzer import RIOTerminologyAnalyzer, apply_rio_fixes_before_embedding
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

    Uses the sitemap.xml to discover pages and Jina Reader for content extraction.
    Falls back to BeautifulSoup if Jina is disabled or fails.
    """

    # Jina Reader base URL
    JINA_READER_URL = "https://r.jina.ai/"

    def __init__(
        self,
        base_url: str | None = None,
        embeddings_client: EmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
        extract_cameras: bool = True,
        use_jina: bool | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            base_url: Docusaurus site URL. Defaults to DOCUSAURUS_URL env var.
            embeddings_client: Client for generating embeddings.
            vector_store: Client for storing vectors.
            extract_cameras: Whether to extract camera compatibility info.
            use_jina: Use Jina Reader API for extraction. Defaults to USE_JINA_READER env var.
        """
        settings = get_settings()
        self._base_url = base_url or settings.docusaurus_url
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._sparse_embeddings = SparseEmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()
        # Use SemanticChunker with settings-based token-aware chunking
        self._chunker = SemanticChunker.from_settings(ContentType.DOCUMENTATION)
        self._extract_cameras = extract_cameras

        # Jina Reader configuration
        self._use_jina = use_jina if use_jina is not None else settings.use_jina_reader
        self._jina_api_key = (
            settings.jina_api_key.get_secret_value() if settings.jina_api_key else None
        )
        self._jina_success_count = 0
        self._jina_fallback_count = 0

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

            # Fetch pages in parallel batches
            # Jina free tier: ~20 requests/min, so use smaller batches with delays
            documents: list[Document] = []
            batch_size = 5 if self._use_jina else 10
            batch_delay = 3.0 if self._use_jina and not self._jina_api_key else 0.0

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

                # Rate limit delay for Jina free tier
                if batch_delay > 0 and i + batch_size < len(url_entries):
                    await asyncio.sleep(batch_delay)

        # Log Jina Reader stats
        if self._use_jina:
            logger.info(
                "Jina Reader stats",
                success=self._jina_success_count,
                fallback=self._jina_fallback_count,
                total=len(documents),
            )

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

        Uses Jina Reader API if enabled, falls back to BeautifulSoup.

        Args:
            client: HTTP client.
            url: Page URL.
            lastmod: Last modification date from sitemap (ISO format).

        Returns:
            Document with page content, or None if failed.
        """
        text: str | None = None
        title: str | None = None
        used_jina = False

        # Try Jina Reader first if enabled
        if self._use_jina:
            jina_result = await self._fetch_with_jina(client, url)
            if jina_result:
                text, title = jina_result
                used_jina = True
                self._jina_success_count += 1

        # Fallback to BeautifulSoup
        if not text:
            if self._use_jina:
                self._jina_fallback_count += 1
            bs_result = await self._fetch_with_beautifulsoup(client, url)
            if bs_result:
                text, title = bs_result

        if not text or len(text) < 100:  # Skip very short pages
            return None

        # Apply product name transformations (RIO-Live -> RIO +LAN, RIO -> RIO +WAN)
        text = self._apply_text_transformations(text)
        if title:
            title = self._apply_text_transformations(title)

        metadata = {
            "source": "docusaurus",
            "url": url,
            "title": title or "Untitled",
            "extractor": "jina" if used_jina else "beautifulsoup",
        }
        if lastmod:
            metadata["lastmod"] = lastmod

        return Document(
            id=str(uuid4()),
            text=text,
            metadata=metadata,
        )

    async def _fetch_with_jina(
        self, client: httpx.AsyncClient, url: str, max_retries: int = 2
    ) -> tuple[str, str | None] | None:
        """Fetch page content using Jina Reader API with retry on rate limit.

        Args:
            client: HTTP client.
            url: Page URL to fetch.
            max_retries: Max retries on 429 rate limit errors.

        Returns:
            Tuple of (text, title) or None if failed.
        """
        jina_url = f"{self.JINA_READER_URL}{url}"

        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "markdown",
            "X-No-Cache": "true",  # Fresh content for ingestion
        }

        # Add API key if available (for higher rate limits)
        if self._jina_api_key:
            headers["Authorization"] = f"Bearer {self._jina_api_key}"

        for attempt in range(max_retries + 1):
            try:
                response = await client.get(jina_url, headers=headers, timeout=60.0)

                # Retry on rate limit with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = 2 ** (attempt + 1)  # 2s, 4s
                        logger.debug("Jina rate limited, retrying", url=url, delay=delay)
                        await asyncio.sleep(delay)
                        continue
                    return None

                response.raise_for_status()

                content = response.text.strip()
                if not content or len(content) < 50:
                    return None

                # Extract title from Jina markdown (first # heading)
                title = self._extract_title_from_markdown(content)

                # Clean up Jina output (remove URL line at start if present)
                lines = content.split("\n")
                if lines and lines[0].startswith("URL:"):
                    content = "\n".join(lines[1:]).strip()

                return (content, title)

            except httpx.HTTPError as e:
                logger.debug("Jina Reader failed", url=url, error=str(e))
                return None
            except Exception as e:
                logger.debug("Jina Reader error", url=url, error=str(e))
                return None

        return None

    async def _fetch_with_beautifulsoup(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str | None] | None:
        """Fetch page content using BeautifulSoup (fallback method).

        Args:
            client: HTTP client.
            url: Page URL to fetch.

        Returns:
            Tuple of (text, title) or None if failed.
        """
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        html = response.text
        text = self._extract_text_from_html(html)
        title = self._extract_title_from_html(html)

        if not text:
            return None

        return (text, title)

    def _extract_title_from_markdown(self, markdown: str) -> str | None:
        """Extract title from markdown content (first # heading).

        Args:
            markdown: Markdown content.

        Returns:
            Title string or None.
        """
        import re

        # Look for first H1 heading
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Try H2 as fallback
        match = re.search(r"^##\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract main content from Docusaurus HTML, excluding navigation.

        Converts HTML tables to markdown format to preserve structure.
        """
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

        # Convert tables to markdown before extracting text
        if content:
            self._convert_tables_to_markdown(content)

        # Get text with proper spacing
        text = content.get_text(separator=" ", strip=True) if content else ""

        return text

    def _convert_tables_to_markdown(self, element: "Tag") -> None:
        """Convert HTML tables to markdown format in-place.

        Replaces <table> elements with their markdown representation
        so that get_text() preserves the table structure.

        Args:
            element: BeautifulSoup element containing tables.
        """
        from bs4 import NavigableString

        tables = element.find_all("table")

        for table in tables:
            markdown_lines: list[str] = []

            # Extract header row
            thead = table.find("thead")
            header_cells: list[str] = []

            if thead:
                header_row = thead.find("tr")
                if header_row:
                    for th in header_row.find_all(["th", "td"]):
                        cell_text = th.get_text(strip=True)
                        header_cells.append(cell_text)
            else:
                # Try first row as header
                first_row = table.find("tr")
                if first_row:
                    for cell in first_row.find_all(["th", "td"]):
                        cell_text = cell.get_text(strip=True)
                        header_cells.append(cell_text)

            if header_cells:
                # Add header row
                markdown_lines.append("| " + " | ".join(header_cells) + " |")
                # Add separator row
                markdown_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

            # Extract body rows
            tbody = table.find("tbody")
            body_rows = tbody.find_all("tr") if tbody else []

            # If no tbody, get all rows except first (used as header)
            if not body_rows:
                all_rows = table.find_all("tr")
                body_rows = all_rows[1:] if header_cells and all_rows else all_rows

            for row in body_rows:
                cells = row.find_all(["td", "th"])
                row_data: list[str] = []
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    # Escape pipe characters in cell content
                    cell_text = cell_text.replace("|", "\\|")
                    row_data.append(cell_text)

                # Pad row to match header length
                while len(row_data) < len(header_cells):
                    row_data.append("")

                if row_data:
                    markdown_lines.append("| " + " | ".join(row_data) + " |")

            # Replace table with markdown text
            if markdown_lines:
                markdown_text = "\n" + "\n".join(markdown_lines) + "\n"
                table.replace_with(NavigableString(markdown_text))

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

        Optionally applies RIO terminology fixes before chunking based on settings.

        Args:
            documents: Raw documents.

        Returns:
            List of (parent_doc, chunk_docs) tuples for contextualized embedding.
        """
        settings = get_settings()
        processed: list[tuple[Document, list[Document]]] = []

        # Apply RIO terminology fixes before chunking if enabled
        if settings.rio_fix_on_ingest:
            documents = await self._apply_rio_fixes(
                documents, settings.rio_fix_min_confidence
            )

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

    async def _apply_rio_fixes(
        self, documents: list[Document], min_confidence: float
    ) -> list[Document]:
        """Apply RIO terminology fixes to documents before chunking.

        Args:
            documents: List of documents to process.
            min_confidence: Minimum confidence threshold for auto-applying fixes.

        Returns:
            List of documents with RIO fixes applied.
        """
        analyzer = RIOTerminologyAnalyzer()
        fixed_documents: list[Document] = []
        total_fixes = 0

        for doc in documents:
            fixed_text, applied_fixes = await apply_rio_fixes_before_embedding(
                doc.text, analyzer, min_confidence
            )

            if applied_fixes:
                total_fixes += len(applied_fixes)
                # Create new document with fixed text
                fixed_documents.append(
                    Document(
                        id=doc.id,
                        text=fixed_text,
                        metadata={
                            **doc.metadata,
                            "rio_fixes_applied": len(applied_fixes),
                        },
                    )
                )
            else:
                fixed_documents.append(doc)

        if total_fixes > 0:
            logger.info(
                "Applied RIO terminology fixes during ingestion",
                documents_fixed=sum(
                    1 for d in fixed_documents if d.metadata.get("rio_fixes_applied", 0) > 0
                ),
                total_fixes=total_fixes,
            )

        return fixed_documents

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
