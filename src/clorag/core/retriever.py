"""Multi-source retriever combining documentation and Gmail cases with hybrid RRF search."""

from dataclasses import dataclass
from enum import Enum

from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import SearchResult, VectorStore


class SearchSource(Enum):
    """Available search sources."""

    DOCS = "documentation"
    CASES = "gmail_cases"
    CUSTOM = "custom_docs"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    source: SearchSource
    results: list[SearchResult]
    total_found: int


# Technical terms that indicate precision-focused queries
TECHNICAL_TERMS = frozenset({
    "firmware", "protocol", "ip", "rcp", "rio", "ci0", "vp4", "cvp",
    "visca", "lanc", "sdi", "hdmi", "ndi", "srt", "ptz", "iris",
    "shutter", "gain", "white balance", "tally", "gpio", "rs422",
    "rs232", "ethernet", "tcp", "udp", "multicast", "dhcp", "static ip",
})


def calculate_dynamic_threshold(query: str) -> float:
    """Calculate score threshold based on query characteristics.

    Dynamic thresholds improve retrieval quality:
    - Short queries (≤2 words): Lower threshold (0.15) - broader matching
    - Medium queries (3-5 words): Medium threshold (0.20)
    - Long queries (>5 words): Higher threshold (0.25) - more specific
    - Technical terms: +0.05 bonus for precision

    Args:
        query: The search query.

    Returns:
        Calculated score threshold between 0.15 and 0.30.
    """
    words = query.lower().split()
    word_count = len(words)

    # Base threshold based on query length
    if word_count <= 2:
        base = 0.15
    elif word_count <= 5:
        base = 0.20
    else:
        base = 0.25

    # Check for technical terms (boost precision)
    query_lower = query.lower()
    has_technical = any(term in query_lower for term in TECHNICAL_TERMS)
    if has_technical:
        base += 0.05

    return min(base, 0.30)  # Cap at 0.30


class MultiSourceRetriever:
    """Retriever that combines results from documentation, Gmail cases, and custom docs.

    Uses hybrid RRF (Reciprocal Rank Fusion) search combining:
    - Dense vectors (Voyage AI voyage-context-3) for semantic similarity
    - Sparse vectors (BM25) for keyword matching

    This dual-vector approach improves retrieval quality by 15-25% compared
    to dense-only search, especially for technical queries with specific terms.
    """

    def __init__(
        self,
        embeddings_client: EmbeddingsClient | None = None,
        sparse_embeddings_client: SparseEmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embeddings_client: Client for generating dense query embeddings.
            sparse_embeddings_client: Client for generating sparse BM25 vectors.
            vector_store: Client for vector search.
        """
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._sparse_embeddings = sparse_embeddings_client or SparseEmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()

    async def retrieve(
        self,
        query: str,
        source: SearchSource = SearchSource.HYBRID,
        limit: int = 5,
        score_threshold: float | None = None,
        use_dynamic_threshold: bool = True,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query using hybrid RRF search.

        Args:
            query: Search query.
            source: Which source(s) to search.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score (overrides dynamic if set).
            use_dynamic_threshold: Use query-based dynamic thresholds.

        Returns:
            RetrievalResult with matched documents.
        """
        # Generate both dense and sparse embeddings for hybrid search
        dense_vector = await self._embeddings.embed_query(query)
        sparse_vector = self._sparse_embeddings.embed_query(query)

        # Calculate threshold
        if score_threshold is not None:
            threshold = score_threshold
        elif use_dynamic_threshold:
            threshold = calculate_dynamic_threshold(query)
        else:
            threshold = 0.20  # Default fallback

        # Search based on source using hybrid RRF
        if source == SearchSource.DOCS:
            results = await self._vectorstore.search_docs_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
            )
        elif source == SearchSource.CASES:
            results = await self._vectorstore.search_cases_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
            )
        elif source == SearchSource.CUSTOM:
            results = await self._vectorstore.search_custom_docs_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
            )
        else:  # HYBRID - search all collections
            results = await self._vectorstore.hybrid_search_rrf(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
                include_custom_docs=True,
            )

        # Apply threshold filtering (RRF scores are different from cosine similarity)
        # RRF scores are typically lower, so we apply a scaled threshold
        rrf_threshold = threshold * 0.5  # RRF scores are typically 0.0-0.1 range
        filtered_results = [r for r in results if r.score >= rrf_threshold]

        # Ensure minimum results (at least 3 if available)
        if len(filtered_results) < 3 and len(results) >= 3:
            filtered_results = results[:3]

        return RetrievalResult(
            query=query,
            source=source,
            results=filtered_results,
            total_found=len(filtered_results),
        )

    async def retrieve_docs(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve from documentation only using hybrid RRF search.

        Args:
            query: Search query.
            limit: Maximum results.
            score_threshold: Minimum score (uses dynamic if None).

        Returns:
            RetrievalResult from docs collection.
        """
        return await self.retrieve(
            query=query,
            source=SearchSource.DOCS,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def retrieve_cases(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve from Gmail cases only using hybrid RRF search.

        Args:
            query: Search query.
            limit: Maximum results.
            score_threshold: Minimum score (uses dynamic if None).

        Returns:
            RetrievalResult from cases collection.
        """
        return await self.retrieve(
            query=query,
            source=SearchSource.CASES,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def retrieve_custom(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve from custom documents only using hybrid RRF search.

        Args:
            query: Search query.
            limit: Maximum results.
            score_threshold: Minimum score (uses dynamic if None).

        Returns:
            RetrievalResult from custom docs collection.
        """
        return await self.retrieve(
            query=query,
            source=SearchSource.CUSTOM,
            limit=limit,
            score_threshold=score_threshold,
        )

    def format_context(self, result: RetrievalResult, max_chars: int = 8000) -> str:
        """Format retrieval results as context for the agent.

        Args:
            result: Retrieval result to format.
            max_chars: Maximum characters in formatted context.

        Returns:
            Formatted context string.
        """
        if not result.results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, r in enumerate(result.results, 1):
            source_label = r.payload.get("_source", result.source.value)
            url = r.payload.get("url", "")
            title = r.payload.get("title", f"Document {i}")

            header = f"[{i}] {title}"
            if url:
                header += f" ({url})"
            header += f" [Source: {source_label}] [Score: {r.score:.3f}]"

            chunk = f"{header}\n{r.text}\n"

            if current_length + len(chunk) > max_chars:
                break

            context_parts.append(chunk)
            current_length += len(chunk)

        return "\n---\n".join(context_parts)

    def get_cache_stats(self) -> dict[str, dict[str, int]]:
        """Get cache statistics for both embedding types.

        Returns:
            Dict with 'dense' and 'sparse' cache stats.
        """
        from clorag.core.embeddings import get_query_cache

        return {
            "dense": get_query_cache().stats(),
            "sparse": self._sparse_embeddings.cache_stats(),
        }
