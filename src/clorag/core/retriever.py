"""Multi-source retriever combining documentation and Gmail cases with hybrid RRF search."""

import asyncio
from dataclasses import dataclass
from enum import Enum

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.reranker import RerankerClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import SearchResult, VectorStore
from clorag.utils.text_transforms import apply_product_name_transforms


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
    reranked: bool = False


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
    - Optional reranking with Voyage rerank-2.5 for refined relevance

    This multi-stage approach improves retrieval quality by 25-40% compared
    to dense-only search, especially for technical queries with specific terms.
    """

    def __init__(
        self,
        embeddings_client: EmbeddingsClient | None = None,
        sparse_embeddings_client: SparseEmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
        reranker_client: RerankerClient | None = None,
        rerank_enabled: bool | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embeddings_client: Client for generating dense query embeddings.
            sparse_embeddings_client: Client for generating sparse BM25 vectors.
            vector_store: Client for vector search.
            reranker_client: Client for reranking results (optional).
            rerank_enabled: Override config setting for reranking. Defaults to config value.
        """
        settings = get_settings()
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._sparse_embeddings = sparse_embeddings_client or SparseEmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()
        self._reranker = reranker_client or RerankerClient()
        self._rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.rerank_enabled
        )

    async def retrieve(
        self,
        query: str,
        source: SearchSource = SearchSource.HYBRID,
        limit: int = 5,
        score_threshold: float | None = None,
        use_dynamic_threshold: bool = True,
        use_reranking: bool | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query using hybrid RRF search + reranking.

        Args:
            query: Search query.
            source: Which source(s) to search.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score (overrides dynamic if set).
            use_dynamic_threshold: Use query-based dynamic thresholds.
            use_reranking: Override instance reranking setting. Defaults to instance setting.

        Returns:
            RetrievalResult with matched documents.
        """
        # Normalize old RIO terms in query so "RIO-Live" matches "RIO +LAN" content
        query = apply_product_name_transforms(query)

        # Determine if reranking is enabled for this query
        should_rerank = use_reranking if use_reranking is not None else self._rerank_enabled

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

        # Over-fetch when reranking is enabled (3x the limit, min 15)
        fetch_limit = max(limit * 3, 15) if should_rerank else limit

        # Search based on source using hybrid RRF
        if source == SearchSource.DOCS:
            results = await self._vectorstore.search_docs_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=fetch_limit,
            )
        elif source == SearchSource.CASES:
            results = await self._vectorstore.search_cases_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=fetch_limit,
            )
        elif source == SearchSource.CUSTOM:
            results = await self._vectorstore.search_custom_docs_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=fetch_limit,
            )
        else:  # HYBRID - search all collections
            results = await self._vectorstore.hybrid_search_rrf(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=fetch_limit,
                include_custom_docs=True,
            )

        # Apply reranking if enabled and we have results
        # NOTE: No pre-rerank threshold filtering — RRF scores are not calibrated
        # (they can range well above 1.0), so we let the reranker decide relevance.
        # Threshold filtering happens post-rerank using calibrated reranker scores.
        was_reranked = False
        if should_rerank and results:
            results = await self._apply_reranking(query, results, limit)
            was_reranked = True

        # Apply threshold filtering post-rerank (reranker scores are calibrated 0-1)
        if was_reranked:
            filtered_results = [r for r in results if r.score >= threshold]
            # Ensure minimum results (at least 3 if available)
            if len(filtered_results) < 3 and len(results) >= 3:
                filtered_results = results[:3]
        else:
            # Without reranking, skip threshold on RRF scores (uncalibrated)
            # Just return all results up to limit
            filtered_results = results

        # Limit results to requested amount
        final_results = filtered_results[:limit]

        return RetrievalResult(
            query=query,
            source=source,
            results=final_results,
            total_found=len(final_results),
            reranked=was_reranked,
        )

    async def _apply_reranking(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Apply reranking to search results using Voyage rerank model.

        Args:
            query: The search query.
            results: List of SearchResult objects to rerank.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of SearchResult objects with updated scores.
        """
        if not results:
            return results

        # Extract document texts for reranking
        documents = [r.text for r in results]

        # Run reranker in thread pool to avoid blocking the event loop
        rerank_response = await asyncio.to_thread(
            self._reranker.rerank,
            query=query,
            documents=documents,
            top_k=top_k,
        )

        # Map reranked results back to SearchResult objects
        reranked_results: list[SearchResult] = []
        for rerank_result in rerank_response.results:
            original_result = results[rerank_result.index]
            # Create new SearchResult with updated score from reranker
            reranked_results.append(
                SearchResult(
                    id=original_result.id,
                    score=rerank_result.relevance_score,  # Use reranker score
                    payload=original_result.payload,
                    text=original_result.text,
                )
            )

        return reranked_results

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

    def get_cache_stats(self) -> dict[str, dict[str, int | float]]:
        """Get cache statistics for embeddings and reranker.

        Returns:
            Dict with 'dense', 'sparse', and 'rerank' cache stats.
        """
        from clorag.core.embeddings import get_query_cache

        return {
            "dense": get_query_cache().stats(),
            "sparse": self._sparse_embeddings.cache_stats(),
            "rerank": self._reranker.cache_stats(),
        }

    @property
    def rerank_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self._rerank_enabled
