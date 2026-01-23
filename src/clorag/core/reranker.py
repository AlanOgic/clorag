"""Voyage AI reranker client for post-retrieval relevance refinement."""

import hashlib
from dataclasses import dataclass

import structlog
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from clorag.config import get_settings
from clorag.core.cache import LRUCache

logger = structlog.get_logger(__name__)

# Rerank cache settings
RERANK_CACHE_MAX_SIZE = 100  # Cache up to 100 unique query+docs combinations


@dataclass
class RerankResult:
    """Result of a reranking operation."""

    index: int  # Original index in the input documents list
    relevance_score: float  # Relevance score from the reranker (0-1)
    text: str  # The document text


@dataclass
class RerankResponse:
    """Full response from a reranking operation."""

    results: list[RerankResult]
    total_tokens: int
    model: str


class RerankCache:
    """Wrapper around LRUCache for rerank results with document-order-independent keys."""

    def __init__(self, max_size: int = RERANK_CACHE_MAX_SIZE) -> None:
        self._cache: LRUCache[list[RerankResult]] = LRUCache(max_size=max_size)

    def _make_key(
        self, query: str, documents: list[str], model: str, top_k: int | None
    ) -> str:
        """Create cache key from query and documents (order-independent)."""
        # Hash each document and sort for order independence
        doc_hashes = sorted(
            hashlib.sha256(d.encode()).hexdigest()[:16] for d in documents
        )
        key_str = f"{query}:{model}:{top_k}:{','.join(doc_hashes)}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self, query: str, documents: list[str], model: str, top_k: int | None
    ) -> list[RerankResult] | None:
        """Get cached reranking results or None if not found."""
        key = self._make_key(query, documents, model, top_k)
        return self._cache.get(key)

    def set(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int | None,
        results: list[RerankResult],
    ) -> None:
        """Cache reranking results."""
        key = self._make_key(query, documents, model, top_k)
        self._cache.set(key, results)

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return self._cache.stats()


# Global rerank cache instance
_rerank_cache: RerankCache | None = None


def get_rerank_cache() -> RerankCache:
    """Get or create rerank cache singleton."""
    global _rerank_cache
    if _rerank_cache is None:
        _rerank_cache = RerankCache()
    return _rerank_cache


class RerankerClient:
    """Client for reranking search results using Voyage AI.

    Rerankers are cross-encoders that jointly process query-document pairs,
    enabling more accurate relevancy prediction than embedding-based similarity.

    Typical usage pattern:
    1. Retrieve top-N candidates with hybrid RRF search (over-fetch, e.g. 20)
    2. Rerank to get the most relevant top-K results (e.g. 5)

    This two-stage approach improves retrieval quality by 15-40% while keeping
    latency acceptable (reranking adds ~100-500ms for typical result sets).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        default_top_k: int | None = None,
    ) -> None:
        """Initialize the reranker client.

        Args:
            api_key: Voyage AI API key. Defaults to VOYAGE_API_KEY env var.
            model: Reranking model name. Defaults to rerank-2.5.
            default_top_k: Default number of top results to return.
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key.get_secret_value()
        self._model = model or settings.voyage_rerank_model
        self._default_top_k = default_top_k or settings.rerank_top_k

        # Initialize Voyage AI client (voyageai lacks proper type stubs)
        self._client = voyageai.Client(api_key=self._api_key)  # type: ignore[attr-defined]

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def default_top_k(self) -> int:
        """Get the default top_k value."""
        return self._default_top_k

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> RerankResponse:
        """Rerank documents by relevance to a query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. Defaults to instance default_top_k.
            use_cache: Whether to use the rerank cache (default True).

        Returns:
            RerankResponse with ordered results and usage info.
        """
        if not documents:
            return RerankResponse(results=[], total_tokens=0, model=self._model)

        effective_top_k = top_k or self._default_top_k

        # Check cache first
        if use_cache:
            cache = get_rerank_cache()
            cached = cache.get(query, documents, self._model, effective_top_k)
            if cached is not None:
                logger.debug(
                    "Rerank cache hit",
                    query_len=len(query),
                    docs=len(documents),
                    top_k=effective_top_k,
                )
                return RerankResponse(
                    results=cached,
                    total_tokens=0,  # No API call, no tokens used
                    model=self._model,
                )

        # Call Voyage AI rerank API
        logger.debug(
            "Calling Voyage rerank API",
            model=self._model,
            query_len=len(query),
            docs=len(documents),
            top_k=effective_top_k,
        )

        result = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=effective_top_k,
            truncation=True,
        )

        # Convert to our dataclass format
        rerank_results = [
            RerankResult(
                index=item.index,
                relevance_score=item.relevance_score,
                text=documents[item.index],
            )
            for item in result.results
        ]

        # Cache the results
        if use_cache:
            cache = get_rerank_cache()
            cache.set(query, documents, self._model, effective_top_k, rerank_results)
            logger.debug(
                "Rerank results cached",
                query_len=len(query),
                docs=len(documents),
            )

        return RerankResponse(
            results=rerank_results,
            total_tokens=result.total_tokens,
            model=self._model,
        )

    def cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics for the reranker."""
        return get_rerank_cache().stats()
