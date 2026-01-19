"""Voyage AI embeddings client."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Literal

import structlog
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from clorag.config import get_settings

logger = structlog.get_logger(__name__)

# Query embedding cache settings
QUERY_CACHE_MAX_SIZE = 200  # Cache up to 200 unique queries


class QueryEmbeddingCache:
    """Thread-safe LRU cache for query embeddings to reduce API calls."""

    def __init__(self, max_size: int = QUERY_CACHE_MAX_SIZE) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, model: str, dimensions: int) -> str:
        """Create cache key from query parameters."""
        key_str = f"{query}:{model}:{dimensions}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, query: str, model: str, dimensions: int) -> list[float] | None:
        """Get cached embedding or None if not found."""
        key = self._make_key(query, model, dimensions)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, query: str, model: str, dimensions: int, embedding: list[float]) -> None:
        """Cache an embedding, evicting oldest if at capacity."""
        key = self._make_key(query, model, dimensions)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)  # Remove oldest
                self._cache[key] = embedding

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
            }


# Global query cache instance
_query_cache: QueryEmbeddingCache | None = None


def get_query_cache() -> QueryEmbeddingCache:
    """Get or create query embedding cache singleton."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryEmbeddingCache()
    return _query_cache


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    vectors: list[list[float]]
    total_tokens: int


class EmbeddingsClient:
    """Client for generating embeddings using Voyage AI.

    Uses voyage-context-3 for contextualized document embeddings (with contextualized_embed)
    and voyage-2 for query embeddings (with regular embed).

    voyage-context-3 is optimized for RAG applications - it encodes both chunk content
    and full document context into each embedding, improving retrieval accuracy by ~14%.
    """

    # Model for query embeddings (must match dimensions with context model)
    QUERY_MODEL = "voyage-2"  # 1024 dimensions, compatible with voyage-context-3

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize the embeddings client.

        Args:
            api_key: Voyage AI API key. Defaults to VOYAGE_API_KEY env var.
            model: Embedding model name for contextualized embeddings. Defaults to voyage-context-3.
            dimensions: Output dimensions (256, 512, 1024, 2048). Defaults to 1024.
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key.get_secret_value()
        self._model = model or settings.voyage_model
        self._dimensions = dimensions or settings.voyage_dimensions

        # Initialize Voyage AI async client for non-blocking API calls
        self._client = voyageai.AsyncClient(api_key=self._api_key)

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Get the output dimensions."""
        return self._dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_documents(
        self,
        texts: list[str],
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingResult:
        """Generate embeddings for documents.

        Args:
            texts: List of text chunks to embed.
            input_type: Type of input - 'document' for indexing, 'query' for search.

        Returns:
            EmbeddingResult with vectors and token count.
        """
        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0)

        # voyage-context-3 supports up to 16K chunks per request
        result = await self._client.embed(
            texts=texts,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )

        return EmbeddingResult(
            vectors=result.embeddings,
            total_tokens=result.total_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_query(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding for a search query with LRU caching.

        Uses voyage-context-3 with contextualized_embed() for queries,
        matching the document embedding approach for consistent retrieval.

        Args:
            text: Query text to embed.
            use_cache: Whether to use the query cache (default True).

        Returns:
            Embedding vector for the query.
        """
        # Check cache first
        if use_cache:
            cache = get_query_cache()
            cached = cache.get(text, self._model, self._dimensions)
            if cached is not None:
                logger.debug("Query embedding cache hit", query_len=len(text))
                return cached

        # Use contextualized_embed for queries too - query as [[text]]
        result = await self._client.contextualized_embed(
            inputs=[[text]],
            model=self._model,
            input_type="query",
            output_dimension=self._dimensions,
        )

        # result.results[0].embeddings[0] = the query embedding
        embedding = result.results[0].embeddings[0]

        # Cache the result
        if use_cache:
            cache = get_query_cache()
            cache.set(text, self._model, self._dimensions, embedding)
            logger.debug("Query embedding cached", query_len=len(text))

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 30,
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingResult:
        """Generate embeddings in batches for large datasets.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch (default 30 for ~120K token limit).
            input_type: Type of input for embedding.

        Returns:
            Combined EmbeddingResult with all vectors.
        """
        all_vectors: list[list[float]] = []
        total_tokens = 0
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            logger.info(
                "Processing embedding batch",
                batch=batch_num,
                total=total_batches,
                texts=len(batch),
            )
            result = await self.embed_documents(batch, input_type=input_type)
            all_vectors.extend(result.vectors)
            total_tokens += result.total_tokens

        return EmbeddingResult(vectors=all_vectors, total_tokens=total_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_contextualized(
        self,
        documents: list[list[str]],
        input_type: Literal["document", "query"] = "document",
    ) -> list[list[list[float]]]:
        """Generate contextualized embeddings for documents with multiple chunks.

        Each document is a list of chunks. The embedding model encodes each chunk
        while understanding its context within the full document.

        Args:
            documents: List of documents, where each document is a list of chunks.
            input_type: Type of input - 'document' for indexing, 'query' for search.

        Returns:
            List of embeddings per document, where each document has a list of
            chunk embeddings.
        """
        if not documents:
            return []

        # voyage-context-3 with contextualized_embed
        result = await self._client.contextualized_embed(
            inputs=documents,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )

        # Extract embeddings from results
        # result.results[doc_idx].embeddings[chunk_idx] = vector
        all_embeddings: list[list[list[float]]] = []
        for doc_result in result.results:
            all_embeddings.append(doc_result.embeddings)

        return all_embeddings

    async def embed_contextualized_batch(
        self,
        documents: list[list[str]],
        batch_size: int = 10,
        input_type: Literal["document", "query"] = "document",
    ) -> list[list[list[float]]]:
        """Generate contextualized embeddings in batches.

        Args:
            documents: List of documents, each containing a list of chunks.
            batch_size: Number of documents per batch.
            input_type: Type of input for embedding.

        Returns:
            List of embeddings per document.
        """
        all_embeddings: list[list[list[float]]] = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            batch = documents[i : i + batch_size]
            total_chunks = sum(len(doc) for doc in batch)
            logger.info(
                "Processing contextualized embedding batch",
                batch=batch_num,
                total=total_batches,
                documents=len(batch),
                chunks=total_chunks,
            )
            result = await self.embed_contextualized(batch, input_type=input_type)
            all_embeddings.extend(result)

        return all_embeddings
