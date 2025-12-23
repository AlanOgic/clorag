"""BM25 sparse embeddings using FastEmbed for hybrid search."""

import hashlib
from collections import OrderedDict
from threading import Lock

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector

from clorag.utils.logger import get_logger

logger = get_logger(__name__)

# Sparse query cache settings
SPARSE_CACHE_MAX_SIZE = 200


class SparseQueryCache:
    """Thread-safe LRU cache for sparse query embeddings."""

    def __init__(self, max_size: int = SPARSE_CACHE_MAX_SIZE) -> None:
        self._cache: OrderedDict[str, SparseVector] = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, model: str) -> str:
        """Create cache key from query parameters."""
        key_str = f"{query}:{model}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, query: str, model: str) -> SparseVector | None:
        """Get cached sparse vector or None if not found."""
        key = self._make_key(query, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, query: str, model: str, vector: SparseVector) -> None:
        """Cache a sparse vector, evicting oldest if at capacity."""
        key = self._make_key(query, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = vector

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


class SparseEmbeddingsClient:
    """Generate BM25 sparse vectors for hybrid search.

    Uses FastEmbed's Qdrant/bm25 model to create sparse vectors
    that complement voyage-context-3 dense vectors for hybrid retrieval.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Initialize the sparse embeddings client.

        Args:
            model_name: FastEmbed model name. Defaults to Qdrant/bm25.
        """
        logger.info("Loading sparse embedding model", model=model_name)
        self._model = SparseTextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._cache = SparseQueryCache()

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse vectors for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of SparseVector objects for Qdrant.
        """
        if not texts:
            return []

        embeddings = list(self._model.embed(texts))
        sparse_vectors = [
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            for emb in embeddings
        ]

        logger.debug("Generated sparse embeddings", count=len(sparse_vectors))
        return sparse_vectors

    def embed_query(self, query: str, use_cache: bool = True) -> SparseVector:
        """Generate sparse vector for a single query with LRU caching.

        Args:
            query: Query text to embed.
            use_cache: Whether to use the query cache (default True).

        Returns:
            SparseVector for the query.
        """
        # Check cache first
        if use_cache:
            cached = self._cache.get(query, self._model_name)
            if cached is not None:
                logger.debug("Sparse query cache hit", query_len=len(query))
                return cached

        result = self.embed_texts([query])
        vector = result[0] if result else SparseVector(indices=[], values=[])

        # Cache the result
        if use_cache:
            self._cache.set(query, self._model_name, vector)

        return vector

    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self._cache.stats()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> list[SparseVector]:
        """Generate sparse vectors in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Size of each batch.

        Returns:
            List of SparseVector objects.
        """
        if not texts:
            return []

        all_vectors: list[SparseVector] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embed_texts(batch)
            all_vectors.extend(vectors)

            logger.debug(
                "Processed sparse embedding batch",
                batch_num=i // batch_size + 1,
                total_batches=(len(texts) + batch_size - 1) // batch_size,
            )

        return all_vectors
