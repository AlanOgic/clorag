"""Qdrant vector store client for storing and retrieving embeddings."""

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from clorag.config import get_settings


@dataclass
class SearchResult:
    """Result of a vector search."""

    id: str
    score: float
    payload: dict[str, Any]
    text: str


class VectorStore:
    """Qdrant vector store client for CLORAG.

    Manages two collections:
    - docs: Documentation from Docusaurus
    - gmail_cases: Support cases from Gmail threads
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL. Defaults to QDRANT_URL env var.
            api_key: Qdrant API key. Defaults to QDRANT_API_KEY env var.
            dimensions: Vector dimensions. Defaults to VOYAGE_DIMENSIONS env var.
        """
        settings = get_settings()
        self._url = url or settings.qdrant_url
        self._api_key = api_key or settings.qdrant_api_key
        self._dimensions = dimensions or settings.voyage_dimensions
        self._docs_collection = settings.qdrant_docs_collection
        self._cases_collection = settings.qdrant_cases_collection

        # Parse URL to extract host, port, https, and prefix for reverse proxy support
        parsed = urlparse(self._url)
        host = parsed.hostname or "localhost"
        use_https = parsed.scheme == "https"
        port = parsed.port or (443 if use_https else 6333)
        prefix = parsed.path.rstrip("/") if parsed.path and parsed.path != "/" else None

        # Initialize async Qdrant client with proper HTTPS reverse proxy config
        self._client = AsyncQdrantClient(
            host=host,
            port=port,
            https=use_https,
            api_key=self._api_key,
            prefix=prefix,
            check_compatibility=False,  # Skip version check behind proxy
        )

    @property
    def docs_collection(self) -> str:
        """Get docs collection name."""
        return self._docs_collection

    @property
    def cases_collection(self) -> str:
        """Get cases collection name."""
        return self._cases_collection

    async def ensure_collections(self, hybrid: bool = True) -> None:
        """Ensure both collections exist with correct configuration.

        Args:
            hybrid: If True, create collections with dense + sparse vector support.
        """
        for collection_name in [self._docs_collection, self._cases_collection]:
            if hybrid:
                await self._ensure_collection_hybrid(collection_name)
            else:
                await self._ensure_collection(collection_name)

    async def _ensure_collection(self, collection_name: str) -> None:
        """Create collection if it doesn't exist (dense vectors only).

        Args:
            collection_name: Name of the collection to ensure.
        """
        collections_response = await self._client.get_collections()
        existing_names = {c.name for c in collections_response.collections}

        if collection_name not in existing_names:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )

    async def _ensure_collection_hybrid(self, collection_name: str) -> None:
        """Create collection with dense + sparse vector support for hybrid search.

        Args:
            collection_name: Name of the collection to ensure.
        """
        collections_response = await self._client.get_collections()
        existing_names = {c.name for c in collections_response.collections}

        if collection_name not in existing_names:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self._dimensions,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                        modifier=Modifier.IDF,  # Required for BM25
                    ),
                },
            )

    async def upsert_documents(
        self,
        collection: str,
        texts: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Upsert documents into a collection.

        Args:
            collection: Collection name (docs or cases).
            texts: Original text content.
            vectors: Embedding vectors.
            metadata: Optional metadata for each document.
            ids: Optional IDs (generated if not provided).

        Returns:
            List of document IDs.
        """
        if not texts or not vectors:
            return []

        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")

        # Generate IDs if not provided
        doc_ids = ids or [str(uuid4()) for _ in texts]

        # Prepare metadata
        payloads = []
        for i, text in enumerate(texts):
            payload = {"text": text}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            payloads.append(payload)

        # Create points
        points = [
            models.PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload,
            )
            for doc_id, vector, payload in zip(doc_ids, vectors, payloads)
        ]

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=collection,
                points=batch,
            )

        return doc_ids

    async def upsert_documents_hybrid(
        self,
        collection: str,
        texts: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Upsert documents with both dense and sparse vectors for hybrid search.

        Args:
            collection: Collection name (docs or cases).
            texts: Original text content.
            dense_vectors: Dense embedding vectors (voyage-context-3).
            sparse_vectors: Sparse BM25 vectors.
            metadata: Optional metadata for each document.
            ids: Optional IDs (generated if not provided).

        Returns:
            List of document IDs.
        """
        if not texts or not dense_vectors or not sparse_vectors:
            return []

        if len(texts) != len(dense_vectors) or len(texts) != len(sparse_vectors):
            raise ValueError("texts, dense_vectors, and sparse_vectors must have the same length")

        # Generate IDs if not provided
        doc_ids = ids or [str(uuid4()) for _ in texts]

        # Prepare metadata
        payloads = []
        for i, text in enumerate(texts):
            payload = {"text": text}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            payloads.append(payload)

        # Create points with named vectors
        points = [
            models.PointStruct(
                id=doc_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload=payload,
            )
            for doc_id, dense_vec, sparse_vec, payload in zip(
                doc_ids, dense_vectors, sparse_vectors, payloads
            )
        ]

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=collection,
                points=batch,
            )

        return doc_ids

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            collection: Collection to search in.
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional filter conditions.

        Returns:
            List of SearchResult objects.
        """
        # Build filter if conditions provided
        query_filter = None
        if filter_conditions:
            must_conditions = [
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filter_conditions.items()
            ]
            query_filter = models.Filter(must=must_conditions)

        # Execute search using query_points (qdrant-client 1.16+)
        response = await self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                text=r.payload.get("text", "") if r.payload else "",
            )
            for r in response.points
        ]

    async def search_docs(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = 0.25,
    ) -> list[SearchResult]:
        """Search documentation collection.

        Args:
            query_vector: Query embedding.
            limit: Max results.
            score_threshold: Min score threshold (0.25 for short queries).

        Returns:
            Search results from docs collection.
        """
        return await self.search(
            collection=self._docs_collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def search_cases(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = 0.25,
    ) -> list[SearchResult]:
        """Search Gmail cases collection.

        Args:
            query_vector: Query embedding.
            limit: Max results.
            score_threshold: Min score threshold (0.25 for short queries).

        Returns:
            Search results from cases collection.
        """
        return await self.search(
            collection=self._cases_collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def hybrid_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = 0.25,
    ) -> list[SearchResult]:
        """Search across both collections and merge results in parallel.

        Args:
            query_vector: Query embedding.
            limit: Max results per collection.
            score_threshold: Min score threshold.

        Returns:
            Merged and sorted results from both collections.
        """
        # Search both collections in parallel for better performance
        docs_results, cases_results = await asyncio.gather(
            self.search_docs(query_vector, limit, score_threshold),
            self.search_cases(query_vector, limit, score_threshold),
        )

        # Mark source in payload
        for r in docs_results:
            r.payload["_source"] = "documentation"
        for r in cases_results:
            r.payload["_source"] = "gmail_case"

        # Merge and sort by score
        all_results = docs_results + cases_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:limit]

    async def search_hybrid_rrf(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search combining dense + sparse vectors with RRF fusion.

        Uses Qdrant's prefetch + RRF fusion to combine semantic (dense)
        and keyword (sparse/BM25) search results.

        Args:
            collection: Collection to search in.
            dense_vector: Dense embedding vector (voyage-context-3).
            sparse_vector: Sparse BM25 vector.
            limit: Maximum number of results.

        Returns:
            List of SearchResult objects ranked by RRF fusion.
        """
        response = await self._client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                text=r.payload.get("text", "") if r.payload else "",
            )
            for r in response.points
        ]

    async def search_docs_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search in documentation collection.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse BM25 vector.
            limit: Max results.

        Returns:
            Search results from docs collection.
        """
        return await self.search_hybrid_rrf(
            collection=self._docs_collection,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

    async def search_cases_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search in Gmail cases collection.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse BM25 vector.
            limit: Max results.

        Returns:
            Search results from cases collection.
        """
        return await self.search_hybrid_rrf(
            collection=self._cases_collection,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

    async def hybrid_search_rrf(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search across both collections with RRF hybrid search in parallel.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse BM25 vector.
            limit: Max results per collection.

        Returns:
            Merged and RRF-sorted results from both collections.
        """
        # Search both collections in parallel for better performance
        docs_results, cases_results = await asyncio.gather(
            self.search_docs_hybrid(dense_vector, sparse_vector, limit),
            self.search_cases_hybrid(dense_vector, sparse_vector, limit),
        )

        # Mark source in payload
        for r in docs_results:
            r.payload["_source"] = "documentation"
        for r in cases_results:
            r.payload["_source"] = "gmail_case"

        # Merge and sort by RRF score
        all_results = docs_results + cases_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:limit]

    async def delete_collection(self, collection: str) -> None:
        """Delete a collection.

        Args:
            collection: Collection name to delete.
        """
        await self._client.delete_collection(collection_name=collection)

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get collection information.

        Args:
            collection: Collection name.

        Returns:
            Collection info dict.
        """
        info = await self._client.get_collection(collection_name=collection)
        return {
            "name": collection,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
