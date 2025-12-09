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
        self._api_key = api_key or (settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None)
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

    async def get_chunk(
        self,
        collection: str,
        chunk_id: str,
        with_vectors: bool = False,
    ) -> dict[str, Any] | None:
        """Retrieve a single chunk by ID.

        Args:
            collection: Collection name (docusaurus_docs or gmail_cases).
            chunk_id: UUID of the chunk to retrieve.
            with_vectors: Whether to include dense/sparse vectors.

        Returns:
            Dict with id, payload, and optionally vectors, or None if not found.
        """
        results = await self._client.retrieve(
            collection_name=collection,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=with_vectors,
        )
        if not results:
            return None

        point = results[0]
        result: dict[str, Any] = {
            "id": str(point.id),
            "payload": point.payload or {},
        }
        if with_vectors and point.vector:
            result["vectors"] = point.vector
        return result

    async def scroll_chunks(
        self,
        collection: str,
        limit: int = 20,
        offset: str | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Paginated listing of chunks with optional filtering.

        Args:
            collection: Collection name.
            limit: Maximum chunks per page.
            offset: Point ID to start from (for pagination).
            filter_conditions: Optional filters (e.g., {"parent_case_id": "xxx"}).

        Returns:
            Tuple of (list of chunk dicts, next_offset for pagination).
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

        records, next_offset = await self._client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        chunks = [
            {
                "id": str(record.id),
                "payload": record.payload or {},
            }
            for record in records
        ]

        # next_offset is a PointId which can be str or int
        next_offset_str = str(next_offset) if next_offset is not None else None
        return chunks, next_offset_str

    async def update_chunk(
        self,
        collection: str,
        chunk_id: str,
        text: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
        dense_vector: list[float] | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> bool:
        """Update a chunk's text, metadata, and/or vectors.

        Args:
            collection: Collection name.
            chunk_id: UUID of the chunk.
            text: New text content (should provide vectors if text changes).
            metadata_updates: Dict of metadata fields to update.
            dense_vector: New dense embedding (required if text changes).
            sparse_vector: New sparse embedding (required if text changes).

        Returns:
            True if successful, False if chunk not found.
        """
        # Check if chunk exists
        existing = await self.get_chunk(collection, chunk_id)
        if not existing:
            return False

        # Build payload updates
        payload_updates: dict[str, Any] = {}
        if text is not None:
            payload_updates["text"] = text
        if metadata_updates:
            payload_updates.update(metadata_updates)

        # Update payload if there are changes
        if payload_updates:
            await self._client.set_payload(
                collection_name=collection,
                payload=payload_updates,
                points=[chunk_id],
            )

        # Update vectors if provided
        if dense_vector is not None or sparse_vector is not None:
            vectors: dict[str, Any] = {}
            if dense_vector is not None:
                vectors["dense"] = dense_vector
            if sparse_vector is not None:
                vectors["sparse"] = sparse_vector

            await self._client.update_vectors(
                collection_name=collection,
                points=[
                    models.PointVectors(
                        id=chunk_id,
                        vector=vectors,
                    )
                ],
            )

        return True

    async def delete_chunk(
        self,
        collection: str,
        chunk_id: str,
    ) -> bool:
        """Delete a single chunk by ID.

        Args:
            collection: Collection name.
            chunk_id: UUID of the chunk to delete.

        Returns:
            True if successful.
        """
        await self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[chunk_id]),
        )
        return True
