"""Qdrant vector store client for storing and retrieving embeddings."""

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from clorag.config import get_settings

logger = structlog.get_logger(__name__)


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
        self._api_key = api_key or (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key else None
        )
        self._dimensions = dimensions or settings.voyage_dimensions
        self._docs_collection = settings.qdrant_docs_collection
        self._cases_collection = settings.qdrant_cases_collection
        self._custom_docs_collection = settings.qdrant_custom_docs_collection

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

    @property
    def custom_docs_collection(self) -> str:
        """Get custom docs collection name."""
        return self._custom_docs_collection

    async def ensure_collections(self, hybrid: bool = True) -> None:
        """Ensure all collections exist with correct configuration.

        Args:
            hybrid: If True, create collections with dense + sparse vector support.
        """
        for collection_name in [
            self._docs_collection,
            self._cases_collection,
            self._custom_docs_collection,
        ]:
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
            query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]  # type: ignore[arg-type]

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
        match_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining dense + sparse vectors with RRF fusion.

        Uses Qdrant's prefetch + RRF fusion to combine semantic (dense)
        and keyword (sparse/BM25) search results.

        Args:
            collection: Collection to search in.
            dense_vector: Dense embedding vector (voyage-context-3).
            sparse_vector: Sparse BM25 vector.
            limit: Maximum number of results.
            match_filters: Optional dict of exact-match field filters applied
                to both prefetch stages (e.g. ``{"category": "troubleshooting"}``).

        Returns:
            List of SearchResult objects ranked by RRF fusion.
        """
        # Dynamic prefetch: fetch more candidates for better RRF fusion quality
        # Scale with requested limit but cap to avoid excessive retrieval
        try:
            from clorag.services.settings_manager import get_setting
            prefetch_mult = int(get_setting("prefetch.multiplier"))
            prefetch_max = int(get_setting("prefetch.max_limit"))
        except (KeyError, ImportError, Exception):
            prefetch_mult = 3
            prefetch_max = 50
        prefetch_limit = min(limit * prefetch_mult, prefetch_max)

        # Build Qdrant filter from exact-match dict
        query_filter = None
        if match_filters:
            must_conditions = [
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in match_filters.items()
            ]
            query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

        response = await self._client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
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

    async def search_custom_docs_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search in custom docs collection.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse BM25 vector.
            limit: Max results.

        Returns:
            Search results from custom docs collection.
        """
        try:
            return await self.search_hybrid_rrf(
                collection=self._custom_docs_collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
            )
        except UnexpectedResponse as e:
            # Collection not found is expected if no custom docs uploaded yet
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                logger.debug(
                    "Custom docs collection not found",
                    collection=self._custom_docs_collection,
                )
                return []
            # Other Qdrant errors should be logged and re-raised
            logger.error(
                "Qdrant error in custom docs search",
                error=str(e),
                collection=self._custom_docs_collection,
            )
            raise
        except Exception as e:
            # Unexpected errors should be logged and re-raised
            logger.error(
                "Unexpected error in custom docs search",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    async def hybrid_search_rrf(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 10,
        include_custom_docs: bool = True,
    ) -> list[SearchResult]:
        """Search across all collections with RRF hybrid search in parallel.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse BM25 vector.
            limit: Max results per collection.
            include_custom_docs: Whether to include custom docs collection.

        Returns:
            Merged and RRF-sorted results from all collections.
        """
        # Build list of search tasks
        search_tasks = [
            self.search_docs_hybrid(dense_vector, sparse_vector, limit),
            self.search_cases_hybrid(dense_vector, sparse_vector, limit),
        ]
        if include_custom_docs:
            search_tasks.append(
                self.search_custom_docs_hybrid(dense_vector, sparse_vector, limit)
            )

        # Search collections in parallel for better performance
        results = await asyncio.gather(*search_tasks)
        docs_results = results[0]
        cases_results = results[1]
        custom_results = results[2] if include_custom_docs else []

        # Mark source in payload
        for r in docs_results:
            r.payload["_source"] = "documentation"
        for r in cases_results:
            r.payload["_source"] = "gmail_case"
        for r in custom_results:
            r.payload["_source"] = "custom_docs"

        # Merge and sort by RRF score
        all_results = docs_results + cases_results + custom_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Source diversity: ensure at least 1 result from each collection
        # that has relevant hits, to avoid top-N being dominated by one source
        if limit >= 3 and len(all_results) > limit:
            top_results = all_results[:limit]
            sources_in_top = {r.payload.get("_source") for r in top_results}

            # Check each collection for missing representation
            source_pools = {
                "documentation": docs_results,
                "gmail_case": cases_results,
                "custom_docs": custom_results,
            }
            for source_name, pool in source_pools.items():
                if source_name not in sources_in_top and pool:
                    # Insert the best result from this source, replacing the
                    # lowest-scoring result in the top set
                    best_from_source = pool[0]  # Already sorted by score
                    # Only add if the source has a reasonable score
                    # (at least 50% of the top result's score)
                    try:
                        from clorag.services.settings_manager import get_setting
                        diversity_thresh = float(
                            get_setting("reranking.source_diversity_threshold")
                        )
                    except (KeyError, ImportError, Exception):
                        diversity_thresh = 0.5
                    min_score = top_results[0].score * diversity_thresh
                    if top_results and best_from_source.score >= min_score:
                        top_results[-1] = best_from_source
                        top_results.sort(key=lambda x: x.score, reverse=True)

            return top_results

        return all_results[:limit]

    async def create_snapshot(self, collection: str) -> str:
        """Create a server-side snapshot of a collection.

        Args:
            collection: Collection name to snapshot.

        Returns:
            Snapshot name (filename on the Qdrant server).
        """
        snapshot = await self._client.create_snapshot(
            collection_name=collection, wait=True
        )
        return snapshot.name  # type: ignore[union-attr]

    async def list_snapshots(self, collection: str) -> list[dict[str, Any]]:
        """List existing snapshots for a collection.

        Args:
            collection: Collection name.

        Returns:
            List of dicts with name, creation_time, size.
        """
        snapshots = await self._client.list_snapshots(collection_name=collection)
        return [
            {
                "name": s.name,
                "creation_time": str(s.creation_time) if s.creation_time else None,
                "size": s.size,
            }
            for s in snapshots
        ]

    async def recover_snapshot(self, collection: str, snapshot_name: str) -> None:
        """Recover a collection from a server-local snapshot.

        Args:
            collection: Collection name to recover into.
            snapshot_name: Name of the snapshot file on the server.
        """
        location = f"{self._url}/collections/{collection}/snapshots/{snapshot_name}"
        await self._client.recover_snapshot(
            collection_name=collection, location=location, wait=True
        )

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
            query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

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

    async def get_chunks_by_field(
        self,
        collection: str,
        field: str,
        value: str,
        with_vectors: bool = False,
        max_chunks: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all chunks matching a field value (e.g., all chunks from same document).

        Args:
            collection: Collection name.
            field: Metadata field to filter on (e.g., 'url', 'thread_id', 'parent_doc_id').
            value: Value to match.
            with_vectors: Whether to include vectors in response.
            max_chunks: Maximum chunks to return (safety limit).

        Returns:
            List of chunk dicts sorted by chunk_index, each containing id, payload,
            and optionally vectors.
        """
        chunks: list[dict[str, Any]] = []
        offset: str | None = None

        while len(chunks) < max_chunks:
            batch, next_offset = await self.scroll_chunks(
                collection=collection,
                limit=50,
                offset=offset,
                filter_conditions={field: value},
            )

            if not batch:
                break

            # If we need vectors, fetch them in a single batch request (not N+1)
            if with_vectors and batch:
                chunk_ids = [chunk["id"] for chunk in batch]
                results = await self._client.retrieve(
                    collection_name=collection,
                    ids=chunk_ids,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in results:
                    chunk_dict: dict[str, Any] = {
                        "id": str(point.id),
                        "payload": point.payload or {},
                    }
                    if point.vector:
                        chunk_dict["vectors"] = point.vector
                    chunks.append(chunk_dict)
            else:
                chunks.extend(batch)

            if not next_offset or len(chunks) >= max_chunks:
                break
            offset = next_offset

        # Sort by chunk_index for correct document order
        chunks.sort(key=lambda c: c.get("payload", {}).get("chunk_index", 0))

        return chunks[:max_chunks]

    async def update_chunks_vectors_batch(
        self,
        collection: str,
        updates: list[tuple[str, list[float], SparseVector]],
    ) -> int:
        """Update vectors for multiple chunks in a batch.

        Args:
            collection: Collection name.
            updates: List of (chunk_id, dense_vector, sparse_vector) tuples.

        Returns:
            Number of chunks updated.
        """
        if not updates:
            return 0

        # Build points for batch update
        points = [
            models.PointVectors(
                id=chunk_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
            )
            for chunk_id, dense_vec, sparse_vec in updates
        ]

        # Update in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.update_vectors(
                collection_name=collection,
                points=batch,
            )

        return len(updates)
