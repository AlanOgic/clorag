"""Chunk management tools for CLORAG MCP server.

Provides low-level access to Qdrant vector chunks: listing,
reading, editing, and searching with metadata filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices

# Map short names to Qdrant collection names
COLLECTION_MAP = {
    "docs": "docusaurus_docs",
    "cases": "gmail_cases",
    "custom": "custom_docs",
    "docusaurus_docs": "docusaurus_docs",
    "gmail_cases": "gmail_cases",
    "custom_docs": "custom_docs",
}

VALID_COLLECTIONS = ", ".join(
    ["docs", "cases", "custom"]
)


def _resolve_collection(name: str) -> str | None:
    """Resolve collection short name to full Qdrant name."""
    return COLLECTION_MAP.get(name.lower())


def _search_result_dict(r: Any, score: float) -> dict[str, Any]:
    """Format a SearchResult for API output."""
    return {
        "text": r.text,
        "score": round(score, 4),
        "id": str(r.id),
        "title": r.payload.get(
            "title", r.payload.get("subject", ""),
        ),
        "url": r.payload.get("url", ""),
        "chunk_index": r.payload.get("chunk_index"),
        "category": r.payload.get("category", ""),
        "keywords": r.payload.get("keywords", []),
    }


def _chunk_to_dict(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert raw chunk dict to clean serializable output."""
    payload = chunk.get("payload", {})
    return {
        "id": str(chunk.get("id", "")),
        "text": payload.get("text", ""),
        "chunk_index": payload.get("chunk_index"),
        "total_chunks": payload.get("total_chunks"),
        "source": payload.get("_source", ""),
        "title": payload.get("title", ""),
        "url": payload.get("url", ""),
        "subject": payload.get("subject", ""),
        "thread_id": payload.get("thread_id", ""),
        "parent_doc_id": payload.get("parent_doc_id", ""),
        "category": payload.get("category", ""),
        "keywords": payload.get("keywords", []),
        "human_edited": payload.get("human_edited", False),
    }


def register_chunk_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register chunk management MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    async def list_chunks(
        collection: str,
        limit: int = 20,
        offset: str | None = None,
        field: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        """List chunks from a Qdrant collection with pagination.

        Optionally filter by a metadata field (e.g. all chunks
        from the same document URL or thread).

        Args:
            collection: Collection to list from.
                Valid: "docs", "cases", "custom".
            limit: Maximum chunks to return (1-50, default 20).
            offset: Pagination cursor from previous response.
            field: Optional metadata field to filter on
                (e.g. "url", "thread_id", "parent_doc_id").
            value: Value to match for the field filter.
                Required if field is set.

        Returns:
            Paginated list of chunks with metadata and
            next_offset for pagination.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        coll = _resolve_collection(collection)
        if not coll:
            return {
                "error": f"Invalid collection '{collection}'."
                f" Valid: {VALID_COLLECTIONS}",
            }

        limit = max(1, min(50, limit))

        # If field+value filter, use get_chunks_by_field
        if field and value:
            chunks = await services.vectorstore.get_chunks_by_field(
                collection=coll,
                field=field,
                value=value,
                max_chunks=limit,
            )
            return {
                "collection": coll,
                "filter": {"field": field, "value": value},
                "total_returned": len(chunks),
                "chunks": [_chunk_to_dict(c) for c in chunks],
            }

        # Otherwise, paginated scroll
        filter_conditions = None
        if field and not value:
            return {
                "error": "'value' is required when 'field' is set",
            }

        chunks, next_offset = await services.vectorstore.scroll_chunks(
            collection=coll,
            limit=limit,
            offset=offset,
            filter_conditions=filter_conditions,
        )

        return {
            "collection": coll,
            "total_returned": len(chunks),
            "next_offset": next_offset,
            "chunks": [_chunk_to_dict(c) for c in chunks],
        }

    @mcp.tool()
    async def get_chunk(
        collection: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        """Get full details of a specific chunk by ID.

        Returns the chunk text, all metadata fields, and
        position within its parent document.

        Args:
            collection: Collection containing the chunk.
                Valid: "docs", "cases", "custom".
            chunk_id: Chunk ID (UUID or composite ID).

        Returns:
            Full chunk details with text and metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        coll = _resolve_collection(collection)
        if not coll:
            return {
                "error": f"Invalid collection '{collection}'."
                f" Valid: {VALID_COLLECTIONS}",
            }

        chunk = await services.vectorstore.get_chunk(
            collection=coll,
            chunk_id=chunk_id,
        )

        if not chunk:
            return {
                "error": f"Chunk '{chunk_id}' not found"
                f" in {coll}",
            }

        return {"chunk": _chunk_to_dict(chunk)}

    @mcp.tool()
    async def edit_chunk(
        collection: str,
        chunk_id: str,
        text: str | None = None,
        title: str | None = None,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """Edit a chunk's text and/or metadata.

        If text is changed, new dense and sparse embeddings are
        automatically generated. The chunk is marked as
        human_edited to prevent overwrite during re-ingestion.

        Args:
            collection: Collection containing the chunk.
                Valid: "docs", "cases", "custom".
            chunk_id: Chunk ID to edit.
            text: New text content (triggers re-embedding).
            title: New title metadata.
            subject: New subject metadata (for support cases).

        Returns:
            Updated chunk details.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        coll = _resolve_collection(collection)
        if not coll:
            return {
                "error": f"Invalid collection '{collection}'."
                f" Valid: {VALID_COLLECTIONS}",
            }

        if text is None and title is None and subject is None:
            return {"error": "At least one field must be provided"}

        # Build metadata updates
        metadata_updates: dict[str, Any] = {"human_edited": True}
        if title is not None:
            metadata_updates["title"] = title
        if subject is not None:
            metadata_updates["subject"] = subject

        # If text changed, generate new embeddings
        dense_vector = None
        sparse_vector = None
        if text is not None:
            import asyncio

            metadata_updates["text"] = text

            dense_task = services.embeddings.embed_documents(
                [text],
            )
            sparse_task = asyncio.to_thread(
                services.sparse_embeddings.embed_texts, [text],
            )

            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task,
            )
            dense_vector = dense_results[0]
            sparse_vector = sparse_results[0]

        success = await services.vectorstore.update_chunk(
            collection=coll,
            chunk_id=chunk_id,
            text=text,
            metadata_updates=metadata_updates,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )

        if not success:
            return {
                "error": f"Chunk '{chunk_id}' not found"
                f" in {coll}",
            }

        # Fetch updated chunk
        updated = await services.vectorstore.get_chunk(
            collection=coll, chunk_id=chunk_id,
        )

        return {
            "status": "updated",
            "re_embedded": text is not None,
            "chunk": _chunk_to_dict(updated) if updated else None,
        }

    @mcp.tool()
    async def delete_chunk(
        collection: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        """Delete a chunk from a Qdrant collection.

        This is irreversible. The chunk and its vectors will be
        permanently removed.

        Args:
            collection: Collection containing the chunk.
                Valid: "docs", "cases", "custom".
            chunk_id: Chunk ID to delete.

        Returns:
            Confirmation of deletion.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        coll = _resolve_collection(collection)
        if not coll:
            return {
                "error": f"Invalid collection '{collection}'."
                f" Valid: {VALID_COLLECTIONS}",
            }

        # Verify chunk exists before deleting
        existing = await services.vectorstore.get_chunk(
            collection=coll, chunk_id=chunk_id,
        )
        if not existing:
            return {
                "error": f"Chunk '{chunk_id}' not found"
                f" in {coll}",
            }

        await services.vectorstore.delete_chunk(
            collection=coll, chunk_id=chunk_id,
        )

        return {
            "status": "deleted",
            "collection": coll,
            "chunk_id": chunk_id,
        }

    @mcp.tool()
    async def search_chunks(
        query: str,
        collection: str,
        limit: int = 10,
        field: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        """Search chunks within a specific collection with
        optional metadata filtering.

        Uses hybrid search (dense + sparse with RRF fusion)
        and reranking within a single collection. Optionally
        filter by metadata field to scope results.

        Args:
            query: Search query text.
            collection: Collection to search in.
                Valid: "docs", "cases", "custom".
            limit: Maximum results (1-20, default 10).
            field: Optional metadata field to pre-filter on
                (e.g. "url", "thread_id", "category",
                "parent_doc_id", "manufacturer").
            value: Value for the metadata filter.
                Required if field is set.

        Returns:
            Ranked search results with scores and metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        coll = _resolve_collection(collection)
        if not coll:
            return {
                "error": f"Invalid collection '{collection}'."
                f" Valid: {VALID_COLLECTIONS}",
            }

        if field and not value:
            return {
                "error": "'value' is required when 'field' is set",
            }

        limit = max(1, min(20, limit))

        import asyncio

        dense_task = services.embeddings.embed_query(query)
        sparse_task = asyncio.to_thread(
            services.sparse_embeddings.embed_query, query,
        )

        dense_vector, sparse_vector = await asyncio.gather(
            dense_task, sparse_task,
        )

        over_fetch = min(limit * 3, 50)
        filter_dict = {field: value} if field and value else None

        results = await services.vectorstore.search_hybrid_rrf(
            collection=coll,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=over_fetch,
            match_filters=filter_dict,
        )

        # Rerank if available
        rerank_enabled = services.retriever.rerank_enabled
        if rerank_enabled and results:
            texts = [r.text for r in results]
            rerank_resp = services.reranker.rerank(
                query=query,
                documents=texts,
                top_k=limit,
            )
            final = []
            for item in rerank_resp.results:
                if item.index < len(results):
                    r = results[item.index]
                    final.append(
                        _search_result_dict(r, item.relevance_score),
                    )
        else:
            final = [
                _search_result_dict(r, r.score)
                for r in results[:limit]
            ]

        return {
            "query": query,
            "collection": coll,
            "filter": (
                {"field": field, "value": value}
                if field else None
            ),
            "reranked": rerank_enabled,
            "total_found": len(final),
            "results": final,
        }
