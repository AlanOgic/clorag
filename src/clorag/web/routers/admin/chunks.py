"""Admin chunk management endpoints.

Provides listing, viewing, updating, and deleting of chunks in Qdrant collections.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter
from clorag.web.schemas import (
    ChunkCollection,
    ChunkDetail,
    ChunkListItem,
    ChunkListResponse,
    ChunkUpdate,
)
from clorag.web.search import generate_embeddings_parallel, get_vectorstore

router = APIRouter(tags=["Chunks"])
logger = structlog.get_logger()


@router.get("/chunks", response_model=ChunkListResponse)
async def api_chunks_list(
    collection: ChunkCollection = ChunkCollection.DOCS,
    limit: int = 20,
    offset: str | None = None,
    search: str | None = None,
    thread_id: str | None = None,
    _: bool = Depends(verify_admin),
) -> ChunkListResponse:
    """List chunks with pagination and optional text search or field filter."""
    vs = get_vectorstore()

    # If thread_id filter provided, use field-based scroll
    if thread_id:
        raw_chunks, next_off = await vs.scroll_chunks(
            collection=collection.value,
            limit=limit,
            offset=offset,
            filter_conditions={"thread_id": thread_id},
        )

        chunks = [
            ChunkListItem(
                id=c["id"],
                collection=collection.value,
                text_preview=(
                    c["payload"].get("text", "")[:200] + "..."
                    if len(c["payload"].get("text", "")) > 200
                    else c["payload"].get("text", "")
                ),
                title=c["payload"].get("title"),
                subject=c["payload"].get("subject"),
                url=c["payload"].get("url"),
                chunk_index=c["payload"].get("chunk_index"),
                source=c["payload"].get("source"),
                thread_id=c["payload"].get("thread_id"),
                parent_case_id=c["payload"].get("parent_case_id"),
            )
            for c in raw_chunks
        ]
        return ChunkListResponse(chunks=chunks, next_offset=next_off, total=len(chunks))

    # If search query provided, use hybrid search
    if search:
        # Generate embeddings in parallel for better latency
        dense_vector, sparse_vector = await generate_embeddings_parallel(search)

        results = await vs.search_hybrid_rrf(
            collection=collection.value,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

        chunks = [
            ChunkListItem(
                id=r.id,
                collection=collection.value,
                text_preview=r.text[:200] + "..." if len(r.text) > 200 else r.text,
                title=r.payload.get("title"),
                subject=r.payload.get("subject"),
                url=r.payload.get("url"),
                chunk_index=r.payload.get("chunk_index"),
                source=r.payload.get("source"),
                thread_id=r.payload.get("thread_id"),
                parent_case_id=r.payload.get("parent_case_id"),
            )
            for r in results
        ]
        return ChunkListResponse(chunks=chunks, next_offset=None, total=len(chunks))

    # Otherwise, scroll through collection
    raw_chunks, next_off = await vs.scroll_chunks(
        collection=collection.value,
        limit=limit,
        offset=offset,
    )

    chunks = [
        ChunkListItem(
            id=c["id"],
            collection=collection.value,
            text_preview=(
                c["payload"].get("text", "")[:200] + "..."
                if len(c["payload"].get("text", "")) > 200
                else c["payload"].get("text", "")
            ),
            title=c["payload"].get("title"),
            subject=c["payload"].get("subject"),
            url=c["payload"].get("url"),
            chunk_index=c["payload"].get("chunk_index"),
            source=c["payload"].get("source"),
            thread_id=c["payload"].get("thread_id"),
            parent_case_id=c["payload"].get("parent_case_id"),
        )
        for c in raw_chunks
    ]

    # Get total count
    info = await vs.get_collection_info(collection.value)
    total = info.get("points_count")

    return ChunkListResponse(chunks=chunks, next_offset=next_off, total=total)


@router.get("/chunks/{collection}/{chunk_id}", response_model=ChunkDetail)
async def api_chunk_get(
    collection: str,
    chunk_id: str,
    _: bool = Depends(verify_admin),
) -> ChunkDetail:
    """Get a single chunk's full details."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()
    chunk = await vs.get_chunk(collection, chunk_id)

    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    payload = chunk["payload"]
    return ChunkDetail(
        id=chunk["id"],
        collection=collection,
        text=payload.get("text", ""),
        source=payload.get("source"),
        chunk_index=payload.get("chunk_index"),
        url=payload.get("url"),
        title=payload.get("title"),
        lastmod=payload.get("lastmod"),
        parent_id=payload.get("parent_id"),
        subject=payload.get("subject"),
        thread_id=payload.get("thread_id"),
        parent_case_id=payload.get("parent_case_id"),
        problem_summary=payload.get("problem_summary"),
        solution_summary=payload.get("solution_summary"),
        category=payload.get("category"),
        product=payload.get("product"),
        keywords=payload.get("keywords"),
        metadata=payload,
    )


@router.put("/chunks/{collection}/{chunk_id}", response_model=ChunkDetail)
@limiter.limit("10/minute")
async def api_chunk_update(
    request: Request,
    collection: str,
    chunk_id: str,
    updates: ChunkUpdate,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> ChunkDetail:
    """Update a chunk. Re-embeds automatically if text changes."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()

    # Get existing chunk
    existing = await vs.get_chunk(collection, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Chunk not found")

    old_text = existing["payload"].get("text", "")
    new_text = updates.text
    text_changed = new_text is not None and new_text != old_text

    # Build metadata updates
    metadata_updates: dict[str, str] = {}
    if updates.title is not None:
        metadata_updates["title"] = updates.title
    if updates.subject is not None:
        metadata_updates["subject"] = updates.subject

    # If text changed, generate new embeddings
    dense_vector = None
    sparse_vector = None
    if text_changed and new_text:
        # Generate embeddings in parallel for better latency
        dense_vector, sparse_vector = await generate_embeddings_parallel(new_text)
        logger.info("Re-embedding chunk", chunk_id=chunk_id, collection=collection)

    # Update chunk
    success = await vs.update_chunk(
        collection=collection,
        chunk_id=chunk_id,
        text=new_text,
        metadata_updates=metadata_updates if metadata_updates else None,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update chunk")

    # Return updated chunk
    return await api_chunk_get(collection, chunk_id, _admin)


@router.delete("/chunks/{collection}/{chunk_id}")
@limiter.limit("10/minute")
async def api_chunk_delete(
    request: Request,
    collection: str,
    chunk_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str]:
    """Delete a single chunk."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()

    # Check if exists
    existing = await vs.get_chunk(collection, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Chunk not found")

    await vs.delete_chunk(collection, chunk_id)
    logger.info("Deleted chunk", chunk_id=chunk_id, collection=collection)

    return {"status": "deleted", "id": chunk_id, "collection": collection}
