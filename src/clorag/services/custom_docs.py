"""Service for managing custom documents in the knowledge base."""

from datetime import datetime
from uuid import uuid4

import structlog

from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.chunker import ContentType, SemanticChunker
from clorag.models.custom_document import (
    CustomDocument,
    CustomDocumentCreate,
    CustomDocumentListItem,
    CustomDocumentUpdate,
    DocumentCategory,
)

logger = structlog.get_logger(__name__)


class CustomDocumentService:
    """Service for CRUD operations on custom documents."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embeddings: EmbeddingsClient | None = None,
        sparse_embeddings: SparseEmbeddingsClient | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            vectorstore: VectorStore instance.
            embeddings: EmbeddingsClient instance.
            sparse_embeddings: SparseEmbeddingsClient instance.
        """
        self._vectorstore = vectorstore or VectorStore()
        self._embeddings = embeddings or EmbeddingsClient()
        self._sparse_embeddings = sparse_embeddings or SparseEmbeddingsClient()
        # Chunker is created per-document based on category (see create_document)

    async def create_document(
        self,
        doc: CustomDocumentCreate,
        created_by: str | None = None,
    ) -> CustomDocument:
        """Create and ingest a new custom document.

        Args:
            doc: Document creation data.
            created_by: Admin who created this document.

        Returns:
            Created CustomDocument with ID.
        """
        # Ensure collection exists
        await self._vectorstore.ensure_collections(hybrid=True)

        # Generate document ID
        doc_id = str(uuid4())
        now = datetime.utcnow()

        # Chunk the content with semantic awareness
        # Map document category to content type for optimal chunking
        content_type = ContentType.GENERIC
        if doc.category.value in ("troubleshooting", "configuration"):
            content_type = ContentType.SUPPORT_CASE  # Similar structure to support cases
        elif doc.category.value == "faq":
            content_type = ContentType.FAQ
        elif doc.category.value == "release_notes":
            content_type = ContentType.RELEASE_NOTES

        # Create chunker with settings-based token-aware sizing for this content type
        chunker = SemanticChunker.from_settings(content_type)
        chunks = chunker.chunk_text(doc.content, content_type=content_type)

        if not chunks:
            # Single chunk for short content
            chunk_texts = [doc.content]
            chunk_meta_list: list[dict[str, str | int | bool]] = [{}]
        else:
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_meta_list = [chunk.metadata for chunk in chunks]

        # Generate embeddings for all chunks
        # Dense embeddings
        dense_result = await self._embeddings.embed_documents(
            chunk_texts,
            input_type="document",
        )
        dense_vectors = dense_result.vectors

        # Sparse embeddings
        sparse_vectors = self._sparse_embeddings.embed_batch(chunk_texts)

        # Prepare metadata for each chunk
        base_metadata = {
            "source": "custom_docs",
            "title": doc.title,
            "tags": doc.tags,
            "category": doc.category.value,
            "url_reference": doc.url_reference,
            "expiration_date": doc.expiration_date.isoformat() if doc.expiration_date else None,
            "notes": doc.notes,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": created_by,
            "parent_doc_id": doc_id,
        }

        # Generate chunk IDs and metadata
        chunk_ids = []
        chunk_metadata = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            # Include semantic metadata (section, has_code, etc.)
            semantic_meta = chunk_meta_list[i] if i < len(chunk_meta_list) else {}
            chunk_metadata.append({
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "text": chunk_text,
                # Add semantic metadata, excluding chunk_index since we already have it
                **{k: v for k, v in semantic_meta.items() if k != "chunk_index"},
            })

        # Store in Qdrant
        await self._vectorstore.upsert_documents_hybrid(
            collection=self._vectorstore.custom_docs_collection,
            texts=chunk_texts,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            metadata=chunk_metadata,
            ids=chunk_ids,
        )

        logger.info(
            "Created custom document",
            doc_id=doc_id,
            title=doc.title,
            chunks=len(chunk_texts),
        )

        return CustomDocument(
            id=doc_id,
            title=doc.title,
            content=doc.content,
            tags=doc.tags,
            category=doc.category,
            url_reference=doc.url_reference,
            expiration_date=doc.expiration_date,
            notes=doc.notes,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

    async def get_document(self, doc_id: str) -> CustomDocument | None:
        """Get a document by ID (reconstructs from chunks).

        Args:
            doc_id: Parent document ID.

        Returns:
            CustomDocument or None if not found.
        """
        # Search for all chunks with this parent_doc_id
        chunks, _ = await self._vectorstore.scroll_chunks(
            collection=self._vectorstore.custom_docs_collection,
            limit=100,
            filter_conditions={"parent_doc_id": doc_id},
        )

        if not chunks:
            return None

        # Sort by chunk_index and reconstruct
        chunks.sort(key=lambda c: c["payload"].get("chunk_index", 0))

        # Reconstruct content
        content_parts = [c["payload"].get("text", "") for c in chunks]
        content = "\n\n".join(content_parts)

        # Get metadata from first chunk
        first_chunk = chunks[0]["payload"]

        return CustomDocument(
            id=doc_id,
            title=first_chunk.get("title", ""),
            content=content,
            tags=first_chunk.get("tags", []),
            category=DocumentCategory(first_chunk.get("category", "other")),
            url_reference=first_chunk.get("url_reference"),
            expiration_date=datetime.fromisoformat(first_chunk["expiration_date"])
            if first_chunk.get("expiration_date")
            else None,
            notes=first_chunk.get("notes"),
            created_at=datetime.fromisoformat(first_chunk["created_at"])
            if first_chunk.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(first_chunk["updated_at"])
            if first_chunk.get("updated_at")
            else None,
            created_by=first_chunk.get("created_by"),
        )

    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        include_expired: bool = False,
    ) -> tuple[list[CustomDocumentListItem], int]:
        """List custom documents with pagination (unique by parent_doc_id).

        Uses incremental scrolling to avoid loading all chunks at once.

        Args:
            limit: Maximum number of documents to return per page.
            offset: Number of documents to skip (for pagination).
            category: Filter by category.
            include_expired: Include expired documents.

        Returns:
            Tuple of (list of CustomDocumentListItem, total count).
        """
        # Scroll through chunks incrementally to find unique documents
        # Only fetch first chunks (chunk_index=0) to get document metadata
        docs_map: dict[str, dict] = {}
        scroll_offset: str | None = None
        batch_size = 100  # Scroll in batches instead of loading all

        try:
            while True:
                chunks, next_offset = await self._vectorstore.scroll_chunks(
                    collection=self._vectorstore.custom_docs_collection,
                    limit=batch_size,
                    offset=scroll_offset,
                )

                if not chunks:
                    break

                # Group by parent_doc_id, keep only first chunk (chunk_index=0)
                for chunk in chunks:
                    payload = chunk["payload"]
                    parent_id = payload.get("parent_doc_id")
                    if parent_id and payload.get("chunk_index", 0) == 0:
                        if parent_id not in docs_map:
                            docs_map[parent_id] = payload

                # Stop if we've collected enough documents (with buffer for filtering)
                # Only stop early if we have way more than needed
                if len(docs_map) >= (limit + offset) * 3:
                    break

                scroll_offset = next_offset
                if scroll_offset is None:
                    break

        except Exception:
            # Collection might not exist yet - return empty list
            return [], 0

        # Convert to list items with filtering
        now = datetime.utcnow()
        items = []
        for doc_id, payload in docs_map.items():
            exp_date = None
            is_expired = False
            if payload.get("expiration_date"):
                exp_date = datetime.fromisoformat(payload["expiration_date"])
                is_expired = exp_date < now

            if not include_expired and is_expired:
                continue

            if category and payload.get("category") != category:
                continue

            content = payload.get("text", "")
            items.append(
                CustomDocumentListItem(
                    id=doc_id,
                    title=payload.get("title", "Untitled"),
                    category=DocumentCategory(payload.get("category", "other")),
                    tags=payload.get("tags", []),
                    content_preview=content[:200] + "..." if len(content) > 200 else content,
                    expiration_date=exp_date,
                    created_at=datetime.fromisoformat(payload["created_at"])
                    if payload.get("created_at")
                    else None,
                    is_expired=is_expired,
                )
            )

        # Sort by created_at descending
        items.sort(key=lambda x: x.created_at or datetime.min, reverse=True)

        total_count = len(items)
        # Apply pagination
        paginated_items = items[offset : offset + limit]

        return paginated_items, total_count

    async def update_document(
        self,
        doc_id: str,
        updates: CustomDocumentUpdate,
    ) -> CustomDocument | None:
        """Update a document (re-embeds if content changes).

        Args:
            doc_id: Document ID to update.
            updates: Fields to update.

        Returns:
            Updated CustomDocument or None if not found.
        """
        existing = await self.get_document(doc_id)
        if not existing:
            return None

        # Delete existing chunks
        await self.delete_document(doc_id)

        # Create new document with updates
        create_data = CustomDocumentCreate(
            title=updates.title or existing.title,
            content=updates.content or existing.content,
            tags=updates.tags if updates.tags is not None else existing.tags,
            category=updates.category or existing.category,
            url_reference=updates.url_reference
            if updates.url_reference is not None
            else existing.url_reference,
            expiration_date=updates.expiration_date
            if updates.expiration_date is not None
            else existing.expiration_date,
            notes=updates.notes if updates.notes is not None else existing.notes,
        )

        # Re-create (generates new ID, but that's acceptable for now)
        return await self.create_document(create_data, existing.created_by)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            doc_id: Document ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Find all chunks with this parent_doc_id
        chunks, _ = await self._vectorstore.scroll_chunks(
            collection=self._vectorstore.custom_docs_collection,
            limit=100,
            filter_conditions={"parent_doc_id": doc_id},
        )

        if not chunks:
            return False

        # Delete each chunk
        for chunk in chunks:
            await self._vectorstore.delete_chunk(
                self._vectorstore.custom_docs_collection,
                chunk["id"],
            )

        logger.info("Deleted custom document", doc_id=doc_id, chunks=len(chunks))
        return True

    async def get_categories(self) -> list[dict[str, str]]:
        """Get available document categories.

        Returns:
            List of category dicts with value and label.
        """
        return [
            {"value": cat.value, "label": cat.value.replace("_", " ").title()}
            for cat in DocumentCategory
        ]
