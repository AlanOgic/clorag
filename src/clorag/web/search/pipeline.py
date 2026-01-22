"""Search pipeline functions for RAG retrieval.

This module provides the main search pipeline including hybrid search,
reranking, and graph enrichment.
"""

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.metrics import (
    get_metrics_collector,
    measure_embedding_generation,
    measure_total_search,
    measure_vector_search,
)
from clorag.core.reranker import RerankerClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.web.schemas import SearchRequest, SearchResult, SearchSource
from clorag.web.search.utils import filter_by_dynamic_threshold, truncate

if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector


logger = structlog.get_logger()

# Lazy-loaded service singletons
_vectorstore: VectorStore | None = None
_embeddings: EmbeddingsClient | None = None
_sparse_embeddings: SparseEmbeddingsClient | None = None
_reranker: RerankerClient | None = None
_graph_enrichment_available: bool | None = None
_graph_enrichment_service: Any = None


def get_vectorstore() -> VectorStore:
    """Get or create VectorStore singleton."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore


def get_embeddings() -> EmbeddingsClient:
    """Get or create EmbeddingsClient singleton."""
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingsClient()
    return _embeddings


def get_sparse_embeddings() -> SparseEmbeddingsClient:
    """Get or create SparseEmbeddingsClient singleton."""
    global _sparse_embeddings
    if _sparse_embeddings is None:
        _sparse_embeddings = SparseEmbeddingsClient()
    return _sparse_embeddings


def get_reranker() -> RerankerClient:
    """Get or create RerankerClient singleton."""
    global _reranker
    if _reranker is None:
        _reranker = RerankerClient()
    return _reranker


async def get_graph_enrichment() -> Any:
    """Get graph enrichment service if available (graceful degradation)."""
    global _graph_enrichment_available, _graph_enrichment_service

    if _graph_enrichment_available is False:
        return None

    if _graph_enrichment_available is True:
        return _graph_enrichment_service

    # Try to initialize
    settings = get_settings()
    if not settings.neo4j_password:
        _graph_enrichment_available = False
        logger.info("graph_enrichment_disabled", reason="no_neo4j_password")
        return None

    try:
        from clorag.graph.enrichment import get_enrichment_service
        _graph_enrichment_service = await get_enrichment_service()
        _graph_enrichment_available = True
        logger.info("graph_enrichment_enabled")
        return _graph_enrichment_service
    except Exception as e:
        _graph_enrichment_available = False
        logger.warning("graph_enrichment_unavailable", error=str(e))
        return None


async def generate_embeddings_parallel(
    query: str,
) -> tuple[list[float], "SparseVector"]:
    """Generate dense and sparse embeddings in parallel.

    Runs dense embedding (async API call) and sparse embedding (sync BM25)
    concurrently to reduce query latency by ~100ms.

    Args:
        query: Search query text.

    Returns:
        Tuple of (dense_vector, sparse_vector).
    """
    with measure_embedding_generation(metadata={"query_length": len(query)}):
        emb = get_embeddings()
        sparse_emb = get_sparse_embeddings()

        # Run dense (async) and sparse (sync wrapped) in parallel
        dense_task = emb.embed_query(query)
        sparse_task = asyncio.to_thread(sparse_emb.embed_query, query)

        dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)
    return dense_vector, sparse_vector


async def perform_search(
    req: SearchRequest,
) -> tuple[list[SearchResult], list[dict[str, Any]], str | None, bool]:
    """Perform hybrid search and return results with chunks for synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.
    Optionally applies reranking with Voyage rerank-2.5 for improved relevance.

    Args:
        req: Search request with query and parameters.

    Returns:
        Tuple of (search_results, chunks_for_synthesis, graph_context, reranked)
    """
    settings = get_settings()
    metrics = get_metrics_collector()
    metrics.record_query()
    rerank_enabled = settings.rerank_enabled
    reranker = get_reranker()

    with measure_total_search(metadata={"source": req.source.value, "limit": req.limit}):
        vs = get_vectorstore()

        # Generate dense and sparse embeddings in parallel for better latency
        dense_vector, sparse_vector = await generate_embeddings_parallel(req.query)

        # Over-fetch when reranking is enabled (3x the limit, min 15)
        fetch_limit = max(req.limit * 3, 15) if rerank_enabled else req.limit

        results: list[SearchResult] = []
        chunks_for_synthesis: list[dict[str, Any]] = []

        if req.source == SearchSource.DOCS:
            # Search only documentation with hybrid RRF
            with measure_vector_search(metadata={"collection": "docs", "limit": fetch_limit}):
                docs = await vs.search_docs_hybrid(dense_vector, sparse_vector, limit=fetch_limit)
            for doc in docs:
                results.append(
                    SearchResult(
                        score=doc.score,
                        source="documentation",
                        title=doc.payload.get("title", "Untitled"),
                        url=doc.payload.get("url"),
                        snippet=truncate(doc.payload.get("text", ""), 300),
                        metadata=doc.payload,
                    )
                )
                chunks_for_synthesis.append({
                    "text": doc.payload.get("text", ""),
                    "source_type": "documentation",
                    "url": doc.payload.get("url"),
                    "title": doc.payload.get("title", "Untitled"),
                })

        elif req.source == SearchSource.GMAIL:
            # Search only Gmail cases with hybrid RRF
            with measure_vector_search(metadata={"collection": "gmail", "limit": fetch_limit}):
                cases = await vs.search_cases_hybrid(dense_vector, sparse_vector, limit=fetch_limit)
            for case in cases:
                results.append(
                    SearchResult(
                        score=case.score,
                        source="gmail_case",
                        title=case.payload.get("subject", "No Subject"),
                        subject=case.payload.get("subject"),
                        snippet=truncate(case.payload.get("text", ""), 300),
                        metadata=case.payload,
                    )
                )
                chunks_for_synthesis.append({
                    "text": case.payload.get("text", ""),
                    "source_type": "gmail_case",
                    "subject": case.payload.get("subject", "Support Case"),
                })

        else:
            # Hybrid RRF search across all collections (docs, cases, custom_docs)
            with measure_vector_search(metadata={"collection": "all", "limit": fetch_limit}):
                hybrid = await vs.hybrid_search_rrf(dense_vector, sparse_vector, limit=fetch_limit)
            for item in hybrid:
                source_type = item.payload.get("_source", "unknown")
                if source_type == "documentation":
                    results.append(
                        SearchResult(
                            score=item.score,
                            source="documentation",
                            title=item.payload.get("title", "Untitled"),
                            url=item.payload.get("url"),
                            snippet=truncate(item.payload.get("text", ""), 300),
                            metadata=item.payload,
                        )
                    )
                    chunks_for_synthesis.append({
                        "text": item.payload.get("text", ""),
                        "source_type": "documentation",
                        "url": item.payload.get("url"),
                        "title": item.payload.get("title", "Untitled"),
                    })
                elif source_type == "custom_docs":
                    results.append(
                        SearchResult(
                            score=item.score,
                            source="custom_docs",
                            title=item.payload.get("title", "Custom Knowledge"),
                            url=item.payload.get("url_reference"),
                            snippet=truncate(item.payload.get("text", ""), 300),
                            metadata=item.payload,
                        )
                    )
                    chunks_for_synthesis.append({
                        "text": item.payload.get("text", ""),
                        "source_type": "custom_docs",
                        "url": item.payload.get("url_reference"),
                        "title": item.payload.get("title", "Custom Knowledge"),
                    })
                else:
                    results.append(
                        SearchResult(
                            score=item.score,
                            source="gmail_case",
                            title=item.payload.get("subject", "No Subject"),
                            subject=item.payload.get("subject"),
                            snippet=truncate(item.payload.get("text", ""), 300),
                            metadata=item.payload,
                        )
                    )
                    chunks_for_synthesis.append({
                        "text": item.payload.get("text", ""),
                        "source_type": "gmail_case",
                        "subject": item.payload.get("subject", "Support Case"),
                    })

        # Apply dynamic threshold filtering based on query characteristics
        results, chunks_for_synthesis = filter_by_dynamic_threshold(
            results, chunks_for_synthesis, req.query
        )

        # Apply reranking if enabled and we have results
        was_reranked = False
        if rerank_enabled and results and chunks_for_synthesis:
            try:
                # Extract texts for reranking
                texts_to_rerank = [c.get("text", "") for c in chunks_for_synthesis]

                # Rerank using Voyage AI
                rerank_response = reranker.rerank(
                    query=req.query,
                    documents=texts_to_rerank,
                    top_k=req.limit,
                )

                # Reorder results and chunks based on rerank scores
                reranked_results: list[SearchResult] = []
                reranked_chunks: list[dict[str, Any]] = []
                for rr in rerank_response.results:
                    idx = rr.index
                    if idx < len(results):
                        # Update score to reranker score
                        orig = results[idx]
                        reranked_results.append(SearchResult(
                            score=rr.relevance_score,
                            source=orig.source,
                            title=orig.title,
                            url=orig.url,
                            subject=orig.subject,
                            snippet=orig.snippet,
                            metadata=orig.metadata,
                        ))
                        reranked_chunks.append(chunks_for_synthesis[idx])

                results = reranked_results
                chunks_for_synthesis = reranked_chunks
                was_reranked = True
                logger.debug(
                    "reranking_applied",
                    query_len=len(req.query),
                    original_count=len(texts_to_rerank),
                    reranked_count=len(results),
                )
            except Exception as e:
                logger.warning("reranking_failed", error=str(e))
                # Fall back to original results, trim to limit
                results = results[:req.limit]
                chunks_for_synthesis = chunks_for_synthesis[:req.limit]
        else:
            # No reranking, just trim to limit
            results = results[:req.limit]
            chunks_for_synthesis = chunks_for_synthesis[:req.limit]

        # Get graph enrichment if available
        graph_context = None
        try:
            enrichment = await get_graph_enrichment()
            if enrichment and results:
                # Extract chunk IDs for graph traversal
                chunk_ids: list[str] = []
                for result in results[:5]:  # Top 5 chunks
                    if hasattr(result, "metadata") and result.metadata:
                        chunk_id = result.metadata.get("id") or result.metadata.get("chunk_id")
                        if chunk_id:
                            chunk_ids.append(str(chunk_id))

                if chunk_ids:
                    enrichment_ctx = await enrichment.enrich_from_chunks(chunk_ids)
                    graph_context = enrichment_ctx.to_context_string()
                    if graph_context:
                        logger.debug("graph_enrichment_added", context_len=len(graph_context))
        except Exception as e:
            logger.debug("graph_enrichment_skipped", error=str(e))

        return results, chunks_for_synthesis, graph_context, was_reranked
