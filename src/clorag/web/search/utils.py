"""Search utility functions for context building and filtering.

This module provides utility functions for the search pipeline including
dynamic threshold computation, result filtering, context building, and
source link extraction.
"""

from typing import Any

from clorag.web.schemas import SearchResult, SourceLink


def truncate(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def compute_dynamic_threshold(query: str, results: list[SearchResult]) -> float:
    """Compute a dynamic score threshold based on query and result characteristics.

    Short/vague queries get lower thresholds to avoid over-filtering.
    Specific queries (with model numbers, technical terms) get higher thresholds.

    Args:
        query: The search query.
        results: List of search results with scores.

    Returns:
        Dynamic threshold value (0.0-1.0 for RRF scores).
    """
    if not results:
        return 0.0

    # Base threshold varies by query length (short queries = lower threshold)
    query_words = len(query.split())
    if query_words <= 2:
        base_threshold = 0.15  # Very short queries - be permissive
    elif query_words <= 5:
        base_threshold = 0.20  # Medium queries
    else:
        base_threshold = 0.25  # Longer, more specific queries

    # Check for specific technical terms that indicate precise intent
    technical_indicators = [
        "rio", "rcp", "ci0", "vp4", "firmware", "ip", "port", "error", "protocol"
    ]
    has_technical = any(term in query.lower() for term in technical_indicators)
    if has_technical:
        base_threshold += 0.05

    # Compute score distribution to set adaptive cutoff
    scores = [r.score for r in results]
    if len(scores) >= 3:
        mean_score = sum(scores) / len(scores)
        # Don't filter below mean if most results are relevant
        threshold = min(base_threshold, mean_score * 0.6)
    else:
        threshold = base_threshold

    return threshold


def filter_by_dynamic_threshold(
    results: list[SearchResult],
    chunks: list[dict[str, Any]],
    query: str,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """Filter results using dynamic threshold based on query characteristics.

    Args:
        results: Search results list.
        chunks: Corresponding chunks for synthesis.
        query: Original search query.

    Returns:
        Filtered (results, chunks) tuple.
    """
    if not results:
        return results, chunks

    threshold = compute_dynamic_threshold(query, results)

    filtered_results: list[SearchResult] = []
    filtered_chunks: list[dict[str, Any]] = []
    for result, chunk in zip(results, chunks):
        if result.score >= threshold:
            filtered_results.append(result)
            filtered_chunks.append(chunk)

    # Always return at least 1 result even if below threshold
    if not filtered_results and results:
        return list(results[:1]), list(chunks[:1])

    return filtered_results, filtered_chunks


def _group_key(chunk: dict[str, Any]) -> str:
    """Get grouping key for a chunk (same page/thread/doc grouped together)."""
    source_type = chunk.get("source_type")
    if source_type == "documentation":
        return chunk.get("url") or chunk.get("title", "unknown")
    elif source_type == "custom_docs":
        return chunk.get("url") or chunk.get("title", "unknown")
    else:
        return chunk.get("subject") or "Support Case"


def build_context(
    chunks: list[dict[str, Any]],
    max_chunks: int = 8,
    graph_context: str | None = None,
) -> str:
    """Build context string from chunks for Claude synthesis.

    Groups chunks from the same source together so Claude sees coherent
    blocks instead of interleaved fragments from different pages.

    Args:
        chunks: Retrieved document chunks (with optional 'score' field).
        max_chunks: Maximum chunks to include.
        graph_context: Optional graph enrichment context string.
    """
    parts: list[str] = []

    # Add graph context first if available
    if graph_context:
        parts.append(f"[Knowledge Graph Relationships]\n{graph_context}")

    # Group chunks by source (same page/thread together)
    groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []  # Preserve order of first appearance
    for chunk in chunks[:max_chunks]:
        key = _group_key(chunk)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(chunk)

    idx = 1
    for key in group_order:
        group = groups[key]
        # Use the best score in the group as the group relevance
        best_score = max(c.get("score", 0) for c in group)
        source_type = group[0].get("source_type")

        if source_type == "documentation":
            url = group[0].get("url", "")
            header = f"[Source {idx}: Doc — {url}] (relevance: {best_score:.2f})"
        elif source_type == "custom_docs":
            url = group[0].get("url") or "Custom Knowledge"
            header = f"[Source {idx}: Knowledge — {url}] (relevance: {best_score:.2f})"
        else:
            header = (
                f"[Source {idx}: Case — {group[0].get('subject', 'Support')}]"
                f" (relevance: {best_score:.2f})"
            )

        # Combine all chunks from this source
        combined_text = "\n\n".join(c.get("text", "")[:2000] for c in group)
        parts.append(f"{header}\n{combined_text}")
        idx += 1

    return "\n---\n".join(parts)


def extract_source_links(
    chunks: list[dict[str, Any]],
    max_links: int = 3,
    as_model: bool = False,
) -> list[SourceLink] | list[dict[str, Any]]:
    """Extract unique source links from chunks.

    Args:
        chunks: Document chunks with source information.
        max_links: Maximum number of links to extract.
        as_model: If True, return SourceLink models; otherwise dicts.

    Returns:
        List of source links (as SourceLink models or dicts).
    """
    seen: set[str] = set()
    links: list[SourceLink] | list[dict[str, Any]] = []

    for chunk in chunks:
        if len(links) >= max_links:
            break

        if chunk.get("source_type") == "documentation":
            url = chunk.get("url")
            if url and url not in seen:
                seen.add(url)
                link_data: dict[str, Any] = {
                    "title": chunk.get("title", "Documentation"),
                    "url": url,
                    "source_type": "documentation",
                }
                if as_model:
                    links.append(SourceLink(**link_data))  # type: ignore[arg-type]
                else:
                    links.append(link_data)  # type: ignore[arg-type]
        else:
            subject = chunk.get("subject", "Support Case")
            key = f"case:{subject}"
            if key not in seen:
                seen.add(key)
                link_data = {
                    "title": subject,
                    "url": None,
                    "source_type": "gmail_case",
                }
                if as_model:
                    links.append(SourceLink(**link_data))  # type: ignore[arg-type]
                else:
                    links.append(link_data)  # type: ignore[arg-type]

    return links
