"""Search utility functions for context building and filtering.

This module provides utility functions for the search pipeline including
dynamic threshold computation, result filtering, context building, and
source link extraction.
"""

from typing import Any

from clorag.core.retriever import calculate_dynamic_threshold
from clorag.web.schemas import SearchResult, SourceLink


def truncate(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def filter_by_dynamic_threshold(
    results: list[SearchResult],
    chunks: list[dict[str, Any]],
    query: str,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """Filter results using dynamic threshold based on query characteristics.

    Uses reranker scores (calibrated 0-1) when available, otherwise falls back
    to the shared dynamic threshold from core.retriever.

    Args:
        results: Search results list (scores should be reranker scores post-reranking).
        chunks: Corresponding chunks for synthesis.
        query: Original search query.

    Returns:
        Filtered (results, chunks) tuple.
    """
    if not results:
        return results, chunks

    threshold = calculate_dynamic_threshold(query)

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


def _get_synthesis_defaults() -> tuple[int, int, int]:
    """Get synthesis defaults from settings with fallback."""
    try:
        from clorag.services.settings_manager import get_setting
        return (
            int(get_setting("synthesis.max_chunks")),
            int(get_setting("synthesis.context_total_budget")),
            int(get_setting("synthesis.context_group_budget")),
        )
    except (KeyError, ImportError, Exception):
        return 8, 12000, 4000


def build_context(
    chunks: list[dict[str, Any]],
    max_chunks: int | None = None,
    graph_context: str | None = None,
    max_total_chars: int | None = None,
    max_group_chars: int | None = None,
) -> str:
    """Build context string from chunks for Claude synthesis.

    Groups chunks from the same source together so Claude sees coherent
    blocks instead of interleaved fragments from different pages.

    Args:
        chunks: Retrieved document chunks (with optional 'score' field).
        max_chunks: Maximum chunks to include.
        graph_context: Optional graph enrichment context string.
        max_total_chars: Maximum total characters across all groups.
        max_group_chars: Maximum characters per source group.
    """
    # Apply defaults from settings if not explicitly provided
    defaults = _get_synthesis_defaults()
    if max_chunks is None:
        max_chunks = defaults[0]
    if max_total_chars is None:
        max_total_chars = defaults[1]
    if max_group_chars is None:
        max_group_chars = defaults[2]

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
    total_chars = 0
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

        # Merge all chunks from this group first, then truncate the group
        # to the group budget (instead of truncating each chunk independently)
        combined_text = "\n\n".join(c.get("text", "") for c in group)
        if len(combined_text) > max_group_chars:
            combined_text = combined_text[:max_group_chars] + "..."

        group_content = f"{header}\n{combined_text}"

        # Check total budget
        if total_chars + len(group_content) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 200:  # Only add if we have meaningful space left
                group_content = group_content[:remaining] + "..."
                parts.append(group_content)
            break

        parts.append(group_content)
        total_chars += len(group_content)
        idx += 1

    return "\n---\n".join(parts)


def extract_source_links(
    chunks: list[dict[str, Any]],
    max_links: int = 3,
    as_model: bool = False,
    rewrite_urls: bool = True,
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
            if url and rewrite_urls:
                url = url.replace("support.cyanview.com", "support.cyanview.cloud")
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
