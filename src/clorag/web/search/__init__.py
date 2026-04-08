"""Search module for the web application.

This module provides search functionality including the search pipeline,
answer synthesis, and utility functions.
"""

from clorag.core.retriever import calculate_dynamic_threshold
from clorag.web.search.pipeline import (
    generate_embeddings_parallel,
    get_embeddings,
    get_graph_enrichment,
    get_reranker,
    get_sparse_embeddings,
    get_vectorstore,
    perform_search,
)
from clorag.web.search.synthesis import (
    get_anthropic,
    synthesize_answer,
    synthesize_answer_stream,
)
from clorag.web.search.utils import (
    build_context,
    extract_source_links,
    filter_by_dynamic_threshold,
    truncate,
)

__all__ = [
    # Pipeline
    "generate_embeddings_parallel",
    "get_embeddings",
    "get_graph_enrichment",
    "get_reranker",
    "get_sparse_embeddings",
    "get_vectorstore",
    "perform_search",
    # Synthesis
    "get_anthropic",
    "synthesize_answer",
    "synthesize_answer_stream",
    # Utils
    "build_context",
    "calculate_dynamic_threshold",
    "extract_source_links",
    "filter_by_dynamic_threshold",
    "truncate",
]
