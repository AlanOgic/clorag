"""Multi-source retriever combining documentation and Gmail cases."""

from dataclasses import dataclass
from enum import Enum

from clorag.core.embeddings import EmbeddingsClient
from clorag.core.vectorstore import SearchResult, VectorStore


class SearchSource(Enum):
    """Available search sources."""

    DOCS = "documentation"
    CASES = "gmail_cases"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    source: SearchSource
    results: list[SearchResult]
    total_found: int


class MultiSourceRetriever:
    """Retriever that combines results from documentation and Gmail cases.

    Provides intelligent query routing and result fusion across both
    knowledge sources.
    """

    def __init__(
        self,
        embeddings_client: EmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embeddings_client: Client for generating query embeddings.
            vector_store: Client for vector search.
        """
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()

    async def retrieve(
        self,
        query: str,
        source: SearchSource = SearchSource.HYBRID,
        limit: int = 5,
        score_threshold: float | None = 0.7,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            source: Which source(s) to search.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score.

        Returns:
            RetrievalResult with matched documents.
        """
        # Generate query embedding
        query_vector = await self._embeddings.embed_query(query)

        # Search based on source
        if source == SearchSource.DOCS:
            results = await self._vectorstore.search_docs(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
        elif source == SearchSource.CASES:
            results = await self._vectorstore.search_cases(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:  # HYBRID
            results = await self._vectorstore.hybrid_search(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )

        return RetrievalResult(
            query=query,
            source=source,
            results=results,
            total_found=len(results),
        )

    async def retrieve_docs(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = 0.7,
    ) -> RetrievalResult:
        """Retrieve from documentation only.

        Args:
            query: Search query.
            limit: Maximum results.
            score_threshold: Minimum score.

        Returns:
            RetrievalResult from docs collection.
        """
        return await self.retrieve(
            query=query,
            source=SearchSource.DOCS,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def retrieve_cases(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = 0.7,
    ) -> RetrievalResult:
        """Retrieve from Gmail cases only.

        Args:
            query: Search query.
            limit: Maximum results.
            score_threshold: Minimum score.

        Returns:
            RetrievalResult from cases collection.
        """
        return await self.retrieve(
            query=query,
            source=SearchSource.CASES,
            limit=limit,
            score_threshold=score_threshold,
        )

    def format_context(self, result: RetrievalResult, max_chars: int = 8000) -> str:
        """Format retrieval results as context for the agent.

        Args:
            result: Retrieval result to format.
            max_chars: Maximum characters in formatted context.

        Returns:
            Formatted context string.
        """
        if not result.results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, r in enumerate(result.results, 1):
            source_label = r.payload.get("_source", result.source.value)
            url = r.payload.get("url", "")
            title = r.payload.get("title", f"Document {i}")

            header = f"[{i}] {title}"
            if url:
                header += f" ({url})"
            header += f" [Source: {source_label}] [Score: {r.score:.2f}]"

            chunk = f"{header}\n{r.text}\n"

            if current_length + len(chunk) > max_chars:
                break

            context_parts.append(chunk)
            current_length += len(chunk)

        return "\n---\n".join(context_parts)
