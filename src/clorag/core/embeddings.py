"""Voyage AI embeddings client."""

from dataclasses import dataclass
from typing import Literal

import structlog
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from clorag.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    vectors: list[list[float]]
    total_tokens: int


class EmbeddingsClient:
    """Client for generating embeddings using Voyage AI.

    Uses voyage-context-3 for contextualized document embeddings (with contextualized_embed)
    and voyage-2 for query embeddings (with regular embed).

    voyage-context-3 is optimized for RAG applications - it encodes both chunk content
    and full document context into each embedding, improving retrieval accuracy by ~14%.
    """

    # Model for query embeddings (must match dimensions with context model)
    QUERY_MODEL = "voyage-2"  # 1024 dimensions, compatible with voyage-context-3

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize the embeddings client.

        Args:
            api_key: Voyage AI API key. Defaults to VOYAGE_API_KEY env var.
            model: Embedding model name for contextualized embeddings. Defaults to voyage-context-3.
            dimensions: Output dimensions (256, 512, 1024, 2048). Defaults to 1024.
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key.get_secret_value()
        self._model = model or settings.voyage_model
        self._dimensions = dimensions or settings.voyage_dimensions

        # Initialize Voyage AI client
        self._client = voyageai.Client(api_key=self._api_key)

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Get the output dimensions."""
        return self._dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_documents(
        self,
        texts: list[str],
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingResult:
        """Generate embeddings for documents.

        Args:
            texts: List of text chunks to embed.
            input_type: Type of input - 'document' for indexing, 'query' for search.

        Returns:
            EmbeddingResult with vectors and token count.
        """
        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0)

        # voyage-context-3 supports up to 16K chunks per request
        result = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )

        return EmbeddingResult(
            vectors=result.embeddings,
            total_tokens=result.total_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query.

        Uses voyage-context-3 with contextualized_embed() for queries,
        matching the document embedding approach for consistent retrieval.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        # Use contextualized_embed for queries too - query as [[text]]
        result = self._client.contextualized_embed(
            inputs=[[text]],
            model=self._model,
            input_type="query",
            output_dimension=self._dimensions,
        )

        # result.results[0].embeddings[0] = the query embedding
        return result.results[0].embeddings[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 30,
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingResult:
        """Generate embeddings in batches for large datasets.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch (default 30 for ~120K token limit).
            input_type: Type of input for embedding.

        Returns:
            Combined EmbeddingResult with all vectors.
        """
        all_vectors: list[list[float]] = []
        total_tokens = 0
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            logger.info(
                "Processing embedding batch",
                batch=batch_num,
                total=total_batches,
                texts=len(batch),
            )
            result = await self.embed_documents(batch, input_type=input_type)
            all_vectors.extend(result.vectors)
            total_tokens += result.total_tokens

        return EmbeddingResult(vectors=all_vectors, total_tokens=total_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_contextualized(
        self,
        documents: list[list[str]],
        input_type: Literal["document", "query"] = "document",
    ) -> list[list[list[float]]]:
        """Generate contextualized embeddings for documents with multiple chunks.

        Each document is a list of chunks. The embedding model encodes each chunk
        while understanding its context within the full document.

        Args:
            documents: List of documents, where each document is a list of chunks.
            input_type: Type of input - 'document' for indexing, 'query' for search.

        Returns:
            List of embeddings per document, where each document has a list of
            chunk embeddings.
        """
        if not documents:
            return []

        # voyage-context-3 with contextualized_embed
        result = self._client.contextualized_embed(
            inputs=documents,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )

        # Extract embeddings from results
        # result.results[doc_idx].embeddings[chunk_idx] = vector
        all_embeddings: list[list[list[float]]] = []
        for doc_result in result.results:
            all_embeddings.append(doc_result.embeddings)

        return all_embeddings

    async def embed_contextualized_batch(
        self,
        documents: list[list[str]],
        batch_size: int = 10,
        input_type: Literal["document", "query"] = "document",
    ) -> list[list[list[float]]]:
        """Generate contextualized embeddings in batches.

        Args:
            documents: List of documents, each containing a list of chunks.
            batch_size: Number of documents per batch.
            input_type: Type of input for embedding.

        Returns:
            List of embeddings per document.
        """
        all_embeddings: list[list[list[float]]] = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            batch = documents[i : i + batch_size]
            total_chunks = sum(len(doc) for doc in batch)
            logger.info(
                "Processing contextualized embedding batch",
                batch=batch_num,
                total=total_batches,
                documents=len(batch),
                chunks=total_chunks,
            )
            result = await self.embed_contextualized(batch, input_type=input_type)
            all_embeddings.extend(result)

        return all_embeddings
