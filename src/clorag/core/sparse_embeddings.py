"""BM25 sparse embeddings using FastEmbed for hybrid search."""

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector

from clorag.utils.logger import get_logger

logger = get_logger(__name__)


class SparseEmbeddingsClient:
    """Generate BM25 sparse vectors for hybrid search.

    Uses FastEmbed's Qdrant/bm25 model to create sparse vectors
    that complement voyage-context-3 dense vectors for hybrid retrieval.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Initialize the sparse embeddings client.

        Args:
            model_name: FastEmbed model name. Defaults to Qdrant/bm25.
        """
        logger.info("Loading sparse embedding model", model=model_name)
        self._model = SparseTextEmbedding(model_name=model_name)
        self._model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse vectors for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of SparseVector objects for Qdrant.
        """
        if not texts:
            return []

        embeddings = list(self._model.embed(texts))
        sparse_vectors = [
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            for emb in embeddings
        ]

        logger.debug("Generated sparse embeddings", count=len(sparse_vectors))
        return sparse_vectors

    def embed_query(self, query: str) -> SparseVector:
        """Generate sparse vector for a single query.

        Args:
            query: Query text to embed.

        Returns:
            SparseVector for the query.
        """
        result = self.embed_texts([query])
        return result[0] if result else SparseVector(indices=[], values=[])

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> list[SparseVector]:
        """Generate sparse vectors in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Size of each batch.

        Returns:
            List of SparseVector objects.
        """
        if not texts:
            return []

        all_vectors: list[SparseVector] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embed_texts(batch)
            all_vectors.extend(vectors)

            logger.debug(
                "Processed sparse embedding batch",
                batch_num=i // batch_size + 1,
                total_batches=(len(texts) + batch_size - 1) // batch_size,
            )

        return all_vectors
