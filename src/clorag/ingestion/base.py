"""Base classes for ingestion pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """A document to be ingested into the vector store."""

    id: str
    text: str
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        """Get the document source."""
        return str(self.metadata.get("source", "unknown"))


class BaseIngestionPipeline(ABC):
    """Abstract base class for ingestion pipelines."""

    @abstractmethod
    async def fetch(self) -> list[Document]:
        """Fetch documents from the source.

        Returns:
            List of documents to process.
        """
        pass

    @abstractmethod
    async def process(self, documents: list[Document]) -> list[Document]:
        """Process documents (chunking, cleaning, etc.).

        Args:
            documents: Raw documents to process.

        Returns:
            Processed documents ready for embedding.
        """
        pass

    @abstractmethod
    async def ingest(self, documents: list[Document]) -> int:
        """Embed and store documents in the vector store.

        Args:
            documents: Processed documents to ingest.

        Returns:
            Number of documents successfully ingested.
        """
        pass

    async def run(self) -> int:
        """Run the full ingestion pipeline.

        Returns:
            Number of documents ingested.
        """
        documents = await self.fetch()
        processed = await self.process(documents)
        return await self.ingest(processed)
