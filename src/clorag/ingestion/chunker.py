"""Text chunking strategies for document processing."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with position information."""

    text: str
    start_index: int
    end_index: int
    chunk_index: int


class TextChunker:
    """Chunk text into smaller pieces for embedding.

    Since we use voyage-context-3 which captures document context,
    we focus on clean chunk boundaries rather than heavy overlap.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separator: str = "\n\n",
    ) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
            separator: Preferred split point (paragraphs by default).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        # First, try to split by separator (paragraphs)
        paragraphs = text.split(self.separator)

        chunks: list[Chunk] = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + len(self.separator) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            start_index=current_start,
                            end_index=current_start + len(current_chunk),
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + self.separator + para if overlap_text else para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += self.separator + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                )
            )

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk.

        Args:
            text: Text to extract overlap from.

        Returns:
            Overlap text (last N characters, preferably at sentence boundary).
        """
        if len(text) <= self.chunk_overlap:
            return text

        overlap = text[-self.chunk_overlap :]

        # Try to find a sentence boundary
        sentence_end = overlap.rfind(". ")
        if sentence_end > len(overlap) // 2:
            return overlap[sentence_end + 2 :]

        return overlap
