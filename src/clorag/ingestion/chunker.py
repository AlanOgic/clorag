"""Text chunking strategies for document processing."""

import re
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Content type for adaptive chunking strategies."""

    GENERIC = "generic"
    DOCUMENTATION = "documentation"
    SUPPORT_CASE = "support_case"
    FAQ = "faq"
    RELEASE_NOTES = "release_notes"


@dataclass
class Chunk:
    """A chunk of text with position information."""

    text: str
    start_index: int
    end_index: int
    chunk_index: int
    metadata: dict[str, str | int | bool] = field(default_factory=dict)


@dataclass
class AtomicBlock:
    """A block of content that should not be split."""

    text: str
    block_type: str  # "code", "table", "list"
    start_index: int
    end_index: int
    language: str | None = None  # For code blocks


class SemanticChunker:
    """Semantic-aware chunker that preserves code blocks, tables, and markdown structure.

    Features:
    - Code block preservation (``` ... ```)
    - Table preservation (| ... | format)
    - Markdown heading-based sectioning
    - Adaptive sizing for short content
    - Section-aware chunking for support cases
    """

    # Regex patterns for atomic blocks
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    TABLE_PATTERN = re.compile(r"(\|[^\n]+\|\n)+", re.MULTILINE)
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    LIST_PATTERN = re.compile(
        r"((?:^[\s]*[-*+]|\d+\.)\s+.+(?:\n(?:^[\s]*[-*+]|\d+\.)\s+.+)*)", re.MULTILINE
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        adaptive_threshold: int = 800,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
        preserve_lists: bool = True,
        respect_headings: bool = True,
    ) -> None:
        """Initialize semantic chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
            min_chunk_size: Minimum chunk size (won't split below this).
            adaptive_threshold: Content below this size stays as single chunk.
            preserve_code_blocks: Keep code blocks (```) as atomic units.
            preserve_tables: Keep markdown tables as atomic units.
            preserve_lists: Keep bullet/numbered lists together when possible.
            respect_headings: Use markdown headings as chunk boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.adaptive_threshold = adaptive_threshold
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.respect_headings = respect_headings

    def chunk_text(
        self,
        text: str,
        content_type: ContentType = ContentType.GENERIC,
    ) -> list[Chunk]:
        """Split text into semantic chunks.

        Args:
            text: Text to chunk.
            content_type: Type of content for adaptive strategies.

        Returns:
            List of Chunk objects with metadata.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Adaptive sizing: short content stays as single chunk
        if len(text) <= self.adaptive_threshold:
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    chunk_index=0,
                    metadata={"is_complete": True},
                )
            ]

        # For support cases, use section-aware chunking
        if content_type == ContentType.SUPPORT_CASE:
            return self._chunk_support_case(text)

        # Extract and protect atomic blocks
        protected_text, atomic_blocks = self._extract_atomic_blocks(text)

        # If we have headings and should respect them, use heading-based chunking
        if self.respect_headings and self.HEADING_PATTERN.search(text):
            return self._chunk_by_headings(text, atomic_blocks)

        # Fall back to paragraph-based chunking with atomic block awareness
        return self._chunk_with_atomic_blocks(text, atomic_blocks)

    def _extract_atomic_blocks(self, text: str) -> tuple[str, list[AtomicBlock]]:
        """Extract code blocks, tables, and lists that shouldn't be split.

        Args:
            text: Original text.

        Returns:
            Tuple of (text with placeholders, list of AtomicBlock).
        """
        blocks: list[AtomicBlock] = []
        protected_text = text

        # Extract code blocks
        if self.preserve_code_blocks:
            for match in self.CODE_BLOCK_PATTERN.finditer(text):
                language = match.group(1) or None
                blocks.append(
                    AtomicBlock(
                        text=match.group(0),
                        block_type="code",
                        start_index=match.start(),
                        end_index=match.end(),
                        language=language,
                    )
                )

        # Extract tables
        if self.preserve_tables:
            for match in self.TABLE_PATTERN.finditer(text):
                # Only consider it a table if it has at least a header separator row
                table_text = match.group(0)
                if "|---" in table_text or "| ---" in table_text or "|:--" in table_text:
                    blocks.append(
                        AtomicBlock(
                            text=table_text,
                            block_type="table",
                            start_index=match.start(),
                            end_index=match.end(),
                        )
                    )

        return protected_text, blocks

    def _chunk_support_case(self, text: str) -> list[Chunk]:
        """Chunk support case by markdown sections (## Problem, ## Solution, etc.).

        Args:
            text: Support case document text.

        Returns:
            List of chunks, one per section.
        """
        # Parse markdown sections
        sections = self._parse_markdown_sections(text)

        if not sections:
            # No sections found, fall back to standard chunking
            return self._chunk_paragraphs(text)

        chunks: list[Chunk] = []
        current_pos = 0

        for section_name, section_content in sections.items():
            section_text = f"## {section_name}\n{section_content}".strip()

            # If section is too large, sub-chunk it
            if len(section_text) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_paragraphs(section_text)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["section"] = section_name
                    sub_chunk.chunk_index = len(chunks)
                    chunks.append(sub_chunk)
            else:
                chunks.append(
                    Chunk(
                        text=section_text,
                        start_index=current_pos,
                        end_index=current_pos + len(section_text),
                        chunk_index=len(chunks),
                        metadata={"section": section_name},
                    )
                )

            current_pos += len(section_text) + 2  # Account for newlines

        return chunks

    def _parse_markdown_sections(self, text: str) -> dict[str, str]:
        """Parse markdown into sections based on ## headings.

        Args:
            text: Markdown text.

        Returns:
            Dict mapping section names to content.
        """
        sections: dict[str, str] = {}

        # Split by ## headings (level 2)
        pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            return sections

        for i, match in enumerate(matches):
            section_name = match.group(1).strip()
            start = match.end()

            # Find end (next ## heading or end of text)
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()
            if content:
                sections[section_name] = content

        return sections

    def _chunk_by_headings(
        self,
        text: str,
        atomic_blocks: list[AtomicBlock],
    ) -> list[Chunk]:
        """Chunk text based on markdown headings.

        Args:
            text: Text to chunk.
            atomic_blocks: Atomic blocks to preserve.

        Returns:
            List of chunks respecting heading boundaries.
        """
        chunks: list[Chunk] = []
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            return self._chunk_paragraphs(text)

        # Process content before first heading
        if matches[0].start() > 0:
            intro = text[: matches[0].start()].strip()
            if intro:
                intro_chunks = self._chunk_paragraphs(intro)
                for chunk in intro_chunks:
                    chunk.metadata["section"] = "Introduction"
                    chunk.chunk_index = len(chunks)
                    chunks.append(chunk)

        # Process each heading section
        for i, match in enumerate(matches):
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Find section end
            start = match.end()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            section_content = text[start:end].strip()
            if not section_content:
                continue

            # Include heading in chunk
            full_section = f"{'#' * heading_level} {heading_text}\n\n{section_content}"

            # Check if section contains atomic blocks
            section_has_code = any(
                b.start_index >= match.start() and b.end_index <= end
                for b in atomic_blocks
                if b.block_type == "code"
            )

            # If section is small enough, keep as single chunk
            if len(full_section) <= self.chunk_size * 1.5:
                chunks.append(
                    Chunk(
                        text=full_section,
                        start_index=match.start(),
                        end_index=end,
                        chunk_index=len(chunks),
                        metadata={
                            "section": heading_text,
                            "heading_level": heading_level,
                            "has_code": section_has_code,
                        },
                    )
                )
            else:
                # Split large section while preserving atomic blocks
                section_blocks = [
                    b for b in atomic_blocks
                    if b.start_index >= match.start() and b.end_index <= end
                ]
                sub_chunks = self._chunk_with_atomic_blocks(full_section, section_blocks)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["section"] = heading_text
                    sub_chunk.metadata["heading_level"] = heading_level
                    sub_chunk.chunk_index = len(chunks)
                    chunks.append(sub_chunk)

        return chunks

    def _chunk_with_atomic_blocks(
        self,
        text: str,
        atomic_blocks: list[AtomicBlock],
    ) -> list[Chunk]:
        """Chunk text while preserving atomic blocks.

        Args:
            text: Text to chunk.
            atomic_blocks: Blocks that must stay intact.

        Returns:
            List of chunks with atomic blocks preserved.
        """
        if not atomic_blocks:
            return self._chunk_paragraphs(text)

        chunks: list[Chunk] = []
        current_pos = 0

        # Sort blocks by position
        sorted_blocks = sorted(atomic_blocks, key=lambda b: b.start_index)

        for block in sorted_blocks:
            # Chunk text before this block
            if block.start_index > current_pos:
                pre_text = text[current_pos : block.start_index].strip()
                if pre_text:
                    pre_chunks = self._chunk_paragraphs(pre_text)
                    for chunk in pre_chunks:
                        chunk.chunk_index = len(chunks)
                        chunks.append(chunk)

            # Add the atomic block as its own chunk (or merge if small)
            block_text = block.text.strip()
            if block_text:
                metadata: dict[str, str | int | bool] = {
                    "is_atomic": True,
                    "block_type": block.block_type,
                }
                if block.language:
                    metadata["code_language"] = block.language

                chunks.append(
                    Chunk(
                        text=block_text,
                        start_index=block.start_index,
                        end_index=block.end_index,
                        chunk_index=len(chunks),
                        metadata=metadata,
                    )
                )

            current_pos = block.end_index

        # Chunk remaining text after last block
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                post_chunks = self._chunk_paragraphs(remaining)
                for chunk in post_chunks:
                    chunk.chunk_index = len(chunks)
                    chunks.append(chunk)

        return chunks

    def _chunk_paragraphs(self, text: str) -> list[Chunk]:
        """Fall back to paragraph-based chunking.

        Args:
            text: Text to chunk.

        Returns:
            List of chunks split by paragraphs.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Short text: single chunk
        if len(text) <= self.adaptive_threshold:
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    chunk_index=0,
                )
            ]

        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            start_index=current_start,
                            end_index=current_start + len(current_chunk),
                            chunk_index=len(chunks),
                        )
                    )
                    # Start new chunk with overlap
                    overlap = self._get_overlap_text(current_chunk)
                    current_start = current_start + len(current_chunk) - len(overlap)
                    current_chunk = overlap + "\n\n" + para if overlap else para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_index=len(chunks),
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
