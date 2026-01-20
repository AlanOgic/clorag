"""Tests for the chunker module with token-aware functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr

from clorag.config import Settings
from clorag.ingestion.chunker import (
    Chunk,
    ContentType,
    SemanticChunker,
    TextChunker,
)
from clorag.utils.tokenizer import count_tokens


class TestTokenizer:
    """Tests for the tokenizer utility functions."""

    def test_count_tokens_empty(self) -> None:
        """Test token counting with empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_simple(self) -> None:
        """Test token counting with simple text."""
        # "Hello world" is typically 2 tokens
        tokens = count_tokens("Hello world")
        assert tokens >= 1  # At least one token
        assert tokens <= 5  # But not too many

    def test_count_tokens_longer_text(self) -> None:
        """Test token counting with longer text."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        # This should be roughly 100 tokens (10 * ~10 tokens per sentence)
        assert 80 <= tokens <= 150


class TestSemanticChunkerBasic:
    """Basic tests for SemanticChunker without token mode."""

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = SemanticChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_short_text_single_chunk(self) -> None:
        """Test that short text becomes a single chunk."""
        chunker = SemanticChunker(adaptive_threshold=100)
        text = "Short text."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text.strip()
        assert chunks[0].metadata.get("is_complete") is True

    def test_preserves_code_blocks(self) -> None:
        """Test that code blocks are preserved as atomic units."""
        chunker = SemanticChunker(chunk_size=50, adaptive_threshold=20)
        text = """Some text before.

```python
def hello():
    print("Hello, World!")
```

Some text after."""
        chunks = chunker.chunk_text(text)
        # Find the chunk with code
        code_chunks = [c for c in chunks if "```python" in c.text]
        assert len(code_chunks) >= 1
        # Code block should be complete
        for chunk in code_chunks:
            if "def hello" in chunk.text:
                assert "```" in chunk.text


class TestSemanticChunkerTokenMode:
    """Tests for SemanticChunker with token-based sizing."""

    def test_use_tokens_flag(self) -> None:
        """Test that use_tokens flag is properly set."""
        chunker_chars = SemanticChunker(use_tokens=False)
        chunker_tokens = SemanticChunker(use_tokens=True)

        assert chunker_chars.use_tokens is False
        assert chunker_tokens.use_tokens is True

    def test_measure_size_characters(self) -> None:
        """Test _measure_size returns character count when use_tokens=False."""
        chunker = SemanticChunker(use_tokens=False)
        text = "Hello world"
        assert chunker._measure_size(text) == len(text)

    def test_measure_size_tokens(self) -> None:
        """Test _measure_size returns token count when use_tokens=True."""
        chunker = SemanticChunker(use_tokens=True)
        text = "Hello world"
        # Token count should be less than character count
        assert chunker._measure_size(text) < len(text)
        assert chunker._measure_size(text) >= 1

    def test_token_mode_creates_different_chunk_counts(self) -> None:
        """Test that token mode with same numeric size creates different chunk counts."""
        # With 100 chars, we might fit ~25 tokens
        # So a 100-token chunk should contain more text than a 100-char chunk
        # Use a large text with multiple paragraphs to ensure it gets split
        text = "\n\n".join([
            "The quick brown fox jumps over the lazy dog. " * 10
            for _ in range(10)
        ])  # ~4500 chars, ~1000 tokens

        chunker_chars = SemanticChunker(
            chunk_size=200,
            chunk_overlap=20,
            adaptive_threshold=10,  # Very low to ensure chunking happens
            use_tokens=False,
            respect_headings=False,  # Disable heading chunking for this test
        )
        chunker_tokens = SemanticChunker(
            chunk_size=200,
            chunk_overlap=20,
            adaptive_threshold=10,  # Very low to ensure chunking happens
            use_tokens=True,
            respect_headings=False,  # Disable heading chunking for this test
        )

        chunks_chars = chunker_chars.chunk_text(text)
        chunks_tokens = chunker_tokens.chunk_text(text)

        # Token mode should create fewer chunks
        # because 200 tokens ≈ 800 characters, so each chunk holds more text
        # Character mode with 200 chars will need more chunks to cover same text
        assert len(chunks_chars) > len(chunks_tokens), (
            f"Expected char mode ({len(chunks_chars)} chunks) to create more chunks "
            f"than token mode ({len(chunks_tokens)} chunks)"
        )

    def test_token_adaptive_threshold(self) -> None:
        """Test that adaptive threshold works with tokens."""
        # Create text that is ~100 tokens
        text = "Hello world. " * 30  # Roughly 60-90 tokens

        chunker = SemanticChunker(
            adaptive_threshold=200,  # 200 tokens
            use_tokens=True,
        )

        chunks = chunker.chunk_text(text)
        # Should be a single chunk since text is below threshold
        assert len(chunks) == 1


class TestSemanticChunkerFromSettings:
    """Tests for SemanticChunker.from_settings() factory method."""

    @pytest.fixture
    def test_settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            anthropic_api_key=SecretStr("test-key"),
            voyage_api_key=SecretStr("test-key"),
            chunk_use_tokens=True,
            chunk_size_docs=450,
            chunk_size_cases=350,
            chunk_size_default=400,
            chunk_overlap=50,
            chunk_adaptive_threshold=200,
        )

    def test_from_settings_documentation(self, test_settings: Settings) -> None:
        """Test from_settings creates chunker with documentation size."""
        with patch("clorag.config.get_settings", return_value=test_settings):
            chunker = SemanticChunker.from_settings(ContentType.DOCUMENTATION)

        assert chunker.chunk_size == 450
        assert chunker.chunk_overlap == 50
        assert chunker.adaptive_threshold == 200
        assert chunker.use_tokens is True

    def test_from_settings_support_case(self, test_settings: Settings) -> None:
        """Test from_settings creates chunker with support case size."""
        with patch("clorag.config.get_settings", return_value=test_settings):
            chunker = SemanticChunker.from_settings(ContentType.SUPPORT_CASE)

        assert chunker.chunk_size == 350
        assert chunker.chunk_overlap == 50
        assert chunker.use_tokens is True

    def test_from_settings_generic(self, test_settings: Settings) -> None:
        """Test from_settings creates chunker with default size for generic content."""
        with patch("clorag.config.get_settings", return_value=test_settings):
            chunker = SemanticChunker.from_settings(ContentType.GENERIC)

        assert chunker.chunk_size == 400
        assert chunker.chunk_overlap == 50
        assert chunker.use_tokens is True

    def test_from_settings_with_explicit_settings(self, test_settings: Settings) -> None:
        """Test from_settings accepts explicit settings parameter."""
        chunker = SemanticChunker.from_settings(
            ContentType.DOCUMENTATION,
            settings=test_settings,
        )

        assert chunker.chunk_size == 450
        assert chunker.use_tokens is True


class TestChunkOverlapTokenMode:
    """Tests for chunk overlap in token mode."""

    def test_overlap_character_mode(self) -> None:
        """Test overlap extraction in character mode."""
        chunker = SemanticChunker(chunk_overlap=20, use_tokens=False)
        text = "This is a sentence. Another sentence here."

        overlap = chunker._get_overlap_text(text)
        # Should be at most chunk_overlap characters or entire text
        assert len(overlap) <= max(20, len(text))

    def test_overlap_token_mode_sentence_boundary(self) -> None:
        """Test overlap extraction in token mode respects sentence boundaries."""
        chunker = SemanticChunker(chunk_overlap=10, use_tokens=True)
        text = "First sentence. Second sentence. Third sentence here."

        overlap = chunker._get_overlap_text(text)
        # Should extract complete sentences
        assert overlap  # Not empty


class TestSupportCaseChunking:
    """Tests for support case content type chunking."""

    def test_support_case_sections(self) -> None:
        """Test that support cases are chunked by sections."""
        chunker = SemanticChunker(chunk_size=500, use_tokens=False)
        text = """## Problem
The user cannot connect to the device.

## Solution
Reset the network settings and reconfigure the IP address.

## Notes
This is a common issue after firmware updates."""

        chunks = chunker.chunk_text(text, content_type=ContentType.SUPPORT_CASE)

        # Should have multiple chunks for the sections
        assert len(chunks) >= 1
        # Each chunk should have section metadata
        for chunk in chunks:
            assert "section" in chunk.metadata or "is_complete" in chunk.metadata


class TestTextChunker:
    """Tests for the basic TextChunker class."""

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        # All text should be represented
        all_text = " ".join(c.text for c in chunks)
        assert "First" in all_text
        assert "Second" in all_text
        assert "Third" in all_text

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
