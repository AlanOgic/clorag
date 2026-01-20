"""Token counting utilities for chunk sizing.

Uses tiktoken's cl100k_base encoding which is compatible with Claude and GPT-4.
"""

import tiktoken

# Lazy-loaded encoder singleton
_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get the tiktoken encoder (lazy-loaded singleton).

    Returns:
        tiktoken Encoding instance using cl100k_base.
    """
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding.

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens in the text.
    """
    if not text:
        return 0
    return len(get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        Truncated text with at most max_tokens tokens.
    """
    if not text or max_tokens <= 0:
        return ""
    encoder = get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def tokens_to_chars_estimate(tokens: int) -> int:
    """Estimate character count from token count.

    Uses ~4 chars/token as a rough estimate (English text average).
    Useful for backward compatibility with character-based limits.

    Args:
        tokens: Number of tokens.

    Returns:
        Estimated character count.
    """
    return tokens * 4


def chars_to_tokens_estimate(chars: int) -> int:
    """Estimate token count from character count.

    Uses ~4 chars/token as a rough estimate (English text average).
    Useful for converting legacy character-based settings.

    Args:
        chars: Number of characters.

    Returns:
        Estimated token count.
    """
    return chars // 4
