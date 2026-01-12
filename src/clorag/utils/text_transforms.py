"""Text transformation utilities for normalizing product names."""

import re

# Product name transformations (order matters - more specific patterns first)
TEXT_TRANSFORMATIONS: list[tuple[re.Pattern[str], str]] = [
    # RIO-Live -> RIO +LAN (must be before RIO replacement)
    (re.compile(r"\bRIO-Live\b"), "RIO +LAN"),
    (re.compile(r"\bRIO Live\b"), "RIO +LAN"),
    (re.compile(r"\bRIOLive\b"), "RIO +LAN"),
    # Standalone RIO -> RIO +WAN (but not RIO +LAN which was already transformed)
    (re.compile(r"\bRIO\b(?!\s*\+)"), "RIO +WAN"),
]


def apply_product_name_transforms(text: str) -> str:
    """Apply product name transformations to text.

    Transforms:
    - RIO-Live / RIO Live / RIOLive -> RIO +LAN
    - RIO (standalone) -> RIO +WAN

    Args:
        text: Input text to transform.

    Returns:
        Transformed text with product name replacements.
    """
    for pattern, replacement in TEXT_TRANSFORMATIONS:
        text = pattern.sub(replacement, text)
    return text
