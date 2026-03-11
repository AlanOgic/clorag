"""Text transformation utilities for normalizing product names."""

import re

# Product name transformations (order matters - more specific patterns first)
# Only unambiguous fixes here. Context-dependent cases (standalone "RIO") are
# handled by the Sonnet-based RIO analyzer which understands hardware vs license context.
TEXT_TRANSFORMATIONS: list[tuple[re.Pattern[str], str]] = [
    # Legacy "RIO-Live" / "RIO Live" variants -> "RIO +LAN" (always correct)
    (re.compile(r"\bRIO[-\s]?Live\b", re.IGNORECASE), "RIO +LAN"),
    # Legacy "RIO +WAN Live" variant -> "RIO +LAN" (always correct)
    (re.compile(r"\bRIO\s*\+\s*WAN\s+Live\b", re.IGNORECASE), "RIO +LAN"),
]


def apply_product_name_transforms(text: str) -> str:
    """Apply unambiguous product name transformations to text.

    Transforms (always correct regardless of context):
    - RIO-Live / RIO Live / RIOLive -> RIO +LAN
    - RIO +WAN Live -> RIO +LAN

    Standalone "RIO" is NOT transformed here — it may be correct as-is
    in hardware contexts. The Sonnet-based RIO analyzer handles
    context-dependent disambiguation when rio_fix_on_ingest is enabled.

    Args:
        text: Input text to transform.

    Returns:
        Transformed text with product name replacements.
    """
    for pattern, replacement in TEXT_TRANSFORMATIONS:
        text = pattern.sub(replacement, text)
    return text
