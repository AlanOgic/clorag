"""RIO terminology analyzer using Claude Sonnet for context analysis."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

import anthropic
import structlog

from clorag.config import get_settings
from clorag.core.terminology_db import TerminologyFix
from clorag.services.prompt_manager import get_prompt

logger = structlog.get_logger(__name__)

# Patterns to detect RIO mentions that need analysis
RIO_PATTERNS = [
    r"\bRIO[-\s]?Live\b",  # Legacy name: RIO-Live, RIO Live
    r"\bRIO\s*\+\s*WAN\s+Live\b",  # Invalid: RIO +WAN Live
    r"\bRIO\s*\+\s*WAN\b",  # RIO +WAN
    r"\bRIO\s*\+\s*LAN\b",  # RIO +LAN
    r"\bthe\s+RIO\b",  # "the RIO" (may need context check)
    r"\bRIOs?\b",  # General RIO/RIOs mentions
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RIO_PATTERNS]


@dataclass
class RIOAnalysisResult:
    """Result of analyzing a RIO terminology mention."""

    needs_fix: bool
    suggestion_type: str
    original_text: str
    suggested_text: str
    confidence: float
    reasoning: str


class RIOTerminologyAnalyzer:
    """Analyzes chunks for RIO terminology issues using Claude Sonnet.

    Uses Sonnet for high-quality analysis of text chunks to identify
    and suggest corrections for RIO product terminology.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the RIO analyzer.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use for analysis. Defaults to settings.sonnet_model.
            max_concurrent: Maximum concurrent requests.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key.get_secret_value()
        self._model = model or settings.sonnet_model
        self._max_concurrent = max_concurrent
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    def find_rio_mentions(self, text: str) -> list[str]:
        """Find all RIO-related mentions in text.

        Args:
            text: Text to search.

        Returns:
            List of unique matched strings.
        """
        matches: set[str] = set()
        for pattern in COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                matches.add(match.group(0))
        return list(matches)

    def has_rio_mentions(self, text: str) -> bool:
        """Check if text contains any RIO mentions worth analyzing.

        Args:
            text: Text to check.

        Returns:
            True if text contains RIO mentions.
        """
        return any(pattern.search(text) for pattern in COMPILED_PATTERNS)

    def _extract_json(
        self, content: str
    ) -> dict[str, str | float | bool] | list[dict[str, str | float | bool]] | None:
        """Extract JSON from LLM response with multiple fallback strategies.

        Args:
            content: Raw response content.

        Returns:
            Parsed JSON data or None if parsing fails.
        """
        result: dict[str, str | float | bool] | list[
            dict[str, str | float | bool]
        ] | None = None

        # Strategy 1: Direct parse
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        if "```" in content:
            try:
                json_str = content.split("```")[1].split("```")[0].strip()
                result = json.loads(json_str)
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: Find JSON object using regex
        obj_match = re.search(r'\{[^{}]*"needs_fix"[^{}]*\}', content, re.DOTALL)
        if obj_match:
            try:
                result = json.loads(obj_match.group())
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find any JSON-like structure with braces
        brace_match = re.search(r'\{[\s\S]*?\}', content)
        if brace_match:
            try:
                result = json.loads(brace_match.group())
                return result
            except json.JSONDecodeError:
                pass

        return None

    async def analyze_mention(
        self,
        chunk_text: str,
        matched_text: str,
    ) -> RIOAnalysisResult | None:
        """Analyze a single RIO mention for terminology issues.

        Args:
            chunk_text: Full text of the chunk for context.
            matched_text: The specific matched RIO text to analyze.

        Returns:
            RIOAnalysisResult if analysis successful, None if failed.
        """
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": get_prompt(
                            "analysis.rio_terminology",
                            chunk_text=chunk_text[:3000],  # Limit context size
                            matched_text=matched_text,
                            product_reference=get_prompt("base.product_reference"),
                        ),
                    }
                ],
            )

            # Extract text content
            first_block = response.content[0] if response.content else None
            content = first_block.text if first_block and hasattr(first_block, "text") else ""

            # Parse JSON response with robust extraction
            data = self._extract_json(content)
            if data is None:
                logger.warning(
                    "Failed to parse JSON response",
                    matched_text=matched_text,
                    content=content[:300],
                )
                return None

            # Handle list responses (sometimes LLM wraps in array)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            if not isinstance(data, dict):
                logger.warning(
                    "Response is not a dict",
                    matched_text=matched_text,
                    data_type=type(data).__name__,
                )
                return None

            return RIOAnalysisResult(
                needs_fix=bool(data.get("needs_fix", False)),
                suggestion_type=str(data.get("suggestion_type", "no_change")),
                original_text=str(data.get("original_text", matched_text)),
                suggested_text=str(data.get("suggested_text", matched_text)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", "")),
            )

        except anthropic.APIError as e:
            logger.error("Anthropic API error", matched_text=matched_text, error=str(e))
            return None
        except Exception as e:
            logger.error("Analysis failed", matched_text=matched_text, error=str(e))
            return None

    async def analyze_chunk(
        self,
        chunk_id: str,
        collection: str,
        chunk_text: str,
    ) -> list[TerminologyFix]:
        """Analyze a chunk for all RIO terminology issues.

        Args:
            chunk_id: UUID of the chunk.
            collection: Collection name (docusaurus_docs, gmail_cases, custom_docs).
            chunk_text: Full text content of the chunk.

        Returns:
            List of TerminologyFix suggestions for this chunk.
        """
        # Find all RIO mentions
        mentions = self.find_rio_mentions(chunk_text)
        if not mentions:
            return []

        fixes: list[TerminologyFix] = []

        for matched_text in mentions:
            result = await self.analyze_mention(chunk_text, matched_text)

            if result and result.needs_fix:
                fix = TerminologyFix(
                    id=str(uuid.uuid4()),
                    chunk_id=chunk_id,
                    collection=collection,
                    original_text=result.original_text,
                    suggested_text=result.suggested_text,
                    suggestion_type=result.suggestion_type,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    status="pending",
                    created_at=datetime.utcnow(),
                )
                fixes.append(fix)
                logger.debug(
                    "Found terminology fix",
                    chunk_id=chunk_id,
                    original=result.original_text,
                    suggested=result.suggested_text,
                    confidence=result.confidence,
                )

        return fixes

    async def analyze_chunks_batch(
        self,
        chunks: list[tuple[str, str, str]],
    ) -> list[TerminologyFix]:
        """Analyze multiple chunks in parallel.

        Args:
            chunks: List of (chunk_id, collection, chunk_text) tuples.

        Returns:
            List of all TerminologyFix suggestions found.
        """
        import anyio

        all_fixes: list[TerminologyFix] = []
        semaphore = anyio.Semaphore(self._max_concurrent)

        async def analyze_with_limit(
            chunk_id: str, collection: str, text: str
        ) -> list[TerminologyFix]:
            async with semaphore:
                return await self.analyze_chunk(chunk_id, collection, text)

        async with anyio.create_task_group() as tg:
            results: list[list[TerminologyFix]] = [[] for _ in chunks]

            async def run_analysis(
                idx: int, chunk_id: str, collection: str, text: str
            ) -> None:
                fixes = await analyze_with_limit(chunk_id, collection, text)
                results[idx] = fixes

            for idx, (chunk_id, collection, text) in enumerate(chunks):
                tg.start_soon(run_analysis, idx, chunk_id, collection, text)

        # Flatten results
        for chunk_fixes in results:
            all_fixes.extend(chunk_fixes)

        chunks_with_fixes = sum(1 for r in results if r)
        logger.info(
            "Batch analysis complete",
            total_chunks=len(chunks),
            chunks_with_fixes=chunks_with_fixes,
            total_fixes=len(all_fixes),
        )

        return all_fixes


def apply_fix_to_text(text: str, original: str, suggested: str) -> str:
    """Apply a terminology fix to text.

    Args:
        text: Original full text.
        original: Text to find and replace.
        suggested: Replacement text.

    Returns:
        Text with fix applied.
    """
    # Use case-insensitive replacement but preserve casing pattern
    pattern = re.compile(re.escape(original), re.IGNORECASE)
    return pattern.sub(suggested, text, count=1)


@dataclass
class AppliedFix:
    """Record of a fix applied during ingestion."""

    original_text: str
    suggested_text: str
    suggestion_type: str
    confidence: float
    reasoning: str


async def apply_rio_fixes_before_embedding(
    text: str,
    analyzer: RIOTerminologyAnalyzer | None = None,
    min_confidence: float = 0.85,
) -> tuple[str, list[AppliedFix]]:
    """Apply high-confidence RIO terminology fixes to text before embedding.

    This function is designed to be called during ingestion, before the text
    is chunked and embedded. It analyzes the text for RIO terminology issues
    and automatically applies fixes that meet the confidence threshold.

    Args:
        text: The text to analyze and fix.
        analyzer: RIOTerminologyAnalyzer instance (created if None).
        min_confidence: Minimum confidence threshold for auto-applying fixes.

    Returns:
        Tuple of (fixed_text, list_of_applied_fixes).
    """
    if not text or not text.strip():
        return text, []

    # Create analyzer if not provided
    if analyzer is None:
        analyzer = RIOTerminologyAnalyzer()

    # Check if text has any RIO mentions worth analyzing
    if not analyzer.has_rio_mentions(text):
        return text, []

    # Find all RIO mentions
    mentions = analyzer.find_rio_mentions(text)
    if not mentions:
        return text, []

    applied_fixes: list[AppliedFix] = []
    fixed_text = text

    # Analyze each mention and apply high-confidence fixes
    for matched_text in mentions:
        result = await analyzer.analyze_mention(fixed_text, matched_text)

        if result and result.needs_fix and result.confidence >= min_confidence:
            # Skip "needs_human_review" suggestions - these should not be auto-applied
            if result.suggestion_type == "needs_human_review":
                logger.debug(
                    "Skipping human review fix during ingestion",
                    original=result.original_text,
                    suggested=result.suggested_text,
                    confidence=result.confidence,
                )
                continue

            # Apply the fix
            new_text = apply_fix_to_text(
                fixed_text, result.original_text, result.suggested_text
            )

            # Only record if text actually changed
            if new_text != fixed_text:
                fixed_text = new_text
                applied_fixes.append(
                    AppliedFix(
                        original_text=result.original_text,
                        suggested_text=result.suggested_text,
                        suggestion_type=result.suggestion_type,
                        confidence=result.confidence,
                        reasoning=result.reasoning,
                    )
                )
                logger.info(
                    "Applied RIO fix during ingestion",
                    original=result.original_text,
                    suggested=result.suggested_text,
                    suggestion_type=result.suggestion_type,
                    confidence=result.confidence,
                )

    if applied_fixes:
        logger.info(
            "RIO fixes applied before embedding",
            total_fixes=len(applied_fixes),
            fix_types=[f.suggestion_type for f in applied_fixes],
        )

    return fixed_text, applied_fixes


def apply_rio_fixes_sync(
    text: str,
    min_confidence: float = 0.85,
) -> tuple[str, list[AppliedFix]]:
    """Synchronous wrapper for apply_rio_fixes_before_embedding.

    Convenience function for use in sync code paths.

    Args:
        text: The text to analyze and fix.
        min_confidence: Minimum confidence threshold for auto-applying fixes.

    Returns:
        Tuple of (fixed_text, list_of_applied_fixes).
    """
    import asyncio

    return asyncio.run(apply_rio_fixes_before_embedding(text, None, min_confidence))
