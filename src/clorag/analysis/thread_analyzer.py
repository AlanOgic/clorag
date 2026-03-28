"""Thread analyzer using Claude Sonnet for parallel analysis."""

import json

import anthropic
import structlog

from clorag.config import get_settings
from clorag.models import ThreadAnalysis
from clorag.services.prompt_manager import get_prompt

logger = structlog.get_logger(__name__)


class ThreadAnalyzer:
    """Analyzes Gmail threads using Claude Sonnet for parallel processing.

    Uses Sonnet for high-quality analysis of email threads
    to determine if they represent resolved support cases.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the thread analyzer.

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

    async def analyze_thread(
        self,
        thread_id: str,
        thread_content: str,
    ) -> ThreadAnalysis | None:
        """Analyze a single thread.

        Args:
            thread_id: Unique identifier for the thread.
            thread_content: Full text content of the email thread.

        Returns:
            ThreadAnalysis if successful, None if analysis failed.
        """
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": get_prompt(
                            "analysis.thread_analyzer",
                            thread_content=thread_content,
                            product_reference=get_prompt("base.product_reference"),
                        ),
                    }
                ],
            )

            # Extract text content
            content = response.content[0].text if response.content else ""

            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    data = json.loads(json_str)
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    data = json.loads(json_str)
                else:
                    logger.error(
                        "Failed to parse JSON response",
                        thread_id=thread_id,
                        content=content[:200],
                    )
                    return None

            return ThreadAnalysis(
                thread_id=thread_id,
                is_resolved=data.get("is_resolved", False),
                confidence=data.get("confidence", 0.0),
                problem_summary=data.get("problem_summary", ""),
                solution_summary=data.get("solution_summary", ""),
                keywords=data.get("keywords", []),
                category=data.get("category", "Other"),
                product=data.get("product"),
                resolution_quality=data.get("resolution_quality"),
                is_cyanview_response=data.get("is_cyanview_response", False),
                reasoning=data.get("reasoning", ""),
                anonymized_subject=data.get("anonymized_subject", ""),
            )

        except anthropic.APIError as e:
            logger.error("Anthropic API error", thread_id=thread_id, error=str(e))
            return None
        except Exception as e:
            logger.error("Analysis failed", thread_id=thread_id, error=str(e))
            return None

    async def analyze_threads_batch(
        self,
        threads: list[tuple[str, str]],
    ) -> list[ThreadAnalysis]:
        """Analyze multiple threads in parallel.

        Args:
            threads: List of (thread_id, thread_content) tuples.

        Returns:
            List of successful ThreadAnalysis results.
        """
        import anyio

        results: list[ThreadAnalysis] = []
        semaphore = anyio.Semaphore(self._max_concurrent)

        async def analyze_with_limit(thread_id: str, content: str) -> ThreadAnalysis | None:
            async with semaphore:
                return await self.analyze_thread(thread_id, content)

        async with anyio.create_task_group() as tg:
            pending_results: list[ThreadAnalysis | None] = [None] * len(threads)

            async def run_analysis(idx: int, thread_id: str, content: str) -> None:
                result = await analyze_with_limit(thread_id, content)
                pending_results[idx] = result

            for idx, (thread_id, content) in enumerate(threads):
                tg.start_soon(run_analysis, idx, thread_id, content)

        # Filter out None results
        for result in pending_results:
            if result is not None:
                results.append(result)

        resolved_count = sum(1 for r in results if r.is_resolved)
        logger.info(
            "Batch analysis complete",
            total=len(threads),
            successful=len(results),
            resolved=resolved_count,
        )

        return results
