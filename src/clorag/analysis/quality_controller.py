"""Quality controller using Claude Sonnet for final QC and structuring."""

import json

import anthropic
import structlog

from clorag.config import get_settings
from clorag.models import QualityControlResult, ThreadAnalysis
from clorag.services.prompt_manager import get_prompt

logger = structlog.get_logger(__name__)


class QualityController:
    """Quality controller using Claude Sonnet for final review and structuring.

    Reviews Haiku's analysis and produces refined, high-quality case documentation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the quality controller.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use for QC. Defaults to settings.sonnet_model.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key.get_secret_value()
        self._model = model or settings.sonnet_model
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def review_case(
        self,
        analysis: ThreadAnalysis,
        thread_content: str,
    ) -> QualityControlResult | None:
        """Review and refine an analyzed support case.

        Args:
            analysis: Haiku's analysis of the thread.
            thread_content: Original thread content.

        Returns:
            QualityControlResult if successful, None if QC failed.
        """
        try:
            prompt = get_prompt(
                "analysis.quality_controller",
                thread_content=thread_content,
                problem_summary=analysis.problem_summary,
                solution_summary=analysis.solution_summary,
                keywords=", ".join(analysis.keywords),
                category=analysis.category,
                product=analysis.product or "N/A",
                resolution_quality=analysis.resolution_quality or "N/A",
                anonymized_subject=analysis.anonymized_subject or "N/A",
            )

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

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
                        "Failed to parse QC JSON response",
                        thread_id=analysis.thread_id,
                        content=content[:200],
                    )
                    return None

            return QualityControlResult(
                approved=data.get("approved", False),
                refined_problem=data.get("refined_problem", analysis.problem_summary),
                refined_solution=data.get("refined_solution", analysis.solution_summary),
                refined_keywords=data.get("refined_keywords", analysis.keywords),
                refined_category=data.get("refined_category", analysis.category),
                suggestions=data.get("suggestions", []),
                final_document=data.get("final_document", ""),
                anonymized_title=data.get("anonymized_title", analysis.anonymized_subject or ""),
            )

        except anthropic.APIError as e:
            logger.error("Anthropic API error in QC", thread_id=analysis.thread_id, error=str(e))
            return None
        except Exception as e:
            logger.error("QC failed", thread_id=analysis.thread_id, error=str(e))
            return None

    async def review_cases_batch(
        self,
        cases: list[tuple[ThreadAnalysis, str]],
        max_concurrent: int = 5,
    ) -> list[tuple[ThreadAnalysis, QualityControlResult]]:
        """Review multiple cases with limited concurrency.

        Args:
            cases: List of (ThreadAnalysis, thread_content) tuples.
            max_concurrent: Maximum concurrent requests (Sonnet is more expensive).

        Returns:
            List of (ThreadAnalysis, QualityControlResult) for approved cases.
        """
        import anyio

        results: list[tuple[ThreadAnalysis, QualityControlResult]] = []
        semaphore = anyio.Semaphore(max_concurrent)

        async def review_with_limit(
            analysis: ThreadAnalysis, content: str
        ) -> tuple[ThreadAnalysis, QualityControlResult | None]:
            async with semaphore:
                qc_result = await self.review_case(analysis, content)
                return (analysis, qc_result)

        async with anyio.create_task_group() as tg:
            pending: list[tuple[ThreadAnalysis, QualityControlResult | None]] = []

            async def run_review(analysis: ThreadAnalysis, content: str) -> None:
                result = await review_with_limit(analysis, content)
                pending.append(result)

            for analysis, content in cases:
                tg.start_soon(run_review, analysis, content)

        # Filter approved cases
        for analysis, qc_result in pending:
            if qc_result is not None and qc_result.approved:
                results.append((analysis, qc_result))

        logger.info(
            "QC batch complete",
            total=len(cases),
            approved=len(results),
        )

        return results
