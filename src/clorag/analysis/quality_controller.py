"""Quality controller using Claude Sonnet for final QC and structuring."""

import json

import anthropic
import structlog

from clorag.config import get_settings
from clorag.models import QualityControlResult, ThreadAnalysis

logger = structlog.get_logger(__name__)

QC_PROMPT = """You are a quality controller for a support case documentation system.

CRITICAL ANONYMIZATION REQUIREMENTS:
- The content has placeholder tokens: [SERIAL:XXX-N], [EMAIL-N], [PHONE-N]
- You MUST ensure ALL customer names, company names, and identifying information are removed
- Use generic terms: "the customer", "the user", "their organization"
- Focus ONLY on technical problems and solutions
- If you detect ANY remaining PII that cannot be anonymized, set approved=false

Review this analyzed support case and:
1. Verify the analysis is accurate
2. Refine and improve the summaries for clarity and searchability
3. ENSURE all summaries are fully anonymized (no customer/company names)
4. Ensure keywords are comprehensive and useful for search
5. Validate the category assignment
6. Generate a final structured document for the knowledge base

<original_thread>
{thread_content}
</original_thread>

<haiku_analysis>
Problem Summary: {problem_summary}
Solution Summary: {solution_summary}
Keywords: {keywords}
Category: {category}
Product: {product}
Resolution Quality: {resolution_quality}
Anonymized Subject: {anonymized_subject}
</haiku_analysis>

Respond with a JSON object:

{{
    "approved": boolean,  // Is this case suitable? Set FALSE if PII cannot be fully anonymized
    "refined_problem": "Improved 2-3 sentence problem description. Be specific and technical. NO customer names.",
    "refined_solution": "Improved 2-3 sentence solution description. Include actionable steps. NO customer names.",
    "refined_keywords": ["array", "of", "keywords"],  // 8-12 keywords for search
    "refined_category": "Category name",
    "suggestions": ["Any suggestions for improvement or notes"],
    "final_document": "A well-structured markdown document summarizing this case for the knowledge base. Include: ## Problem, ## Solution, ## Technical Details sections. MUST be fully anonymized.",
    "anonymized_title": "A clean, technical title for this case (5-10 words). Example: 'RCP Firmware Update Issue on VLAN Network'. NO customer names, company names, or serial numbers."
}}

Only approve if:
- The problem is clearly described
- A concrete solution was provided
- The case adds value to the knowledge base
- ALL content is fully anonymized (no customer/company names remain)

Respond ONLY with valid JSON."""


class QualityController:
    """Quality controller using Claude Sonnet for final review and structuring.

    Reviews Haiku's analysis and produces refined, high-quality case documentation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        """Initialize the quality controller.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use for QC. Defaults to Sonnet.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model
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
            prompt = QC_PROMPT.format(
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
