"""Thread analyzer using Claude Haiku for fast parallel analysis."""

import json

import anthropic
import structlog

from clorag.config import get_settings
from clorag.models import ThreadAnalysis

logger = structlog.get_logger(__name__)

ANALYSIS_PROMPT = """Analyze this support email thread and extract structured information.

IMPORTANT: The content has been pre-processed with placeholder tokens:
- [SERIAL:XXX-N] = Device serial number (e.g., [SERIAL:RCP-1])
- [EMAIL-N] = Email address placeholder
- [PHONE-N] = Phone number placeholder

These placeholders are intentional for anonymization. Do NOT try to reveal or guess the original values.

<thread>
{thread_content}
</thread>

Analyze the thread and respond with a JSON object containing:

1. **is_resolved**: boolean - Is this a resolved support case?
   - TRUE if: CyanView provided a solution AND the customer confirmed it worked OR no further issues were raised
   - FALSE if: Issue still pending, no solution provided, or customer reported the solution didn't work

2. **confidence**: float (0.0-1.0) - How confident are you in the resolution status?

3. **is_cyanview_response**: boolean - Did someone from CyanView (@cyanview.com) respond?

4. **problem_summary**: string - 2-3 sentence summary of the customer's problem.
   - Be specific and technical
   - NEVER include customer names, company names, or organization names
   - Use generic terms: "the customer", "the user", "their system"
   - Focus ONLY on the technical problem

5. **solution_summary**: string - 2-3 sentence summary of the solution provided.
   - If unresolved, describe what was attempted
   - NEVER include customer names or company names
   - Focus ONLY on the technical solution

6. **keywords**: array of strings - 5-10 technical keywords for search (e.g., "RCP", "network", "connection", "firmware", "IP address")

7. **category**: string - Main category: "RCP", "Network", "Hardware", "Software", "Configuration", "Installation", "Other"

8. **product**: string or null - Specific CyanView product mentioned (e.g., "RCP", "RIO", "CVP", null if unclear)

9. **resolution_quality**: integer 1-5 or null - Quality of the resolution:
   - 5: Excellent - Clear problem, complete solution, verified working
   - 4: Very Good - Clear solution provided and likely resolved
   - 3: Good - Solution provided but not verified
   - 2: Fair - Partial solution or workaround
   - 1: Poor - Minimal help provided
   - null: If unresolved

10. **reasoning**: string - Brief explanation of your classification decision

11. **anonymized_subject**: string - Create a clean, technical title for this case.
    - Describe the main technical issue (e.g., "RCP firmware update failure", "Network connectivity issues with RIO")
    - NEVER include customer names, company names, serial numbers, or any identifying information
    - Keep it concise (5-10 words)

Respond ONLY with valid JSON, no markdown formatting."""


class ThreadAnalyzer:
    """Analyzes Gmail threads using Claude Haiku for fast parallel processing.

    Uses Haiku for cost-effective, high-throughput analysis of email threads
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
            model: Model to use for analysis. Defaults to settings.haiku_model.
            max_concurrent: Maximum concurrent requests.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key.get_secret_value()
        self._model = model or settings.haiku_model
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
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": ANALYSIS_PROMPT.format(thread_content=thread_content),
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
