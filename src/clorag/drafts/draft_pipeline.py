"""Draft creation pipeline orchestrating the full flow."""

from typing import Any

from clorag.analysis.thread_analyzer import ThreadAnalyzer
from clorag.config import get_settings
from clorag.drafts.draft_generator import DraftResponseGenerator
from clorag.drafts.gmail_service import GmailDraftService
from clorag.drafts.models import (
    DraftPreview,
    DraftResult,
    PendingThread,
    PipelineRunResult,
)
from clorag.utils.anonymizer import TextAnonymizer
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


class DraftCreationPipeline:
    """Pipeline for creating draft replies to unanswered support threads.

    Flow:
    1. Fetch threads with support label from Gmail
    2. Get existing draft thread IDs (to avoid duplicates)
    3. Analyze threads with ThreadAnalyzer (Sonnet) to identify unanswered ones
    4. Filter for is_cyanview_response=false AND no existing draft
    5. Generate RAG-based response for each
    6. Create Gmail draft as reply
    """

    def __init__(
        self,
        gmail_service: GmailDraftService | None = None,
        thread_analyzer: ThreadAnalyzer | None = None,
        response_generator: DraftResponseGenerator | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            gmail_service: Gmail service for reading/writing.
            thread_analyzer: Thread analyzer for classification.
            response_generator: RAG-based response generator.
        """
        self._gmail = gmail_service or GmailDraftService()
        self._analyzer = thread_analyzer or ThreadAnalyzer()
        self._generator = response_generator or DraftResponseGenerator()
        self._settings = get_settings()

    async def run(
        self,
        max_drafts: int | None = None,
        preview_only: bool = False,
    ) -> PipelineRunResult:
        """Run the full draft creation pipeline.

        Args:
            max_drafts: Maximum number of drafts to create.
            preview_only: If True, generate previews but don't create drafts.

        Returns:
            PipelineRunResult with statistics and results.
        """
        max_drafts = max_drafts or self._settings.draft_max_per_run
        label = self._settings.gmail_label

        logger.info(
            "Starting draft creation pipeline",
            label=label,
            max_drafts=max_drafts,
            preview_only=preview_only,
        )

        # Step 1: Fetch threads
        threads = await self._gmail.get_threads_by_label(label, max_results=50)
        logger.info("Fetched threads", count=len(threads))

        if not threads:
            return PipelineRunResult(
                threads_checked=0,
                drafts_created=0,
                skipped=0,
            )

        # Step 2: Get existing draft thread IDs
        existing_drafts = await self._gmail.get_draft_thread_ids()

        # Step 3: Fetch full thread content and analyze
        pending_threads: list[tuple[PendingThread, dict[str, Any], str]] = []
        errors: list[str] = []

        for thread_info in threads:
            if len(pending_threads) >= max_drafts * 2:  # Fetch extra in case some are skipped
                break

            thread_id = thread_info["id"]

            # Skip if already has draft
            if thread_id in existing_drafts:
                continue

            # Fetch full thread
            thread = await self._gmail.get_thread(thread_id)
            if not thread:
                continue

            # Extract thread info
            thread_info_obj = self._gmail.extract_thread_info(thread)
            if not thread_info_obj:
                continue

            # Extract content for analysis
            content = self._gmail.extract_thread_content(thread)
            if not content:
                continue

            pending_threads.append((thread_info_obj, thread, content))

        logger.info("Threads to analyze", count=len(pending_threads))

        # Step 4: Analyze threads to find unanswered ones
        unanswered: list[tuple[PendingThread, dict[str, Any], str, str]] = []

        for pending_info, thread, content in pending_threads:
            if len(unanswered) >= max_drafts:
                break

            try:
                # Anonymize content before analysis
                anonymizer = TextAnonymizer()
                anonymized, _ = anonymizer.anonymize(content)

                # Analyze with Sonnet
                analysis = await self._analyzer.analyze_thread(
                    thread_id=pending_info.thread_id,
                    thread_content=anonymized,
                )

                if not analysis:
                    continue

                # Filter: only process threads where Cyanview hasn't responded
                if not analysis.is_cyanview_response:
                    unanswered.append((
                        pending_info,
                        thread,
                        content,
                        analysis.problem_summary,
                    ))
                    logger.info(
                        "Found unanswered thread",
                        thread_id=pending_info.thread_id,
                        subject=pending_info.subject[:50],
                        problem=analysis.problem_summary[:100],
                    )

            except Exception as e:
                error_msg = f"Analysis failed for {pending_info.thread_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)

        logger.info("Unanswered threads found", count=len(unanswered))

        # Step 5 & 6: Generate responses and create drafts
        results: list[DraftResult] = []
        skipped = 0

        for unanswered_info, thread, content, problem_summary in unanswered:
            try:
                # Generate draft response
                preview = await self._generator.generate_draft(
                    problem_summary=problem_summary,
                    thread_content=content,
                    subject=unanswered_info.subject,
                    to_address=unanswered_info.from_address,
                    thread_id=unanswered_info.thread_id,
                )

                if preview_only:
                    logger.info(
                        "Preview generated (not creating draft)",
                        thread_id=unanswered_info.thread_id,
                        confidence=preview.confidence,
                    )
                    skipped += 1
                    continue

                # Create draft in Gmail
                result = await self._gmail.create_draft_reply(
                    thread_id=unanswered_info.thread_id,
                    original_message_id=unanswered_info.last_message_id,
                    to_address=unanswered_info.from_address,
                    subject=unanswered_info.subject,
                    content=preview.content,
                )

                results.append(result)
                logger.info(
                    "Draft created successfully",
                    thread_id=result.thread_id,
                    draft_id=result.draft_id,
                )

            except Exception as e:
                error_msg = f"Draft creation failed for {unanswered_info.thread_id}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        return PipelineRunResult(
            threads_checked=len(pending_threads),
            drafts_created=len(results),
            skipped=skipped,
            errors=errors,
            results=results,
        )

    async def get_pending_threads(
        self,
        limit: int = 20,
    ) -> list[PendingThread]:
        """Get threads that are pending draft creation.

        Args:
            limit: Maximum number of threads to return.

        Returns:
            List of pending threads with their info.
        """
        label = self._settings.gmail_label

        # Fetch threads
        threads = await self._gmail.get_threads_by_label(label, max_results=limit * 2)

        # Get existing draft thread IDs
        existing_drafts = await self._gmail.get_draft_thread_ids()

        pending: list[PendingThread] = []

        for thread_info in threads:
            if len(pending) >= limit:
                break

            thread_id = thread_info["id"]

            # Skip if already has draft
            if thread_id in existing_drafts:
                continue

            # Fetch full thread
            thread = await self._gmail.get_thread(thread_id)
            if not thread:
                continue

            # Extract thread info
            info = self._gmail.extract_thread_info(thread)
            if info:
                pending.append(info)

        return pending

    async def process_single_thread(
        self,
        thread_id: str,
        preview_only: bool = False,
    ) -> DraftPreview | DraftResult | None:
        """Process a single thread for draft creation.

        Args:
            thread_id: Gmail thread ID to process.
            preview_only: If True, return preview without creating draft.

        Returns:
            DraftPreview or DraftResult, or None if thread not eligible.
        """
        logger.info("Processing single thread", thread_id=thread_id, preview_only=preview_only)

        # Fetch thread
        thread = await self._gmail.get_thread(thread_id)
        if not thread:
            logger.warning("Thread not found", thread_id=thread_id)
            return None

        # Extract info
        thread_info = self._gmail.extract_thread_info(thread)
        if not thread_info:
            logger.warning("Could not extract thread info", thread_id=thread_id)
            return None

        # Extract content
        content = self._gmail.extract_thread_content(thread)
        if not content:
            logger.warning("Could not extract thread content", thread_id=thread_id)
            return None

        # Anonymize and analyze
        anonymizer = TextAnonymizer()
        anonymized, _ = anonymizer.anonymize(content)

        analysis = await self._analyzer.analyze_thread(
            thread_id=thread_id,
            thread_content=anonymized,
        )

        if not analysis:
            logger.warning("Analysis returned None", thread_id=thread_id)
            return None

        # Generate draft response
        preview = await self._generator.generate_draft(
            problem_summary=analysis.problem_summary,
            thread_content=content,
            subject=thread_info.subject,
            to_address=thread_info.from_address,
            thread_id=thread_id,
        )

        if preview_only:
            return preview

        # Create draft in Gmail
        result = await self._gmail.create_draft_reply(
            thread_id=thread_id,
            original_message_id=thread_info.last_message_id,
            to_address=thread_info.from_address,
            subject=thread_info.subject,
            content=preview.content,
        )

        return result


async def check_and_create_drafts() -> PipelineRunResult:
    """Background job function for scheduled draft creation.

    This function is called by APScheduler at regular intervals.
    """
    logger.info("Running scheduled draft creation")
    pipeline = DraftCreationPipeline()
    return await pipeline.run()
