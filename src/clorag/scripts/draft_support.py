"""CLI script for creating draft replies to unanswered support threads."""

import argparse

import anyio
import structlog

from clorag.drafts import DraftCreationPipeline, DraftPreview, DraftResult, PipelineRunResult
from clorag.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


async def run_draft_creation(
    max_drafts: int,
    preview_only: bool,
    thread_id: str | None,
) -> int:
    """Run draft creation pipeline.

    Args:
        max_drafts: Maximum number of drafts to create.
        preview_only: If True, preview without creating drafts.
        thread_id: Specific thread ID to process (optional).

    Returns:
        Number of drafts created (or previewed).
    """
    pipeline = DraftCreationPipeline()

    if thread_id:
        # Process single thread
        result = await pipeline.process_single_thread(
            thread_id=thread_id,
            preview_only=preview_only,
        )

        if result is None:
            logger.warning("Thread not found or not eligible", thread_id=thread_id)
            return 0

        if isinstance(result, DraftPreview):
            print(f"\n{'='*60}")
            print(f"DRAFT PREVIEW for thread: {thread_id}")
            print(f"{'='*60}")
            print(f"Subject: Re: {result.subject}")
            print(f"To: {result.to_address}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"\nProblem Summary:\n{result.problem_summary}")
            print(f"\n{'-'*60}")
            print(f"DRAFT CONTENT:\n{'-'*60}")
            print(result.content)
            print(f"\n{'-'*60}")
            print(f"Sources ({len(result.sources)}):")
            for source in result.sources:
                if source.get("url"):
                    print(f"  - {source['title']}: {source['url']}")
                else:
                    print(f"  - {source['title']} (support case)")
            print(f"{'='*60}\n")
            return 1

        elif isinstance(result, DraftResult):
            print("\nDraft created successfully!")
            print(f"  Thread ID: {result.thread_id}")
            print(f"  Draft ID: {result.draft_id}")
            print(f"  Subject: Re: {result.subject}")
            print(f"  To: {result.to_address}")
            return 1

    else:
        # Run full pipeline
        pipeline_result: PipelineRunResult = await pipeline.run(
            max_drafts=max_drafts,
            preview_only=preview_only,
        )

        print(f"\n{'='*60}")
        print("DRAFT CREATION PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Threads checked: {pipeline_result.threads_checked}")
        print(f"Drafts created: {pipeline_result.drafts_created}")
        print(f"Skipped: {pipeline_result.skipped}")

        if pipeline_result.errors:
            print(f"\nErrors ({len(pipeline_result.errors)}):")
            for error in pipeline_result.errors:
                print(f"  - {error}")

        if pipeline_result.results:
            print("\nCreated drafts:")
            for draft in pipeline_result.results:
                print(f"  - {draft.draft_id}: Re: {draft.subject[:50]}...")

        print(f"{'='*60}\n")

        return pipeline_result.drafts_created


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Create draft replies for unanswered support threads"
    )
    parser.add_argument(
        "--max-drafts", "-m",
        type=int,
        default=10,
        help="Maximum number of drafts to create (default: 10)",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview generated responses without creating drafts",
    )
    parser.add_argument(
        "--thread", "-t",
        type=str,
        help="Process a specific thread ID",
    )

    args = parser.parse_args()

    logger.info(
        "Starting draft creation",
        max_drafts=args.max_drafts,
        preview_only=args.preview,
        thread_id=args.thread,
    )

    try:
        count = anyio.run(
            run_draft_creation,
            args.max_drafts,
            args.preview,
            args.thread,
        )
        if args.preview:
            logger.info("Draft preview completed", previews=count)
        else:
            logger.info("Draft creation completed", drafts_created=count)
    except Exception as e:
        logger.error("Draft creation failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
