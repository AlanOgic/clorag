"""Admin draft creation management endpoints.

Provides status monitoring and manual triggering of the draft creation pipeline.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from clorag.config import get_settings
from clorag.web.auth import verify_admin, verify_csrf

router = APIRouter(tags=["Drafts"])


def _get_scheduler() -> Any:
    """Get the background scheduler instance from app module."""
    from clorag.web.app import get_scheduler

    return get_scheduler()


@router.get("/drafts/status")
async def api_drafts_status(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get draft creation system status."""
    settings = get_settings()
    scheduler = _get_scheduler()

    scheduler_running = (
        scheduler is not None and scheduler.running if scheduler else False
    )
    next_run = None

    if scheduler_running and scheduler:
        job = scheduler.get_job("draft_creation")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

    return {
        "scheduler_running": scheduler_running,
        "next_run": next_run,
        "poll_interval_minutes": settings.draft_poll_interval_minutes,
        "polling_enabled": settings.draft_polling_enabled,
    }


@router.get("/drafts/pending")
async def api_pending_threads(
    limit: int = 20,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """Get threads pending draft creation."""
    from clorag.drafts import DraftCreationPipeline

    pipeline = DraftCreationPipeline()
    pending = await pipeline.get_pending_threads(limit=limit)

    return [t.to_dict() for t in pending]


@router.get("/drafts/thread/{thread_id}")
async def api_thread_detail(
    thread_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get full thread details with all messages."""
    from clorag.drafts import GmailDraftService

    gmail = GmailDraftService()
    thread = await gmail.get_thread(thread_id)

    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    detail = gmail.extract_thread_detail(thread)
    if not detail:
        raise HTTPException(status_code=404, detail="Could not extract thread details")

    return detail.to_dict()


@router.post("/drafts/preview/{thread_id}")
async def api_preview_draft(
    thread_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Preview draft for a specific thread without creating it."""
    from clorag.drafts import DraftCreationPipeline, DraftPreview

    pipeline = DraftCreationPipeline()
    result = await pipeline.process_single_thread(thread_id, preview_only=True)

    if not result or not isinstance(result, DraftPreview):
        raise HTTPException(
            status_code=404, detail="Thread not found or not eligible"
        )

    return result.to_dict()


@router.post("/drafts/create/{thread_id}")
async def api_create_draft(
    thread_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Create draft for a specific thread."""
    from clorag.drafts import DraftCreationPipeline, DraftResult

    pipeline = DraftCreationPipeline()
    result = await pipeline.process_single_thread(thread_id, preview_only=False)

    if not result or not isinstance(result, DraftResult):
        raise HTTPException(
            status_code=404, detail="Thread not found or draft creation failed"
        )

    return {
        "success": True,
        **result.to_dict(),
    }


@router.post("/drafts/run")
async def api_run_draft_pipeline(
    max_drafts: int = 5,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Manually trigger the draft creation pipeline."""
    from clorag.drafts import DraftCreationPipeline

    pipeline = DraftCreationPipeline()
    result = await pipeline.run(max_drafts=max_drafts)

    return result.to_dict()
