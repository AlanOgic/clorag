"""Admin terminology fixes management endpoints.

Provides scanning, review, and application of RIO terminology fixes.
"""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from clorag.core.terminology_db import get_terminology_fix_database
from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.schemas import TerminologyBatchStatusUpdate, TerminologyStatusUpdate
from clorag.web.search import get_embeddings, get_sparse_embeddings, get_vectorstore

router = APIRouter(tags=["Terminology Fixes"])
logger = structlog.get_logger()


@router.get("/terminology-fixes")
async def api_list_terminology_fixes(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    collection: str | None = None,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """List terminology fixes with optional filtering."""
    db = get_terminology_fix_database()
    fixes, total = db.list_fixes(
        status=status,  # type: ignore[arg-type]
        collection=collection,
        limit=limit,
        offset=offset,
    )
    return {
        "fixes": [fix.to_dict() for fix in fixes],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/terminology-fixes/stats")
async def api_terminology_fixes_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get terminology fix statistics."""
    db = get_terminology_fix_database()
    return db.get_stats()


@router.get("/terminology-fixes/{fix_id}")
async def api_get_terminology_fix(
    fix_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a terminology fix by ID."""
    db = get_terminology_fix_database()
    fix = db.get_fix(fix_id)
    if not fix:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return fix.to_dict()


@router.put("/terminology-fixes/{fix_id}/status")
async def api_update_terminology_fix_status(
    fix_id: str,
    body: TerminologyStatusUpdate,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update the status of a terminology fix."""
    db = get_terminology_fix_database()
    updated = db.update_status(fix_id, body.status)  # type: ignore[arg-type]
    if not updated:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return {"updated": True, "fix_id": fix_id, "status": body.status}


@router.put("/terminology-fixes/batch-status")
async def api_batch_update_terminology_fix_status(
    body: TerminologyBatchStatusUpdate,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update status for multiple terminology fixes."""
    db = get_terminology_fix_database()
    count = db.update_statuses_batch(body.ids, body.status)  # type: ignore[arg-type]
    return {"updated": count, "status": body.status}


@router.post("/terminology-fixes/apply")
async def api_apply_terminology_fixes(
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Apply all approved terminology fixes to the vector database.

    Uses document-context re-embedding: when a chunk is fixed, all sibling
    chunks from the same document are re-embedded together using contextualized
    embeddings to preserve semantic understanding of the full document.
    """
    from clorag.scripts.fix_rio_terminology import apply_approved_fixes

    db = get_terminology_fix_database()
    approved_fixes = db.get_approved_fixes()

    if not approved_fixes:
        return {"applied": 0, "failed": 0, "message": "No approved fixes to apply"}

    vectorstore = get_vectorstore()
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()

    try:
        applied_count = await apply_approved_fixes(
            vectorstore=vectorstore,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Calculate failed from total - applied
        total_approved = len(approved_fixes)
        failed_count = total_approved - applied_count

        return {
            "applied": applied_count,
            "failed": failed_count,
            "message": (
                f"Applied {applied_count} fixes with document-context re-embedding, "
                f"{failed_count} failed"
            ),
        }
    except Exception as e:
        logger.error("Failed to apply terminology fixes", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply terminology fixes: {str(e)}",
        )


@router.post("/terminology-fixes/scan")
async def api_scan_terminology_fixes(
    max_chunks: int | None = None,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Scan vector database for RIO terminology issues.

    This runs the same scan as `uv run fix-rio-terminology --preview`.
    Suggestions are saved to the database for review.

    Args:
        max_chunks: Maximum chunks to scan (optional, for quick testing).
    """
    from clorag.analysis.rio_analyzer import RIOTerminologyAnalyzer
    from clorag.scripts.fix_rio_terminology import scan_for_rio_mentions

    db = get_terminology_fix_database()
    vectorstore = get_vectorstore()
    analyzer = RIOTerminologyAnalyzer()

    try:
        # Clear previous pending fixes
        cleared = db.clear_pending()

        # Run scan
        fixes = await scan_for_rio_mentions(vectorstore, analyzer, max_chunks)

        if fixes:
            count = db.insert_fixes_batch(fixes)
        else:
            count = 0

        # Get updated stats
        stats = db.get_stats()

        return {
            "scanned": True,
            "max_chunks": max_chunks,
            "cleared_pending": cleared,
            "fixes_found": count,
            "stats": stats,
        }
    except Exception as e:
        logger.error("Failed to scan for terminology issues", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan: {str(e)}",
        )


@router.delete("/terminology-fixes/{fix_id}")
async def api_delete_terminology_fix(
    fix_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Delete a terminology fix."""
    db = get_terminology_fix_database()
    deleted = db.delete_fix(fix_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return {"deleted": True, "fix_id": fix_id}
