"""Admin support cases management endpoints.

Provides CRUD operations for support case database with full-text search.
"""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from clorag.core.support_case_db import get_support_case_database
from clorag.web.auth import verify_admin

router = APIRouter(tags=["Support Cases"])


@router.get("/support-cases")
async def api_list_support_cases(
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    product: str | None = None,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """List support cases with optional filtering."""
    db = get_support_case_database()
    cases, total = db.list_cases(
        category=category,
        product=product,
        limit=limit,
        offset=offset,
    )
    return {
        "cases": [asdict(case) for case in cases],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/support-cases/stats")
async def api_support_cases_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get support case statistics."""
    db = get_support_case_database()
    return db.get_stats()


@router.get("/support-cases/search")
async def api_search_support_cases(
    q: str,
    limit: int = 20,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Search support cases using full-text search."""
    db = get_support_case_database()
    cases = db.search_cases(q, limit=limit)
    return {"cases": [asdict(case) for case in cases], "total": len(cases)}


@router.get("/support-cases/{case_id}")
async def api_get_support_case(
    case_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a support case by ID with full document."""
    db = get_support_case_database()
    case = db.get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Support case not found")
    return asdict(case)


@router.get("/support-cases/{case_id}/raw-thread")
async def api_get_support_case_raw_thread(
    case_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, str]:
    """Get the raw anonymized thread content for a case."""
    db = get_support_case_database()
    raw_thread = db.get_raw_thread(case_id)
    if raw_thread is None:
        raise HTTPException(status_code=404, detail="Raw thread not found")
    return {"raw_thread": raw_thread}


@router.delete("/support-cases/{case_id}")
async def api_delete_support_case(
    case_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Delete a support case."""
    db = get_support_case_database()
    deleted = db.delete_case(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Support case not found")
    return {"deleted": True, "case_id": case_id}
