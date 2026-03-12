"""Admin routers package.

This module provides all admin API routes organized by domain.
All routes under /api/admin require authentication via verify_admin.
"""

from fastapi import APIRouter

from clorag.web.routers.admin import (
    analytics,
    auth,
    cameras,
    chunks,
    debug,
    documents,
    drafts,
    graph,
    ingestion,
    prompts,
    support,
    terminology,
)

# Create main admin router
router = APIRouter(prefix="/api/admin", tags=["Admin"])

# Include all admin sub-routers
router.include_router(auth.router)
router.include_router(cameras.router)
router.include_router(analytics.router)
router.include_router(support.router)
router.include_router(terminology.router)
router.include_router(graph.router)
router.include_router(chunks.router)
router.include_router(documents.router)
router.include_router(drafts.router)
router.include_router(prompts.router)
router.include_router(ingestion.router)
router.include_router(debug.router)

__all__ = ["router"]
