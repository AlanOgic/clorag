"""Web routers package.

This module provides all public and admin API routes organized by domain.
"""

from fastapi import APIRouter

from clorag.web.routers import cameras, pages, search
from clorag.web.routers.admin import router as admin_router

# Create main public router
public_router = APIRouter()

# Include all public sub-routers
public_router.include_router(search.router)
public_router.include_router(cameras.router)
public_router.include_router(pages.router)

__all__ = ["admin_router", "public_router"]
