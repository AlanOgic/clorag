"""Admin settings management endpoints.

Provides CRUD operations for RAG tuning settings with version history and hot reload.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter
from clorag.web.schemas import (
    SettingMetadataUpdateRequest,
    SettingRollbackRequest,
    SettingUpdateRequest,
)

router = APIRouter(tags=["Settings"])


# -------------------------------------------------------------------------
# Static routes MUST be defined before parameterized routes so that
# FastAPI does not capture "initialize", "reload", or "cache-stats" as a
# {setting_id} path parameter.
# -------------------------------------------------------------------------


@router.post("/settings/initialize")
@limiter.limit("5/minute")
async def api_settings_initialize(
    request: Request,
    force: bool = False,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Seed the settings database with default values."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    result = sm.initialize_defaults(force=force)
    return result


@router.post("/settings/reload")
async def api_settings_reload(
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str]:
    """Hot reload all settings caches."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    sm.reload_all()
    return {"status": "ok", "message": "All settings caches reloaded"}


@router.get("/settings/cache-stats")
async def api_settings_cache_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get settings cache statistics."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    return sm.get_cache_stats()


# -------------------------------------------------------------------------
# Parameterized routes
# -------------------------------------------------------------------------


@router.get("/settings")
async def api_settings_list(
    category: str | None = None,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """List all settings, optionally filtered by category."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    return sm.get_all(category=category)


@router.get("/settings/{setting_id}")
async def api_setting_get(
    setting_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a single setting by ID."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    setting = sm.get_setting_by_id(setting_id)
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    return {"setting": setting.to_dict()}


@router.get("/settings/{setting_id}/versions")
async def api_setting_versions(
    setting_id: str,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """Get version history for a setting."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    versions = sm.get_setting_versions(setting_id)
    return [v.to_dict() for v in versions]


@router.put("/settings/{setting_id}")
@limiter.limit("30/minute")
async def api_setting_update(
    request: Request,
    setting_id: str,
    updates: Annotated[SettingUpdateRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update a setting value (creates a new version if changed)."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    try:
        setting = sm.update(
            setting_id, updates.value, updates.change_note, "admin"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting.to_dict()


@router.put("/settings/{setting_id}/metadata")
@limiter.limit("30/minute")
async def api_setting_update_metadata(
    request: Request,
    setting_id: str,
    updates: Annotated[SettingMetadataUpdateRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update setting metadata (min, max, default) without changing the value."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    try:
        setting = sm.update_metadata(
            setting_id,
            min_value=updates.min_value,
            max_value=updates.max_value,
            default_value=updates.default_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting.to_dict()


@router.post("/settings/{setting_id}/rollback")
@limiter.limit("10/minute")
async def api_setting_rollback(
    request: Request,
    setting_id: str,
    data: Annotated[SettingRollbackRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Rollback a setting to a previous version."""
    from clorag.services.settings_manager import get_settings_manager

    sm = get_settings_manager()
    setting = sm.rollback_setting(setting_id, data.version, rolled_back_by="admin")
    if not setting:
        raise HTTPException(status_code=404, detail="Version not found")
    return setting.to_dict()
