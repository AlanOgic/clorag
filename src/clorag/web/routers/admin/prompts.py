"""Admin prompt management endpoints.

Provides CRUD operations for LLM prompts with version history and hot reload.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter
from clorag.web.schemas import (
    PromptCreateFromDefaultRequest,
    PromptRollbackRequest,
    PromptTestRequest,
    PromptUpdateRequest,
)

router = APIRouter(tags=["Prompts"])


@router.get("/prompts")
async def api_prompts_list(
    category: str | None = None,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """List all prompts from database and defaults."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    return pm.list_all_prompts(category=category)


@router.get("/prompts/by-key/{key:path}")
async def api_prompt_by_key(
    key: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a prompt by its key (works for both database and default prompts)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    try:
        return pm.get_prompt_with_metadata(key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {key}")


@router.get("/prompts/{prompt_id}")
async def api_prompt_get(
    prompt_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a database prompt by ID with metadata."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    prompt = pm.get_prompt_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"source": "database", "prompt": prompt.to_dict()}


@router.get("/prompts/{prompt_id}/versions")
async def api_prompt_versions(
    prompt_id: str,
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """Get version history for a prompt."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    versions = pm.get_prompt_versions(prompt_id)
    return [v.to_dict() for v in versions]


@router.put("/prompts/{prompt_id}")
@limiter.limit("30/minute")
async def api_prompt_update(
    request: Request,
    prompt_id: str,
    updates: Annotated[PromptUpdateRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update a prompt (creates a new version if content changed)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()

    # Detect variables in new content if provided
    variables = None
    if updates.content:
        variables = pm.detect_variables(updates.content)

    prompt = pm.update_prompt(
        prompt_id=prompt_id,
        content=updates.content,
        name=updates.name,
        description=updates.description,
        model=updates.model,
        variables=variables,
        change_note=updates.change_note,
        updated_by="admin",
    )
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt.to_dict()


@router.post("/prompts/from-default")
@limiter.limit("10/minute")
async def api_prompt_create_from_default(
    request: Request,
    data: Annotated[PromptCreateFromDefaultRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Create a database prompt from a default, optionally with modifications."""
    from clorag.core.prompt_db import get_prompt_database
    from clorag.services.default_prompts import get_default_prompt
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    default = get_default_prompt(data.key)
    if not default:
        raise HTTPException(
            status_code=404, detail=f"Default prompt not found: {data.key}"
        )

    db = get_prompt_database()

    # Check if already exists in database
    existing = db.get_prompt_by_key(data.key)
    if existing:
        # Update existing
        variables = pm.detect_variables(data.content or default.content)
        prompt = pm.update_prompt(
            prompt_id=existing.id,
            content=data.content or existing.content,
            name=data.name or existing.name,
            description=data.description or existing.description,
            model=data.model or existing.model,
            variables=variables,
            change_note="Updated from admin UI",
            updated_by="admin",
        )
        return prompt.to_dict() if prompt else existing.to_dict()

    # Create new from default
    content = data.content or default.content
    variables = pm.detect_variables(content)

    prompt = db.create_prompt(
        key=data.key,
        name=data.name or default.name,
        content=content,
        category=default.category,
        description=data.description or default.description,
        model=data.model or default.model,
        variables=variables,
        created_by="admin",
    )
    return prompt.to_dict()


@router.post("/prompts/{prompt_id}/rollback")
@limiter.limit("10/minute")
async def api_prompt_rollback(
    request: Request,
    prompt_id: str,
    data: Annotated[PromptRollbackRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Rollback a prompt to a previous version."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    prompt = pm.rollback_prompt(prompt_id, data.version, rolled_back_by="admin")
    if not prompt:
        raise HTTPException(status_code=404, detail="Version not found")
    return prompt.to_dict()


@router.post("/prompts/initialize")
@limiter.limit("5/minute")
async def api_prompts_initialize(
    request: Request,
    force: bool = False,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Initialize database with default prompts."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    result = pm.initialize_defaults(force=force)
    return result


@router.post("/prompts/reload")
async def api_prompts_reload(
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str]:
    """Reload all prompt caches (hot reload)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    pm.reload_all()
    return {"status": "ok", "message": "All prompt caches reloaded"}


@router.get("/prompts/cache-stats")
async def api_prompts_cache_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get prompt cache statistics."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    return pm.get_cache_stats()


@router.post("/prompts/test")
@limiter.limit("20/minute")
async def api_prompt_test(
    request: Request,
    data: Annotated[PromptTestRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Test a prompt with variable substitution (no LLM call)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    try:
        result = pm._substitute_variables(data.content, data.variables)
        return {
            "success": True,
            "result": result,
            "detected_variables": pm.detect_variables(data.content),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
