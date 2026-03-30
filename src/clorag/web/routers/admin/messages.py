"""Admin messages management endpoints.

Provides CRUD operations for public-facing announcement messages.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter
from clorag.web.schemas import MessageCreateRequest, MessageUpdateRequest

router = APIRouter(tags=["Messages"])


@router.get("/messages")
async def api_list_messages(
    _admin: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """List all messages (including inactive/expired)."""
    from clorag.core.messages_db import get_messages_database

    db = get_messages_database()
    return [m.to_dict() for m in db.get_all_messages()]


@router.post("/messages")
@limiter.limit("10/minute")
async def api_create_message(
    request: Request,
    data: MessageCreateRequest,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Create a new message."""
    from clorag.core.messages_db import get_messages_database

    db = get_messages_database()
    message = db.create_message(
        title=data.title,
        body=data.body,
        message_type=data.message_type,
        link_url=data.link_url,
        is_active=data.is_active,
        sort_order=data.sort_order,
        expires_at=data.expires_at,
    )
    return message.to_dict()


@router.put("/messages/{message_id}")
@limiter.limit("10/minute")
async def api_update_message(
    request: Request,
    message_id: str,
    data: MessageUpdateRequest,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Update a message."""
    from clorag.core.messages_db import get_messages_database

    db = get_messages_database()
    kwargs: dict[str, Any] = {}
    if data.title is not None:
        kwargs["title"] = data.title
    if data.body is not None:
        kwargs["body"] = data.body
    if data.message_type is not None:
        kwargs["message_type"] = data.message_type
    if data.link_url is not None:
        kwargs["link_url"] = data.link_url
    if data.is_active is not None:
        kwargs["is_active"] = data.is_active
    if data.sort_order is not None:
        kwargs["sort_order"] = data.sort_order
    if data.expires_at is not None:
        kwargs["expires_at"] = data.expires_at
    message = db.update_message(message_id, **kwargs)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message.to_dict()


@router.delete("/messages/{message_id}")
async def api_delete_message(
    message_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str]:
    """Delete a message."""
    from clorag.core.messages_db import get_messages_database

    db = get_messages_database()
    if not db.delete_message(message_id):
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "deleted"}
