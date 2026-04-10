"""HTML page routes.

Provides HTML page rendering for public and admin pages.
Note: Admin HTML pages are accessible without auth - JavaScript prompts for
password which is used for API calls (which are protected).
"""

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Cookie, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from itsdangerous import BadSignature, SignatureExpired
from starlette.responses import JSONResponse

from clorag.core.database import get_camera_database
from clorag.web.auth import (
    ADMIN_SESSION_MAX_AGE,
    get_session_serializer,
)
from clorag.web.dependencies import get_templates

router = APIRouter()
logger = structlog.get_logger()

# Cache for admin OpenAPI schema
_admin_openapi_schema: dict[str, Any] | None = None

# Company information for legal pages (update with real values)
COMPANY_INFO: dict[str, str] = {
    "company_name": "Cyanview SA",
    "legal_form": "Societe Anonyme",
    "registered_office": "[Address], Belgium",
    "rpr_number": "[RPR Number]",
    "rpr_court": "Tribunal de l'entreprise de [District]",
    "cbe_number": "[CBE Number]",
    "vat_number": "BE [VAT Number]",
    "contact_email": "support@cyanview.com",
    "dpo_email": "privacy@cyanview.com",
    "responsible_person": "Cyanview SA",
    "hosting_provider": "[Hosting Provider Name and Address]",
    "hosting_location": "European Union",
    "retention_analytics": "12 months",
    "retention_session": "24 hours",
    "last_updated": "March 2026",
}


# =============================================================================
# Public Pages
# =============================================================================


@router.get("/help", response_class=HTMLResponse)
async def help_page(request: Request) -> HTMLResponse:
    """Public user guide page."""
    templates = get_templates()
    return templates.TemplateResponse("help.html", {"request": request})


@router.get("/video", response_class=HTMLResponse)
async def video_page(request: Request) -> HTMLResponse:
    """Public video showcase page."""
    templates = get_templates()
    return templates.TemplateResponse("video.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request) -> HTMLResponse:
    """Privacy policy page (GDPR requirement)."""
    templates = get_templates()
    return templates.TemplateResponse(
        "privacy.html", {"request": request, **COMPANY_INFO}
    )


@router.get("/terms", response_class=HTMLResponse)
async def terms_of_service(request: Request) -> HTMLResponse:
    """Terms of service page."""
    templates = get_templates()
    return templates.TemplateResponse(
        "terms.html", {"request": request, **COMPANY_INFO}
    )


@router.get("/legal", response_class=HTMLResponse)
async def legal_notice(request: Request) -> HTMLResponse:
    """Legal notice / mentions legales (Belgian law requirement)."""
    templates = get_templates()
    return templates.TemplateResponse(
        "legal.html", {"request": request, **COMPANY_INFO}
    )


@router.get("/cookies", response_class=HTMLResponse)
async def cookie_policy(request: Request) -> HTMLResponse:
    """Cookie policy page."""
    templates = get_templates()
    return templates.TemplateResponse(
        "cookies.html", {"request": request, **COMPANY_INFO}
    )


# =============================================================================
# Admin Pages
# =============================================================================

# Public documentation pages (no auth required)
_PUBLIC_DOC_PAGES = frozenset(
    {"architecture", "retrieval", "ingestion", "content", "features", "api", "mcp"}
)


@router.get("/docs", response_class=HTMLResponse)
async def public_docs_index(request: Request) -> HTMLResponse:
    """Public technical documentation - index page."""
    templates = get_templates()
    return templates.TemplateResponse("public_docs/index.html", {"request": request})


@router.get("/docs/{page}", response_class=HTMLResponse)
async def public_docs_page(request: Request, page: str) -> HTMLResponse:
    """Public technical documentation - specific page."""
    if page not in _PUBLIC_DOC_PAGES:
        raise HTTPException(status_code=404, detail="Page not found")
    templates = get_templates()
    return templates.TemplateResponse(
        f"public_docs/{page}.html", {"request": request}
    )


# =============================================================================
# Admin pages
# =============================================================================


@router.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request) -> HTMLResponse:
    """Admin index page with links to all admin pages."""
    templates = get_templates()
    return templates.TemplateResponse("admin_index.html", {"request": request})


@router.get("/admin/docs", response_class=HTMLResponse)
async def admin_docs_index(request: Request) -> HTMLResponse:
    """Admin technical documentation - index page."""
    templates = get_templates()
    return templates.TemplateResponse("docs/index.html", {"request": request})


@router.get("/admin/docs/{page}", response_class=HTMLResponse)
async def admin_docs_page(request: Request, page: str) -> HTMLResponse:
    """Admin technical documentation - specific page."""
    templates = get_templates()
    # Validate page name to prevent path traversal
    if not page.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=404, detail="Page not found")
    return templates.TemplateResponse(f"docs/{page}.html", {"request": request})


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request) -> HTMLResponse:
    """Admin login page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_login.html", {"request": request})


@router.get("/admin/cameras", response_class=HTMLResponse)
async def admin_cameras_list(
    request: Request,
    manufacturer: str | None = None,
) -> HTMLResponse:
    """Admin camera list with edit capabilities.

    Note: HTML pages are accessible without auth - password is prompted
    via JavaScript and used for API calls (which are protected).
    """
    db = get_camera_database()
    templates = get_templates()
    cameras = db.list_cameras(manufacturer=manufacturer)
    manufacturers = db.get_manufacturers()
    stats = db.get_stats()

    return templates.TemplateResponse(
        "admin_cameras.html",
        {
            "request": request,
            "cameras": cameras,
            "manufacturers": manufacturers,
            "selected_manufacturer": manufacturer,
            "stats": stats,
        },
    )


@router.get("/admin/cameras/{camera_id}/edit", response_class=HTMLResponse)
async def admin_camera_edit(
    request: Request,
    camera_id: int,
) -> HTMLResponse:
    """Camera edit form."""
    db = get_camera_database()
    templates = get_templates()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return templates.TemplateResponse(
        "camera_edit.html",
        {
            "request": request,
            "camera": camera,
        },
    )


@router.get("/admin/cameras/new", response_class=HTMLResponse)
async def admin_camera_new(
    request: Request,
) -> HTMLResponse:
    """New camera form."""
    templates = get_templates()
    return templates.TemplateResponse(
        "camera_edit.html",
        {
            "request": request,
            "camera": None,
        },
    )


@router.get("/admin/cameras/review", response_class=HTMLResponse)
async def admin_cameras_review(
    request: Request,
    page: int = 1,
    page_size: int = 25,
) -> HTMLResponse:
    """Admin page for reviewing low-confidence camera extractions."""
    db = get_camera_database()
    templates = get_templates()

    # Get total count for pagination
    total_count = db.count_cameras_needing_review()
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Clamp page to valid range
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size

    cameras = db.list_cameras_needing_review(offset=offset, limit=page_size)

    return templates.TemplateResponse(
        "admin_cameras_review.html",
        {
            "request": request,
            "cameras": cameras,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


@router.get("/admin/cameras-list", response_class=HTMLResponse)
async def admin_cameras_browser(request: Request) -> HTMLResponse:
    """Admin camera browser page with detail popup."""
    templates = get_templates()
    return templates.TemplateResponse("admin_cameras_list.html", {"request": request})


@router.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics(request: Request) -> HTMLResponse:
    """Admin analytics dashboard page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_analytics.html", {"request": request})


@router.get("/admin/support-cases", response_class=HTMLResponse)
async def admin_support_cases(request: Request) -> HTMLResponse:
    """Admin support cases management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_support_cases.html", {"request": request})


@router.get("/admin/terminology-fixes", response_class=HTMLResponse)
async def admin_terminology_fixes(request: Request) -> HTMLResponse:
    """Admin terminology fixes management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_terminology_fixes.html", {"request": request})


@router.get("/admin/chunks", response_class=HTMLResponse)
async def admin_chunks_list(request: Request) -> HTMLResponse:
    """Admin chunk browser page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_chunks.html", {"request": request})


@router.get("/admin/chunks/{collection}/{chunk_id}", response_class=HTMLResponse)
async def admin_chunk_detail_page(
    request: Request,
    collection: str,
    chunk_id: str,
) -> HTMLResponse:
    """Admin chunk detail/edit page."""
    templates = get_templates()
    return templates.TemplateResponse(
        "admin_chunk_edit.html",
        {"request": request, "collection": collection, "chunk_id": chunk_id},
    )


@router.get("/admin/graph", response_class=HTMLResponse)
async def admin_graph_page(request: Request) -> HTMLResponse:
    """Admin knowledge graph explorer page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_graph.html", {"request": request})


@router.get("/admin/knowledge", response_class=HTMLResponse)
async def admin_knowledge(request: Request) -> HTMLResponse:
    """Admin custom knowledge management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_knowledge.html", {"request": request})


@router.get("/admin/drafts", response_class=HTMLResponse)
async def admin_drafts(request: Request) -> HTMLResponse:
    """Admin draft management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_drafts.html", {"request": request})


@router.get("/admin/search-debug", response_class=HTMLResponse)
async def admin_search_debug(request: Request) -> HTMLResponse:
    """Admin search debug page - shows chunks and LLM response."""
    templates = get_templates()
    return templates.TemplateResponse("admin_search_debug.html", {"request": request})


@router.get("/admin/prompts", response_class=HTMLResponse)
async def admin_prompts(request: Request) -> HTMLResponse:
    """Admin prompt management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_prompts.html", {"request": request})


@router.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request) -> HTMLResponse:
    """Admin RAG settings page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_settings.html", {"request": request})


@router.get("/admin/ingestion", response_class=HTMLResponse)
async def admin_ingestion(request: Request) -> HTMLResponse:
    """Admin ingestion management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_ingestion.html", {"request": request})


@router.get("/admin/messages", response_class=HTMLResponse)
async def admin_messages(request: Request) -> HTMLResponse:
    """Admin messages management page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_messages.html", {"request": request})


@router.get("/admin/metrics", response_class=HTMLResponse)
async def admin_metrics(request: Request) -> HTMLResponse:
    """Admin performance metrics dashboard page."""
    templates = get_templates()
    return templates.TemplateResponse("admin_metrics.html", {"request": request})


# =============================================================================
# Admin OpenAPI Documentation
# =============================================================================


def _get_admin_openapi_schema(routes: list[Any]) -> dict[str, Any]:
    """Generate OpenAPI schema for admin endpoints only.

    Filters the main app schema to include only /api/admin/* routes.
    """
    global _admin_openapi_schema
    if _admin_openapi_schema:
        return _admin_openapi_schema

    # Generate full schema
    full_schema = get_openapi(
        title="Cyanview Admin API",
        version="1.0.0",
        description="Admin API for Cyanview AI Search.",
        routes=routes,
    )

    # Filter paths to include only /api/admin/* routes
    admin_paths = {
        path: ops
        for path, ops in full_schema.get("paths", {}).items()
        if path.startswith("/api/admin")
    }
    full_schema["paths"] = admin_paths

    # Filter tags to include only admin-related tags
    admin_tags = ["Authentication", "Backup", "Cameras", "Analytics", "Drafts", "Debug"]
    full_schema["tags"] = [
        {"name": tag, "description": f"{tag} operations"}
        for tag in admin_tags
    ]

    _admin_openapi_schema = full_schema
    return _admin_openapi_schema


@router.get("/admin/openapi.json", include_in_schema=False)
async def admin_openapi_json(
    request: Request,
    admin_session: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """Serve OpenAPI schema for admin endpoints only (requires authentication)."""
    # Verify session
    if not admin_session:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if not data.get("authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")
    except (SignatureExpired, BadSignature):
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    # Get routes from the app via request
    return JSONResponse(content=_get_admin_openapi_schema(request.app.routes))


@router.get("/admin/api-docs", include_in_schema=False, response_model=None)
async def admin_swagger_ui(
    admin_session: Annotated[str | None, Cookie()] = None,
) -> Response:
    """Serve Swagger UI for admin API (requires authentication)."""
    # Verify session
    if not admin_session:
        return RedirectResponse(url="/admin/login?next=/admin/api-docs")

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if not data.get("authenticated"):
            return RedirectResponse(url="/admin/login?next=/admin/api-docs")
    except (SignatureExpired, BadSignature):
        return RedirectResponse(url="/admin/login?next=/admin/api-docs")

    return get_swagger_ui_html(
        openapi_url="/admin/openapi.json",
        title="Cyanview Admin API",
        swagger_favicon_url="/static/favicon.ico",
    )
