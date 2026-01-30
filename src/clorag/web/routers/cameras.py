"""Public camera API endpoints.

Provides read-only camera compatibility data for public consumption.
"""

import csv
import io
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.responses import Response as StarletteResponse

from clorag.core.database import get_camera_database
from clorag.models.camera import Camera
from clorag.web.dependencies import limiter

router = APIRouter()
logger = structlog.get_logger()


@router.get("/cameras")
async def cameras_redirect() -> RedirectResponse:
    """Redirect public cameras URL to admin cameras page."""
    return RedirectResponse(url="/admin/cameras", status_code=302)


@router.get("/api/cameras", response_model=list[Camera])
@limiter.limit("60/minute")
async def api_cameras_list(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
    page: int | None = None,
    page_size: int = 50,
) -> list[Camera]:
    """Get cameras as JSON with optional pagination.

    If page is not specified, returns all cameras.
    If page is specified, returns paginated results.
    """
    db = get_camera_database()

    if page is not None:
        # Paginated request
        page_size = max(10, min(100, page_size))
        page = max(1, page)
        offset = (page - 1) * page_size
        return db.list_cameras(
            manufacturer=manufacturer,
            device_type=device_type,
            port=port,
            protocol=protocol,
            offset=offset,
            limit=page_size,
        )

    # Non-paginated (all cameras)
    return db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
    )


@router.get("/api/cameras/search", response_model=list[Camera])
@limiter.limit("60/minute")
async def api_cameras_search(request: Request, q: str) -> list[Camera]:
    """Search cameras by name or manufacturer."""
    db = get_camera_database()
    return db.search_cameras(q)


@router.get("/api/cameras/stats")
@limiter.limit("60/minute")
async def api_cameras_stats(request: Request) -> dict[str, Any]:
    """Get camera database statistics."""
    db = get_camera_database()
    return db.get_stats()


@router.get("/api/cameras/{camera_id}", response_model=Camera)
@limiter.limit("120/minute")
async def api_camera_get(request: Request, camera_id: int) -> Camera:
    """Get a single camera by ID."""
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.get("/api/cameras/{camera_id}/related", response_model=list[Camera], tags=["Cameras"])
@limiter.limit("60/minute")
async def api_camera_related(request: Request, camera_id: int, limit: int = 5) -> list[Camera]:
    """Get cameras related to the specified camera.

    Finds cameras with similar manufacturer, device type, ports, or protocols.
    """
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return db.find_related_cameras(camera_id, limit=min(limit, 10))


@router.post("/api/cameras/compare", response_model=list[Camera], tags=["Cameras"])
@limiter.limit("60/minute")
async def api_cameras_compare(request: Request, camera_ids: list[int]) -> list[Camera]:
    """Get multiple cameras for comparison.

    Accepts a list of camera IDs and returns the camera objects in order.
    Maximum 5 cameras can be compared at once.
    """
    if not camera_ids:
        raise HTTPException(status_code=400, detail="No camera IDs provided")
    if len(camera_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 cameras can be compared")

    db = get_camera_database()
    cameras = db.get_cameras_by_ids(camera_ids)
    if not cameras:
        raise HTTPException(status_code=404, detail="No cameras found")
    return cameras


@router.get("/api/cameras/export.csv", tags=["Cameras"])
@limiter.limit("10/minute")
async def api_cameras_export_csv(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
) -> StarletteResponse:
    """Export cameras to CSV format.

    Optionally filter by manufacturer or device type.
    """
    db = get_camera_database()
    cameras = db.list_cameras(manufacturer=manufacturer, device_type=device_type)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "ID", "Name", "Manufacturer", "Code Model", "Device Type",
        "Ports", "Protocols", "Supported Controls", "Notes",
        "Doc URL", "Manufacturer URL", "Source", "Confidence"
    ])

    # Data rows
    for cam in cameras:
        writer.writerow([
            cam.id,
            cam.name,
            cam.manufacturer or "",
            cam.code_model or "",
            cam.device_type.value if cam.device_type else "",
            "|".join(cam.ports),
            "|".join(cam.protocols),
            "|".join(cam.supported_controls),
            "|".join(cam.notes),
            cam.doc_url or "",
            cam.manufacturer_url or "",
            cam.source.value if cam.source else "",
            cam.confidence,
        ])

    # Return as downloadable CSV
    return StarletteResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cameras.csv"}
    )
