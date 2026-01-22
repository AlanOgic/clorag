"""Public camera API endpoints.

Provides read-only camera compatibility data for public consumption.
"""

import csv
import io

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.responses import Response as StarletteResponse

from clorag.core.database import get_camera_database
from clorag.models.camera import Camera
from clorag.web.dependencies import get_templates, limiter

router = APIRouter()
logger = structlog.get_logger()


@router.get("/cameras", response_class=HTMLResponse)
async def cameras_list(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> Response:
    """Render the public camera compatibility list with pagination."""
    db = get_camera_database()
    templates = get_templates()

    # Clamp page_size to reasonable limits
    page_size = max(10, min(100, page_size))
    page = max(1, page)

    # Get total count for pagination
    total_count = db.count_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
    )
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Clamp page to valid range
    page = min(page, total_pages)
    offset = (page - 1) * page_size

    cameras = db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
        offset=offset,
        limit=page_size,
    )
    manufacturers = db.get_manufacturers()
    device_types = db.get_device_types()
    ports = db.get_all_ports()
    protocols = db.get_all_protocols()
    stats = db.get_stats()

    return templates.TemplateResponse(
        "cameras.html",
        {
            "request": request,
            "cameras": cameras,
            "manufacturers": manufacturers,
            "device_types": device_types,
            "ports": ports,
            "protocols": protocols,
            "selected_manufacturer": manufacturer,
            "selected_device_type": device_type,
            "selected_port": port,
            "selected_protocol": protocol,
            "stats": stats,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


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
async def api_cameras_stats(request: Request) -> dict[str, int]:
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
