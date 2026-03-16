"""Admin camera management endpoints.

Provides CRUD operations for camera database entries including
review queue management, CSV import/export, and duplicate merging.
"""

import csv
import io
from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request, UploadFile

from clorag.core.database import get_camera_database
from clorag.models.camera import (
    Camera,
    CameraCreate,
    CameraSource,
    CameraUpdate,
    DeviceType,
)
from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter
from clorag.web.schemas import CameraMergeRequest, CameraMergeResponse

logger = structlog.get_logger()

router = APIRouter(tags=["Cameras"])

# Maximum CSV file upload size (5MB - CSV files are typically small)
MAX_CSV_UPLOAD_BYTES = 5 * 1024 * 1024


@router.get("/cameras/duplicates")
async def api_cameras_duplicates(
    request: Request,
    _: bool = Depends(verify_admin),
) -> list[list[Camera]]:
    """Find groups of suspected duplicate cameras."""
    db = get_camera_database()
    return db.find_duplicate_candidates()


@router.post("/cameras/merge", response_model=CameraMergeResponse)
@limiter.limit("3/minute")
async def api_cameras_merge(
    request: Request,
    body: Annotated[CameraMergeRequest, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> CameraMergeResponse:
    """Merge duplicate cameras into one primary camera."""
    if body.primary_id in body.merge_ids:
        raise HTTPException(status_code=400, detail="primary_id cannot be in merge_ids")

    db = get_camera_database()

    try:
        merged, deleted_ids, deleted_names = db.merge_cameras(
            primary_id=body.primary_id,
            merge_ids=body.merge_ids,
            custom_name=body.custom_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Try to reassign Neo4j graph references (non-fatal)
    try:
        from clorag.core.graph_store import reassign_camera_db_id

        for old_id in deleted_ids:
            try:
                await reassign_camera_db_id(old_id, body.primary_id)
            except Exception:
                logger.warning(
                    "Failed to reassign graph camera_db_id",
                    old_id=old_id,
                    new_id=body.primary_id,
                )
    except ImportError:
        pass
    except Exception:
        logger.warning("Neo4j graph reassignment skipped")

    return CameraMergeResponse(
        merged_camera=merged.model_dump(mode="json"),
        deleted_ids=deleted_ids,
        deleted_names=deleted_names,
    )


@router.post("/cameras", response_model=Camera)
@limiter.limit("5/minute")
async def api_camera_create(
    request: Request,
    camera: Annotated[CameraCreate, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> Camera:
    """Create a new camera entry."""
    db = get_camera_database()
    return db.create_camera(camera, CameraSource.MANUAL)


@router.put("/cameras/{camera_id}", response_model=Camera)
@limiter.limit("5/minute")
async def api_camera_update(
    camera_id: int,
    request: Request,
    updates: Annotated[CameraUpdate, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> Camera:
    """Update an existing camera."""
    db = get_camera_database()
    camera = db.update_camera(camera_id, updates)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/cameras/{camera_id}")
@limiter.limit("5/minute")
async def api_camera_delete(
    camera_id: int,
    request: Request,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str | int]:
    """Delete a camera entry."""
    db = get_camera_database()
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"status": "deleted", "id": camera_id}


@router.get("/cameras/review", response_model=list[Camera])
async def api_cameras_needing_review(
    request: Request,
    page: int = 1,
    page_size: int = 25,
    _: bool = Depends(verify_admin),
) -> list[Camera]:
    """Get cameras needing review."""
    db = get_camera_database()
    page_size = max(10, min(100, page_size))
    page = max(1, page)
    offset = (page - 1) * page_size
    return db.list_cameras_needing_review(offset=offset, limit=page_size)


@router.get("/cameras/review/count")
async def api_cameras_review_count(
    request: Request,
    _: bool = Depends(verify_admin),
) -> dict[str, int]:
    """Get count of cameras needing review."""
    db = get_camera_database()
    return {"count": db.count_cameras_needing_review()}


@router.post("/cameras/{camera_id}/approve", response_model=Camera)
@limiter.limit("30/minute")
async def api_camera_approve(
    camera_id: int,
    request: Request,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> Camera:
    """Approve a camera (clear needs_review flag)."""
    db = get_camera_database()
    camera = db.approve_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.post("/cameras/import")
@limiter.limit("5/minute")
async def api_cameras_import_csv(
    request: Request,
    file: UploadFile,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, int | list[str]]:
    """Import cameras from CSV file.

    CSV format should have headers:
    Name, Manufacturer, Code Model, Device Type, Ports, Protocols, Supported Controls, Notes

    Ports, Protocols, Controls, and Notes should be pipe-separated (|).
    Existing cameras with same name will be updated (merged).
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # Validate MIME type (defense in depth)
    allowed_mimes = {"text/csv", "text/plain", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed_mimes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type for CSV: {file.content_type}",
        )

    # Read with size limit to prevent DoS
    content = await file.read(MAX_CSV_UPLOAD_BYTES + 1)
    if len(content) > MAX_CSV_UPLOAD_BYTES:
        max_mb = MAX_CSV_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"CSV file too large. Maximum size is {max_mb}MB.",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))

    db = get_camera_database()
    imported = 0
    updated = 0
    errors: list[str] = []

    for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
        try:
            name = row.get("Name", "").strip()
            if not name:
                errors.append(f"Row {row_num}: Missing name")
                continue

            # Parse fields
            manufacturer = row.get("Manufacturer", "").strip() or None
            code_model = row.get("Code Model", "").strip() or None

            # Parse device type
            device_type_str = row.get("Device Type", "").strip()
            device_type = None
            if device_type_str:
                try:
                    device_type = DeviceType(device_type_str)
                except ValueError:
                    pass  # Invalid device type, leave as None

            # Parse pipe-separated lists
            ports = [p.strip() for p in row.get("Ports", "").split("|") if p.strip()]
            protocols = [
                p.strip() for p in row.get("Protocols", "").split("|") if p.strip()
            ]
            controls = [
                c.strip()
                for c in row.get("Supported Controls", "").split("|")
                if c.strip()
            ]
            notes = [n.strip() for n in row.get("Notes", "").split("|") if n.strip()]

            # URLs
            doc_url = row.get("Doc URL", "").strip() or None
            manufacturer_url = row.get("Manufacturer URL", "").strip() or None

            camera_data = CameraCreate(
                name=name,
                manufacturer=manufacturer,
                code_model=code_model,
                device_type=device_type,
                ports=ports,
                protocols=protocols,
                supported_controls=controls,
                notes=notes,
                doc_url=doc_url,
                manufacturer_url=manufacturer_url,
            )

            # Check if camera exists
            existing = db.get_camera_by_name(name)
            if existing:
                db.upsert_camera(camera_data, CameraSource.MANUAL)
                updated += 1
            else:
                db.create_camera(camera_data, CameraSource.MANUAL)
                imported += 1

        except Exception as e:
            errors.append(f"Row {row_num}: {str(e)}")

    return {
        "imported": imported,
        "updated": updated,
        "errors": errors[:10],  # Return first 10 errors
        "total_errors": len(errors),
    }
