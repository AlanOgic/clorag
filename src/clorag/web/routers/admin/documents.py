"""Admin custom knowledge documents management endpoints.

Provides CRUD operations for custom knowledge documents including file uploads.
"""

import io
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, Form, HTTPException, Request, UploadFile

from clorag.models.custom_document import (
    CustomDocument,
    CustomDocumentCreate,
    CustomDocumentUpdate,
    DocumentCategory,
)
from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import get_custom_docs_service, limiter
from clorag.web.schemas import KnowledgeListResponse

router = APIRouter(tags=["Knowledge"])
logger = structlog.get_logger()

# Maximum file upload size (10MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# PDF safety limits
MAX_PDF_PAGES = 500  # Maximum pages to process
MAX_PDF_PAGE_TEXT = 100_000  # Maximum text per page (100KB)


async def read_upload_with_limit(
    file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES
) -> bytes:
    """Read uploaded file with streaming size limit to prevent memory exhaustion.

    Args:
        file: FastAPI UploadFile object.
        max_bytes: Maximum allowed file size in bytes.

    Returns:
        File contents as bytes.

    Raises:
        HTTPException: If file exceeds size limit.
    """
    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 64 * 1024  # 64KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_bytes // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


@router.get("/knowledge", response_model=KnowledgeListResponse)
async def api_knowledge_list(
    category: str | None = None,
    include_expired: bool = False,
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(verify_admin),
) -> KnowledgeListResponse:
    """List custom documents with pagination."""
    service = get_custom_docs_service()
    items, total = await service.list_documents(
        limit=limit,
        offset=offset,
        category=category,
        include_expired=include_expired,
    )
    return KnowledgeListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/knowledge/categories")
async def api_knowledge_categories(
    _: bool = Depends(verify_admin),
) -> list[dict[str, str]]:
    """Get available document categories."""
    service = get_custom_docs_service()
    return await service.get_categories()


@router.get("/knowledge/{doc_id}", response_model=CustomDocument)
async def api_knowledge_get(
    doc_id: str,
    _: bool = Depends(verify_admin),
) -> CustomDocument:
    """Get a custom document by ID."""
    service = get_custom_docs_service()
    doc = await service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/knowledge", response_model=CustomDocument)
@limiter.limit("10/minute")
async def api_knowledge_create(
    request: Request,
    doc: Annotated[CustomDocumentCreate, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> CustomDocument:
    """Create a new custom document."""
    service = get_custom_docs_service()
    return await service.create_document(doc, created_by="admin")


@router.put("/knowledge/{doc_id}", response_model=CustomDocument)
@limiter.limit("10/minute")
async def api_knowledge_update(
    request: Request,
    doc_id: str,
    updates: Annotated[CustomDocumentUpdate, Body()],
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> CustomDocument:
    """Update a custom document."""
    service = get_custom_docs_service()
    doc = await service.update_document(doc_id, updates)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/knowledge/{doc_id}")
@limiter.limit("10/minute")
async def api_knowledge_delete(
    request: Request,
    doc_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, str]:
    """Delete a custom document."""
    service = get_custom_docs_service()
    if not await service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": doc_id}


@router.post("/knowledge/upload", response_model=CustomDocument)
@limiter.limit("10/minute")
async def api_knowledge_upload(
    request: Request,
    file: UploadFile,
    title: Annotated[str, Form()] = "",
    category: Annotated[str, Form()] = "other",
    tags: Annotated[str, Form()] = "",
    url_reference: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> CustomDocument:
    """Upload a file (txt, md, pdf) as a custom document.

    Extracts text content from the uploaded file and creates a new document.
    Supported formats: .txt, .md, .pdf
    Max file size: 10MB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # SECURITY: Validate filename to prevent path traversal attacks
    # Reject filenames containing path separators or parent directory references
    if "/" in file.filename or "\\" in file.filename or ".." in file.filename:
        logger.warning(
            "Path traversal attempt in file upload",
            filename=file.filename,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Filenames cannot contain path separators or '..'",
        )

    # Validate file extension
    filename = file.filename.lower()
    if not filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .txt, .md, .pdf",
        )

    # Validate MIME type (defense in depth)
    allowed_mimes = {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/pdf",
        "application/octet-stream",  # Some browsers send this for .md files
    }
    if file.content_type and file.content_type not in allowed_mimes:
        logger.warning(
            "Suspicious file upload: MIME type mismatch",
            filename=filename,
            content_type=file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}",
        )

    # Read file content with streaming size limit
    content_bytes = await read_upload_with_limit(file)
    content = ""

    if filename.endswith((".txt", ".md")):
        # Text/Markdown files - decode as UTF-8
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            try:
                content = content_bytes.decode("latin-1")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode file. Please ensure it's a valid text file.",
                )
    elif filename.endswith(".pdf"):
        # PDF files - extract text using pypdf with safety limits
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError

            pdf_file = io.BytesIO(content_bytes)
            try:
                reader = PdfReader(pdf_file)
            except PdfReadError as e:
                logger.warning("Invalid PDF file", error=str(e))
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or corrupted PDF file.",
                )

            # Enforce page limit to prevent DoS
            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF too large. Maximum {MAX_PDF_PAGES} pages allowed.",
                )

            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Enforce per-page size limit
                    if len(page_text) > MAX_PDF_PAGE_TEXT:
                        logger.warning(
                            "PDF page text too large, truncating",
                            page=i + 1,
                            size=len(page_text),
                        )
                        page_text = page_text[:MAX_PDF_PAGE_TEXT]
                    text_parts.append(page_text)
            content = "\n\n".join(text_parts)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support not available. Install pypdf package.",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("PDF extraction failed", error=str(e))
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF. Please ensure the file is a valid PDF.",
            )

    # Validate content
    content = content.strip()
    if not content or len(content) < 10:
        raise HTTPException(
            status_code=400,
            detail="File content is empty or too short (min 10 characters).",
        )

    # Use filename as title if not provided
    doc_title = title.strip() if title.strip() else Path(file.filename).stem

    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Validate category
    try:
        doc_category = DocumentCategory(category)
    except ValueError:
        doc_category = DocumentCategory.OTHER

    # Create document
    doc_create = CustomDocumentCreate(
        title=doc_title,
        content=content,
        tags=tag_list,
        category=doc_category,
        url_reference=url_reference.strip() if url_reference.strip() else None,
        notes=notes.strip() if notes.strip() else None,
    )

    try:
        logger.info(
            "Creating custom document from upload",
            title=doc_title,
            category=doc_category.value,
            content_length=len(content),
            filename=file.filename,
        )
        service = get_custom_docs_service()
        doc = await service.create_document(doc_create, created_by="admin")
        logger.info(
            "Custom document created successfully", doc_id=doc.id, title=doc.title
        )
        return doc
    except Exception as e:
        logger.error(
            "Failed to create custom document",
            error=str(e),
            error_type=type(e).__name__,
            title=doc_title,
            content_length=len(content),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create document. Please try again or contact support.",
        )
