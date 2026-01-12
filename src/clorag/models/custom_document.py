"""Custom document models for manual knowledge base entries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentCategory(str, Enum):
    """Categories for custom documents."""

    PRODUCT_INFO = "product_info"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    FIRMWARE = "firmware"
    RELEASE_NOTES = "release_notes"
    FAQ = "faq"
    BEST_PRACTICES = "best_practices"
    PRE_SALES = "pre_sales"
    INTERNAL = "internal"
    OTHER = "other"


class CustomDocument(BaseModel):
    """A custom document in the knowledge base."""

    id: str | None = None
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=10, description="Document content text")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    category: DocumentCategory = Field(
        default=DocumentCategory.OTHER,
        description="Document category",
    )
    url_reference: str | None = Field(
        None,
        description="External URL reference (optional)",
    )
    expiration_date: datetime | None = Field(
        None,
        description="Date when document should be reviewed/expired",
    )
    notes: str | None = Field(
        None,
        description="Internal notes about this document",
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None
    created_by: str | None = Field(None, description="Admin who created this")


class CustomDocumentCreate(BaseModel):
    """Model for creating a new custom document."""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=10)
    tags: list[str] = Field(default_factory=list)
    category: DocumentCategory = DocumentCategory.OTHER
    url_reference: str | None = None
    expiration_date: datetime | None = None
    notes: str | None = None


class CustomDocumentUpdate(BaseModel):
    """Model for updating an existing custom document."""

    title: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    category: DocumentCategory | None = None
    url_reference: str | None = None
    expiration_date: datetime | None = None
    notes: str | None = None


class CustomDocumentListItem(BaseModel):
    """Lightweight model for listing documents."""

    id: str
    title: str
    category: DocumentCategory
    tags: list[str]
    content_preview: str = Field(description="First 200 chars of content")
    expiration_date: datetime | None = None
    created_at: datetime | None = None
    is_expired: bool = False
