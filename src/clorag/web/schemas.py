"""Pydantic models and schemas for the web API.

This module contains all request/response models, enums, and dataclasses
used by the FastAPI web application.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from clorag.models.custom_document import CustomDocumentListItem

# =============================================================================
# Enums
# =============================================================================


class SearchSource(str, Enum):
    """Search source options."""

    DOCS = "docs"
    GMAIL = "gmail"
    BOTH = "both"


class ChunkCollection(str, Enum):
    """Available chunk collections."""

    DOCS = "docusaurus_docs"
    CASES = "gmail_cases"
    CUSTOM = "custom_docs"


# =============================================================================
# Search Models
# =============================================================================


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=2000)
    source: SearchSource = SearchSource.BOTH
    limit: int = Field(10, ge=1, le=50)
    session_id: str | None = Field(None, description="Session ID for follow-up conversations")


class SourceLink(BaseModel):
    """A source link for the answer."""

    title: str
    url: str | None = None
    source_type: str  # "documentation" or "gmail_case"


class SearchResult(BaseModel):
    """Individual search result."""

    score: float
    source: str
    title: str
    url: str | None = None
    subject: str | None = None
    snippet: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model with AI-generated answer."""

    query: str
    source: str
    answer: str  # Claude-generated comprehensive answer
    source_links: list[SourceLink]  # Top 3 relevant sources
    results: list[SearchResult]
    total: int
    session_id: str | None = None  # Session ID for follow-up conversations


class DebugSearchResponse(BaseModel):
    """Debug search response with full chunk details."""

    query: str
    source: str
    # Timing
    retrieval_time_ms: int
    synthesis_time_ms: int
    total_time_ms: int
    # Chunks retrieved
    chunks: list[dict[str, Any]]
    # Prompt sent to LLM
    llm_prompt: str
    system_prompt: str
    # LLM response
    llm_response: str
    model: str


# =============================================================================
# Authentication Models
# =============================================================================


class LoginRequest(BaseModel):
    """Login request model."""

    password: str


class LoginResponse(BaseModel):
    """Login response model."""

    success: bool
    message: str


# =============================================================================
# Chunk Models
# =============================================================================


class ChunkListItem(BaseModel):
    """Chunk item for listing."""

    id: str
    collection: str
    text_preview: str = Field(description="First 200 chars of text")
    title: str | None = None
    subject: str | None = None
    url: str | None = None
    chunk_index: int | None = None
    source: str | None = None


class ChunkListResponse(BaseModel):
    """Paginated chunk list response."""

    chunks: list[ChunkListItem]
    next_offset: str | None = None
    total: int | None = None


class ChunkDetail(BaseModel):
    """Full chunk details for viewing/editing."""

    id: str
    collection: str
    text: str
    # Common metadata
    source: str | None = None
    chunk_index: int | None = None
    # Documentation-specific
    url: str | None = None
    title: str | None = None
    lastmod: str | None = None
    parent_id: str | None = None
    # Gmail case-specific
    subject: str | None = None
    thread_id: str | None = None
    parent_case_id: str | None = None
    problem_summary: str | None = None
    solution_summary: str | None = None
    category: str | None = None
    product: str | None = None
    keywords: list[str] | None = None
    # Raw metadata for anything else
    metadata: dict[str, Any]


class ChunkUpdate(BaseModel):
    """Chunk update request."""

    text: str | None = Field(None, description="New text content (triggers re-embedding)")
    title: str | None = Field(None, description="New title (docs)")
    subject: str | None = Field(None, description="New subject (cases)")


# =============================================================================
# Knowledge Document Models
# =============================================================================


class KnowledgeListResponse(BaseModel):
    """Paginated response for custom documents list."""

    items: list[CustomDocumentListItem]
    total: int
    limit: int
    offset: int


# =============================================================================
# Terminology Models
# =============================================================================


class TerminologyStatusUpdate(BaseModel):
    """Request body for status update."""

    status: str = Field(..., pattern="^(pending|approved|rejected|applied)$")


class TerminologyBatchStatusUpdate(BaseModel):
    """Request body for batch status update."""

    ids: list[str]
    status: str = Field(..., pattern="^(pending|approved|rejected|applied)$")


# =============================================================================
# Camera Merge Models
# =============================================================================


class CameraMergeRequest(BaseModel):
    """Request body for merging duplicate cameras."""

    primary_id: int
    merge_ids: list[int] = Field(..., min_length=1)
    custom_name: str | None = None


class CameraMergeResponse(BaseModel):
    """Response for a camera merge operation."""

    merged_camera: dict[str, Any]
    deleted_ids: list[int]
    deleted_names: list[str]


# =============================================================================
# Graph Models
# =============================================================================


class RelationshipDeleteRequest(BaseModel):
    """Request body for deleting a relationship."""

    source_type: str
    source_name: str
    rel_type: str
    target_type: str
    target_name: str


class RelationshipUpdateRequest(BaseModel):
    """Request body for updating a relationship type."""

    source_type: str
    source_name: str
    old_rel_type: str
    new_rel_type: str
    target_type: str
    target_name: str


# =============================================================================
# Prompt Models
# =============================================================================


class PromptUpdateRequest(BaseModel):
    """Request body for updating a prompt."""

    name: str | None = None
    description: str | None = None
    model: str | None = None
    content: str | None = None
    change_note: str | None = None


class PromptCreateFromDefaultRequest(BaseModel):
    """Request body for creating a prompt from defaults."""

    key: str
    name: str | None = None
    description: str | None = None
    model: str | None = None
    content: str | None = None


class PromptRollbackRequest(BaseModel):
    """Request body for rollback."""

    version: int


class PromptTestRequest(BaseModel):
    """Request body for testing a prompt."""

    content: str
    variables: dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Settings Models
# =============================================================================


class SettingUpdateRequest(BaseModel):
    """Request body for updating a setting value."""

    value: str
    change_note: str | None = None


class SettingRollbackRequest(BaseModel):
    """Request body for rolling back a setting to a previous version."""

    version: int


# =============================================================================
# Session Dataclasses
# =============================================================================

MAX_CONVERSATION_HISTORY = 3  # Keep last 3 Q&A exchanges
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes session timeout


@dataclass
class ConversationExchange:
    """A single Q&A exchange in the conversation."""

    query: str
    answer: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """Server-side conversation session with history."""

    session_id: str
    exchanges: list[ConversationExchange] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def add_exchange(self, query: str, answer: str) -> None:
        """Add a new exchange and trim to max history."""
        self.exchanges.append(ConversationExchange(query=query, answer=answer))
        # Keep only the last N exchanges
        if len(self.exchanges) > MAX_CONVERSATION_HISTORY:
            self.exchanges = self.exchanges[-MAX_CONVERSATION_HISTORY:]
        self.last_accessed = time.time()

    def get_context_messages(self) -> list[dict[str, str]]:
        """Get conversation history as Claude message format."""
        messages: list[dict[str, str]] = []
        for exchange in self.exchanges:
            messages.append({"role": "user", "content": exchange.query})
            messages.append({"role": "assistant", "content": exchange.answer})
        return messages

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_accessed) > SESSION_TTL_SECONDS
