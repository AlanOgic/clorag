"""Data models for draft creation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ThreadMessage:
    """A single message in a Gmail thread."""

    message_id: str
    from_address: str
    from_name: str
    date: datetime | None
    snippet: str
    body: str
    is_cyanview: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "message_id": self.message_id,
            "from_address": self.from_address,
            "from_name": self.from_name,
            "date": self.date.isoformat() if self.date else None,
            "snippet": self.snippet,
            "body": self.body,
            "is_cyanview": self.is_cyanview,
        }


@dataclass
class ThreadDetail:
    """Full details of a Gmail thread with all messages."""

    thread_id: str
    subject: str
    gmail_link: str
    messages: list[ThreadMessage] = field(default_factory=list)
    problem_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "gmail_link": self.gmail_link,
            "messages": [m.to_dict() for m in self.messages],
            "problem_summary": self.problem_summary,
        }


@dataclass
class PendingThread:
    """A Gmail thread pending draft creation."""

    thread_id: str
    subject: str
    from_address: str
    last_message_id: str
    message_count: int
    received_at: datetime | None
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "from_address": self.from_address,
            "last_message_id": self.last_message_id,
            "message_count": self.message_count,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "snippet": self.snippet,
        }


@dataclass
class DraftPreview:
    """Preview of a draft response before creation."""

    thread_id: str
    subject: str
    to_address: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    problem_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "to_address": self.to_address,
            "content": self.content,
            "sources": self.sources,
            "confidence": self.confidence,
            "problem_summary": self.problem_summary,
        }


@dataclass
class DraftResult:
    """Result of creating a draft in Gmail."""

    thread_id: str
    draft_id: str
    subject: str
    to_address: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "thread_id": self.thread_id,
            "draft_id": self.draft_id,
            "subject": self.subject,
            "to_address": self.to_address,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PipelineRunResult:
    """Result of running the draft creation pipeline."""

    threads_checked: int
    drafts_created: int
    skipped: int
    errors: list[str] = field(default_factory=list)
    results: list[DraftResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "threads_checked": self.threads_checked,
            "drafts_created": self.drafts_created,
            "skipped": self.skipped,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results],
        }
