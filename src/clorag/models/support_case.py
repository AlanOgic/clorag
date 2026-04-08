"""Support case data model for curated Gmail threads."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CaseStatus(str, Enum):
    """Status of a support case."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    UNCLEAR = "unclear"


class ResolutionQuality(int, Enum):
    """Quality score for case resolution."""

    POOR = 1
    FAIR = 2
    GOOD = 3
    VERY_GOOD = 4
    EXCELLENT = 5


@dataclass
class SupportCase:
    """A curated support case extracted from Gmail threads.

    This represents a fully analyzed and structured support case,
    ready for embedding and storage in the vector database.
    """

    # Identifiers
    id: str
    thread_id: str
    subject: str

    # Status and quality
    status: CaseStatus
    resolution_quality: ResolutionQuality | None = None

    # Extracted content (by Sonnet)
    problem_summary: str = ""
    solution_summary: str = ""
    keywords: list[str] = field(default_factory=list)
    category: str = ""  # e.g., "RCP", "Network", "Hardware", "Software"
    product: str | None = None

    # Structured document for embedding
    document: str = ""  # Full structured case document

    # Raw data
    raw_thread: str = ""
    messages_count: int = 0

    # Metadata
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    participants: list[str] = field(default_factory=list)

    def to_embedding_document(self) -> str:
        """Generate the document text for embedding.

        Returns a structured document that provides good context
        for the embedding model.
        """
        parts = [
            f"# Support Case: {self.subject}",
            "",
            f"**Category:** {self.category}",
            f"**Product:** {self.product or 'N/A'}",
            f"**Status:** {self.status.value}",
            f"**Keywords:** {', '.join(self.keywords)}",
            "",
            "## Problem",
            self.problem_summary,
            "",
            "## Solution",
            self.solution_summary,
        ]
        return "\n".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        """Generate metadata for Qdrant storage."""
        return {
            "source": "gmail",
            "thread_id": self.thread_id,
            "subject": self.subject,
            "status": self.status.value,
            "resolution_quality": (
                self.resolution_quality.value if self.resolution_quality else None
            ),
            "problem_summary": self.problem_summary,
            "solution_summary": self.solution_summary,
            "category": self.category,
            "product": self.product,
            "keywords": self.keywords,
            "messages_count": self.messages_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "participants": self.participants,
        }


@dataclass
class ThreadAnalysis:
    """Result of Sonnet analysis on a Gmail thread."""

    thread_id: str
    is_resolved: bool
    confidence: float  # 0.0 to 1.0

    # Extracted content
    problem_summary: str
    solution_summary: str
    keywords: list[str]
    category: str
    product: str | None

    # Quality assessment
    resolution_quality: int | None  # 1-5
    is_cyanview_response: bool  # Did CyanView respond?

    # Reason for classification
    reasoning: str

    # Anonymized subject line for public display
    anonymized_subject: str = ""


@dataclass
class QualityControlResult:
    """Result of Sonnet QC on analyzed thread."""

    approved: bool
    refined_problem: str
    refined_solution: str
    refined_keywords: list[str]
    refined_category: str
    suggestions: list[str]
    final_document: str

    # Anonymized title for public display
    anonymized_title: str = ""
