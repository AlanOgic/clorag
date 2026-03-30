"""Shared dependencies for web routers.

This module provides shared dependencies that multiple routers need access to,
including rate limiting, templates, database access, and authentication.
"""

from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address

from clorag.core.analytics_db import AnalyticsDatabase
from clorag.core.database import get_camera_database
from clorag.services.custom_docs import CustomDocumentService

# Re-export get_camera_database for convenience
__all__ = [
    "get_analytics_db",
    "get_camera_database",
    "get_custom_docs_service",
    "get_templates",
    "limiter",
    "templates",
]

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _timeago(value: str | datetime | None) -> str:
    """Convert a datetime to a relative time string."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return str(value)
    now = datetime.now()
    diff = now - value
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    return value.strftime("%Y-%m-%d")


templates.env.filters["timeago"] = _timeago


def get_templates() -> Jinja2Templates:
    """Get templates instance."""
    return templates


# Lazy-loaded singletons
_analytics_db: AnalyticsDatabase | None = None
_custom_docs_service: CustomDocumentService | None = None


def get_analytics_db() -> AnalyticsDatabase:
    """Get or create AnalyticsDatabase instance (separate from camera DB)."""
    global _analytics_db
    if _analytics_db is None:
        from clorag.config import get_settings

        settings = get_settings()
        _analytics_db = AnalyticsDatabase(settings.analytics_database_path)
    return _analytics_db


def get_custom_docs_service() -> CustomDocumentService:
    """Get or create CustomDocumentService instance."""
    global _custom_docs_service
    if _custom_docs_service is None:
        _custom_docs_service = CustomDocumentService()
    return _custom_docs_service
