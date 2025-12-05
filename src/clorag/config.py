"""Configuration management for CLORAG using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    voyage_api_key: SecretStr = Field(..., description="Voyage AI API key")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: SecretStr | None = Field(default=None, description="Qdrant API key (optional)")
    qdrant_docs_collection: str = Field(
        default="docusaurus_docs", description="Collection for documentation"
    )
    qdrant_cases_collection: str = Field(
        default="gmail_cases", description="Collection for Gmail cases"
    )

    # Voyage AI
    voyage_model: str = Field(default="voyage-context-3", description="Voyage embedding model")
    voyage_dimensions: int = Field(default=1024, description="Embedding dimensions")

    # Docusaurus
    docusaurus_url: str | None = Field(default=None, description="Docusaurus documentation URL")

    # Gmail
    gmail_label: str = Field(default="supports", description="Gmail label for support threads")
    google_credentials_path: Path = Field(
        default=Path("credentials.json"), description="Google OAuth credentials file"
    )
    google_token_path: Path = Field(
        default=Path("token.json"), description="Google OAuth token file"
    )

    # Agent
    claude_model: str = Field(
        default="claude-sonnet-4-20250514", description="Claude model for agent"
    )
    max_turns: int = Field(default=50, description="Maximum conversation turns")

    # LLM Models for analysis pipeline
    haiku_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude Haiku model for fast analysis and synthesis",
    )
    sonnet_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude Sonnet model for quality control",
    )

    # Camera Database
    database_path: str = Field(
        default="data/clorag.db", description="SQLite database path for camera data"
    )
    admin_password: SecretStr | None = Field(
        default=None, description="Admin password for camera management"
    )

    # Analytics Database (separate from camera database)
    analytics_database_path: str = Field(
        default="data/analytics.db", description="SQLite database path for search analytics"
    )

    # Search Engine
    searxng_url: str = Field(
        default="https://search.sapti.me",
        description="SearXNG instance URL for web searches",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Note: Required fields (API keys) are loaded from environment variables
    by pydantic-settings. The type ignore is needed because mypy doesn't
    understand this behavior.
    """
    return Settings()  # type: ignore[call-arg]
