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
    qdrant_custom_docs_collection: str = Field(
        default="custom_docs", description="Collection for custom knowledge documents"
    )

    # Voyage AI
    voyage_model: str = Field(default="voyage-context-3", description="Voyage embedding model")
    voyage_dimensions: int = Field(default=1024, description="Embedding dimensions")

    # Reranking
    rerank_enabled: bool = Field(
        default=True,
        description="Enable reranking after hybrid search for improved relevance",
    )
    voyage_rerank_model: str = Field(
        default="rerank-2.5",
        description="Voyage AI reranking model (rerank-2.5, rerank-2.5-lite)",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of top results to return after reranking",
    )

    # Docusaurus
    docusaurus_url: str | None = Field(default=None, description="Docusaurus documentation URL")

    # Jina Reader (for optimized web content extraction)
    jina_api_key: SecretStr | None = Field(
        default=None,
        description="Jina AI API key for higher rate limits (optional, free tier available)",
    )
    use_jina_reader: bool = Field(
        default=True,
        description="Use Jina Reader API for web content extraction (r.jina.ai)",
    )

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

    # Security Settings
    secure_cookies: bool = Field(
        default=True,
        description="Use secure cookies (HTTPS only). Set to False for local development.",
    )

    # Neo4j Graph Database (GraphRAG)
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt protocol URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: SecretStr | None = Field(
        default=None,
        description="Neo4j password",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )

    # Draft Creation Settings
    draft_polling_enabled: bool = Field(
        default=False,
        description="Enable automatic draft creation polling",
    )
    draft_poll_interval_minutes: int = Field(
        default=10,
        description="Interval in minutes for polling Gmail for new threads",
    )
    draft_from_email: str = Field(
        default="support@cyanview.com",
        description="From address for draft emails",
    )
    draft_max_per_run: int = Field(
        default=5,
        description="Maximum drafts to create per scheduled run",
    )
    draft_token_path: Path = Field(
        default=Path("token_drafts.json"),
        description="OAuth token file for draft creation (with compose scope)",
    )

    # Chunking Configuration
    chunk_use_tokens: bool = Field(
        default=True,
        description="Use token-based chunking (vs character-based for backward compatibility)",
    )
    chunk_size_docs: int = Field(
        default=450,
        description="Chunk size for documentation (tokens or chars based on chunk_use_tokens)",
    )
    chunk_size_cases: int = Field(
        default=350,
        description="Chunk size for support cases (tokens or chars based on chunk_use_tokens)",
    )
    chunk_size_default: int = Field(
        default=400,
        description="Default chunk size in tokens (or characters if chunk_use_tokens=false)",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Chunk overlap (tokens or chars). ~12.5% of default size.",
    )
    chunk_adaptive_threshold: int = Field(
        default=200,
        description="Content below this token count stays as single chunk",
    )

    # RIO Terminology Fix Configuration
    rio_fix_on_ingest: bool = Field(
        default=True,
        description="Apply high-confidence RIO terminology fixes during ingestion",
    )
    rio_fix_min_confidence: float = Field(
        default=0.85,
        description="Minimum confidence threshold for auto-applying RIO fixes during ingestion",
    )

    # Prompt Management Configuration
    prompts_cache_ttl: int = Field(
        default=300,
        description="TTL in seconds for prompt cache (default: 5 minutes)",
    )

    # Odoo MCP Integration
    odoo_mcp_enabled: bool = Field(
        default=False,
        description="Enable Odoo MCP integration for CRM/ERP operations",
    )
    odoo_mcp_url: str = Field(
        default="http://localhost:8081",
        description="Odoo MCP server URL (FastMCP streamable-http endpoint)",
    )
    odoo_mcp_api_key: SecretStr | None = Field(
        default=None,
        description="API key for authenticating with Odoo MCP server",
    )
    odoo_mcp_timeout: int = Field(
        default=30,
        description="Timeout in seconds for Odoo MCP requests",
    )
    odoo_mcp_cache_ttl: int = Field(
        default=300,
        description="TTL in seconds for Odoo MCP read operation cache (default: 5 minutes)",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Note: Required fields (API keys) are loaded from environment variables
    by pydantic-settings. The type ignore is needed because mypy doesn't
    understand this behavior.
    """
    return Settings()  # type: ignore[call-arg]
