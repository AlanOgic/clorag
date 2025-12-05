"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from clorag.config import Settings, get_settings


class TestSettings:
    """Test Settings class configuration loading."""

    def test_settings_from_env_vars(self) -> None:
        """Test that Settings loads correctly from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "VOYAGE_API_KEY": "test-voyage-key",
            },
            clear=True,
        ):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.anthropic_api_key.get_secret_value() == "test-anthropic-key"
            assert settings.voyage_api_key.get_secret_value() == "test-voyage-key"

    def test_settings_defaults(self, test_settings: Settings) -> None:
        """Test that default values are set correctly."""
        assert test_settings.qdrant_url == "http://localhost:6333"
        assert test_settings.qdrant_docs_collection == "test_docs"
        assert test_settings.qdrant_cases_collection == "test_cases"
        assert test_settings.voyage_model == "voyage-context-3"
        assert test_settings.voyage_dimensions == 1024
        assert test_settings.max_turns == 50

    def test_settings_custom_values(self) -> None:
        """Test that custom values override defaults."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "custom-anthropic",
                "VOYAGE_API_KEY": "custom-voyage",
                "QDRANT_URL": "https://custom.qdrant.io",
                "VOYAGE_MODEL": "voyage-2",
                "VOYAGE_DIMENSIONS": "512",
                "MAX_TURNS": "100",
            },
            clear=True,
        ):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.qdrant_url == "https://custom.qdrant.io"
            assert settings.voyage_model == "voyage-2"
            assert settings.voyage_dimensions == 512
            assert settings.max_turns == 100

    def test_settings_missing_required_fields(self, cleanup_settings_cache: None, tmp_path: Path) -> None:
        """Test that ValidationError is raised when required fields are missing."""
        # Create an empty .env file for testing
        empty_env = tmp_path / "test.env"
        empty_env.write_text("")

        # Prevent reading from actual .env by overriding env_file location
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("clorag.config.Settings.model_config", {"env_file": str(empty_env), "extra": "ignore"})
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()  # type: ignore[call-arg]

            errors = exc_info.value.errors()
            error_fields = {error["loc"][0] for error in errors}
            assert "anthropic_api_key" in error_fields
            assert "voyage_api_key" in error_fields

    def test_secret_str_masking(self, test_settings: Settings) -> None:
        """Test that SecretStr fields are properly masked."""
        # Secret values should not appear in repr
        settings_repr = repr(test_settings)
        assert "test-anthropic-key" not in settings_repr
        assert "test-voyage-key" not in settings_repr

        # But should be accessible via get_secret_value()
        assert test_settings.anthropic_api_key.get_secret_value() == "test-anthropic-key"
        assert test_settings.voyage_api_key.get_secret_value() == "test-voyage-key"

    def test_optional_fields(self, cleanup_settings_cache: None, tmp_path: Path) -> None:
        """Test that optional fields can be None."""
        # Create an empty .env file for testing
        empty_env = tmp_path / "test.env"
        empty_env.write_text("")

        # Prevent reading from actual .env and only set required fields
        with (
            patch.dict(os.environ, {
                "ANTHROPIC_API_KEY": "test-key",
                "VOYAGE_API_KEY": "test-key",
            }, clear=True),
            patch("clorag.config.Settings.model_config", {"env_file": str(empty_env), "extra": "ignore"})
        ):
            settings = Settings()  # type: ignore[call-arg]
            # These fields are optional
            assert settings.qdrant_api_key is None
            assert settings.admin_password is None

    def test_path_fields(self, test_settings: Settings) -> None:
        """Test that Path fields are correctly parsed."""
        assert isinstance(test_settings.google_credentials_path, Path)
        assert isinstance(test_settings.google_token_path, Path)
        assert test_settings.google_credentials_path == Path("test_credentials.json")
        assert test_settings.google_token_path == Path("test_token.json")

    def test_database_paths(self, test_settings: Settings) -> None:
        """Test that database path fields use correct defaults."""
        assert test_settings.database_path == "test_clorag.db"
        assert test_settings.analytics_database_path == "test_analytics.db"

    def test_model_names(self, test_settings: Settings) -> None:
        """Test that model names are configured correctly."""
        assert test_settings.claude_model == "claude-sonnet-4-20250514"
        assert test_settings.haiku_model == "claude-haiku-4-5-20251001"
        assert test_settings.sonnet_model == "claude-sonnet-4-5-20250929"


class TestGetSettings:
    """Test get_settings() caching behavior."""

    def test_get_settings_returns_singleton(
        self, cleanup_settings_cache: None
    ) -> None:
        """Test that get_settings() returns the same instance (cached)."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "VOYAGE_API_KEY": "test-key",
            },
            clear=True,
        ):
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2  # Same instance due to lru_cache

    def test_get_settings_uses_env_vars(
        self, cleanup_settings_cache: None
    ) -> None:
        """Test that get_settings() loads from environment variables."""
        get_settings.cache_clear()  # Ensure clean state
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "env-anthropic-key",
                "VOYAGE_API_KEY": "env-voyage-key",
                "QDRANT_URL": "https://env.qdrant.io",
            },
            clear=True,
        ):
            settings = get_settings()
            assert settings.anthropic_api_key.get_secret_value() == "env-anthropic-key"
            assert settings.voyage_api_key.get_secret_value() == "env-voyage-key"
            assert settings.qdrant_url == "https://env.qdrant.io"

    def test_cache_clear(self, cleanup_settings_cache: None) -> None:
        """Test that cache can be cleared for test isolation."""
        get_settings.cache_clear()  # Start fresh

        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "key1",
                "VOYAGE_API_KEY": "key1",
            },
            clear=True,
        ):
            settings1 = get_settings()
            assert settings1.anthropic_api_key.get_secret_value() == "key1"

        # Clear cache
        get_settings.cache_clear()

        # New environment
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "key2",
                "VOYAGE_API_KEY": "key2",
            },
            clear=True,
        ):
            settings2 = get_settings()
            assert settings2.anthropic_api_key.get_secret_value() == "key2"
            # Different instance after cache clear
            assert settings1 is not settings2


class TestSettingsValidation:
    """Test Settings validation and type conversion."""

    def test_integer_field_validation(self) -> None:
        """Test that integer fields are validated."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "VOYAGE_API_KEY": "test-key",
                "VOYAGE_DIMENSIONS": "invalid",  # Not an integer
            },
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()  # type: ignore[call-arg]

            errors = exc_info.value.errors()
            assert any(
                error["loc"][0] == "voyage_dimensions" and error["type"] == "int_parsing"
                for error in errors
            )

    def test_url_field_types(self, test_settings: Settings) -> None:
        """Test that URL fields are strings."""
        assert isinstance(test_settings.qdrant_url, str)
        assert isinstance(test_settings.docusaurus_url, str)
        assert isinstance(test_settings.searxng_url, str)

    def test_case_insensitive_env_vars(self) -> None:
        """Test that environment variables are case-insensitive."""
        with patch.dict(
            os.environ,
            {
                "anthropic_api_key": "lower-case-key",  # lowercase
                "VOYAGE_API_KEY": "upper-case-key",  # uppercase
            },
            clear=True,
        ):
            settings = Settings()  # type: ignore[call-arg]
            # Both should work due to case_sensitive=False
            assert settings.anthropic_api_key.get_secret_value() == "lower-case-key"
            assert settings.voyage_api_key.get_secret_value() == "upper-case-key"

    def test_extra_fields_ignored(self) -> None:
        """Test that extra environment variables are ignored."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "VOYAGE_API_KEY": "test-key",
                "UNKNOWN_FIELD": "should-be-ignored",
            },
            clear=True,
        ):
            # Should not raise ValidationError due to extra="ignore"
            settings = Settings()  # type: ignore[call-arg]
            assert not hasattr(settings, "unknown_field")
