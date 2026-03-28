"""Tests for product knowledge deduplication in prompts."""

import pytest

from clorag.services.default_prompts import (
    DEFAULT_PROMPTS,
    PRODUCT_ECOSYSTEM_REFERENCE,
    get_default_prompt,
)


class TestProductEcosystemReference:
    """Tests for the shared product_ecosystem reference."""

    def test_constant_contains_product_ecosystem_tags(self) -> None:
        assert "<product_ecosystem>" in PRODUCT_ECOSYSTEM_REFERENCE
        assert "</product_ecosystem>" in PRODUCT_ECOSYSTEM_REFERENCE

    def test_constant_contains_all_products(self) -> None:
        for product in ["RCP", "RCP-J", "CI0", "CI0BM", "RIO", "VP4", "NIO", "RSBM"]:
            assert product in PRODUCT_ECOSYSTEM_REFERENCE

    def test_constant_contains_connection_rules(self) -> None:
        assert "<connection_rules>" in PRODUCT_ECOSYSTEM_REFERENCE
        assert "Sony FX6" in PRODUCT_ECOSYSTEM_REFERENCE

    def test_constant_contains_decision_points(self) -> None:
        assert "<decision_points>" in PRODUCT_ECOSYSTEM_REFERENCE
        assert "REMI" in PRODUCT_ECOSYSTEM_REFERENCE

    def test_registry_entry_exists(self) -> None:
        prompt = get_default_prompt("base.product_reference")
        assert prompt is not None
        assert prompt.key == "base.product_reference"
        assert prompt.category == "base"
        assert prompt.content == PRODUCT_ECOSYSTEM_REFERENCE

    def test_base_system_prompt_embeds_same_content(self) -> None:
        base = get_default_prompt("base.system_prompt")
        assert base is not None
        assert PRODUCT_ECOSYSTEM_REFERENCE in base.content


class TestPromptsUseProductReferenceVariable:
    """Tests that analysis/draft prompts use {product_reference} instead of inline knowledge."""

    @pytest.mark.parametrize(
        "key",
        [
            "analysis.thread_analyzer",
            "analysis.quality_controller",
            "drafts.email_generator",
        ],
    )
    def test_prompt_has_product_reference_variable(self, key: str) -> None:
        prompt = get_default_prompt(key)
        assert prompt is not None
        assert "product_reference" in prompt.variables
        assert "{product_reference}" in prompt.content

    @pytest.mark.parametrize(
        "key",
        [
            "analysis.thread_analyzer",
            "analysis.quality_controller",
            "drafts.email_generator",
        ],
    )
    def test_prompt_no_inline_product_duplication(self, key: str) -> None:
        """Ensure prompts don't contain inline product specs that should come from the variable."""
        prompt = get_default_prompt(key)
        assert prompt is not None
        # These patterns were in the old inline blocks
        assert "CI0: serial-to-IP" not in prompt.content
        assert "CI0: Serial-to-IP converter" not in prompt.content
        assert "RIO: autonomous" not in prompt.content


class TestPromptVariableSubstitution:
    """Test that {product_reference} substitutes correctly in prompts."""

    @pytest.mark.parametrize(
        "key",
        [
            "analysis.thread_analyzer",
            "analysis.quality_controller",
            "drafts.email_generator",
        ],
    )
    def test_product_reference_substitution(self, key: str) -> None:
        prompt = get_default_prompt(key)
        assert prompt is not None
        # Simulate what PromptManager._substitute_variables does
        result = prompt.content.format(
            product_reference=PRODUCT_ECOSYSTEM_REFERENCE,
            # Provide dummy values for other variables
            thread_content="test thread",
            problem_summary="test problem",
            solution_summary="test solution",
            keywords="test",
            category="RCP",
            product="RCP",
            resolution_quality="5",
            anonymized_subject="test subject",
        )
        assert "<product_ecosystem>" in result
        assert "{product_reference}" not in result


class TestRegistryIntegrity:
    """Ensure the registry is consistent after changes."""

    def test_total_prompt_count(self) -> None:
        # Was 12, now 13 with base.product_reference
        assert len(DEFAULT_PROMPTS) == 13

    def test_all_keys_unique(self) -> None:
        keys = [p.key for p in DEFAULT_PROMPTS]
        assert len(keys) == len(set(keys))

    def test_all_variables_present_in_content(self) -> None:
        for prompt in DEFAULT_PROMPTS:
            for var in prompt.variables:
                assert f"{{{var}}}" in prompt.content, (
                    f"Variable '{var}' declared but not found in {prompt.key} content"
                )
