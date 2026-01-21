"""System prompts for the CLORAG agent.

Prompts are loaded from the database via PromptManager, with fallback to defaults.
"""

from clorag.services.prompt_manager import get_prompt


def get_system_prompt() -> str:
    """Get the English system prompt for the CLORAG agent."""
    return get_prompt("agent.system_prompt_en")


def get_system_prompt_fr() -> str:
    """Get the French system prompt for the CLORAG agent."""
    return get_prompt("agent.system_prompt_fr")


# Backward compatibility: expose as module-level constants via lazy evaluation
# This allows existing code that does `from clorag.agent.prompts import SYSTEM_PROMPT`
# to continue working, while using the prompt manager under the hood.
class _LazyPrompt:
    """Lazy prompt loader for backward compatibility."""

    def __init__(self, key: str) -> None:
        self._key = key
        self._cached: str | None = None

    def __str__(self) -> str:
        if self._cached is None:
            self._cached = get_prompt(self._key)
        return self._cached

    def __repr__(self) -> str:
        return f"LazyPrompt({self._key})"


# These will load from database when accessed as strings
SYSTEM_PROMPT: str = _LazyPrompt("agent.system_prompt_en")  # type: ignore[assignment]
SYSTEM_PROMPT_FR: str = _LazyPrompt("agent.system_prompt_fr")  # type: ignore[assignment]
