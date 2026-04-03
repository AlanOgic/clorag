"""System prompts for the CLORAG agent.

Composes the shared base prompt with the CLI-specific tools layer.
"""

from clorag.services.prompt_manager import get_composed_prompt


def get_system_prompt() -> str:
    """Get the composed system prompt for the CLORAG agent."""
    return get_composed_prompt("base.identity", "base.product_reference", "agent.tools_layer")
