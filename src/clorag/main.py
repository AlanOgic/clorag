"""Main entry point for the CLORAG agent."""

import sys

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from clorag.agent.prompts import get_system_prompt
from clorag.agent.tools import create_rag_tools_server
from clorag.config import get_settings


async def run_agent(prompt: str | None = None) -> None:
    """Run the CLORAG agent.

    Args:
        prompt: Optional initial prompt. If None, enters interactive mode.
    """
    settings = get_settings()

    # Create MCP server with RAG tools
    rag_server = create_rag_tools_server()

    # Configure agent options
    options = ClaudeAgentOptions(
        system_prompt=get_system_prompt(),
        max_turns=settings.max_turns,
        mcp_servers={"rag": rag_server},
        allowed_tools=[
            "mcp__rag__search_docs",
            "mcp__rag__search_cases",
            "mcp__rag__hybrid_search",
        ],
    )

    # Create client
    client = ClaudeSDKClient(options=options)

    if prompt:
        # Single query mode
        async for message in client.process(prompt):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text)
    else:
        # Interactive mode
        print("CLORAG - Multi-RAG Support Assistant")
        print("=" * 40)
        print("Commands: /quit to exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    print("Goodbye!")
                    break

                print("\nCLORAG: ", end="", flush=True)
                async for message in client.process(user_input):
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                print(block.text, end="", flush=True)
                print("\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


def main() -> None:
    """Main entry point."""
    # Check for command line argument
    prompt = None
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    anyio.run(run_agent, prompt)


if __name__ == "__main__":
    main()
