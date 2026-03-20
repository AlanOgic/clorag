"""Run the RAG search web server."""

import os

import uvicorn


def main() -> None:
    """Run the FastAPI web server."""
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8080"))
    uvicorn.run(
        "clorag.web:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
