"""Run the RAG search web server."""

import uvicorn


def main() -> None:
    """Run the FastAPI web server."""
    uvicorn.run(
        "clorag.web:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
