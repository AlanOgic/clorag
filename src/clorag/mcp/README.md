# CLORAG MCP Server

MCP (Model Context Protocol) server that exposes CLORAG's RAG capabilities to Claude Desktop and other MCP clients.

## Features

- **Hybrid RAG Search**: Combines semantic search with BM25 keyword matching, plus Voyage reranking
- **Camera Database**: Full-text search across camera compatibility information
- **Custom Documents**: Access admin-managed knowledge documents
- **Support Cases**: Search and retrieve past support case resolutions

## Installation

The MCP server is included with CLORAG. Ensure dependencies are installed:

```bash
cd /path/to/clorag
uv sync
```

## Claude Desktop Configuration

Add to `~/.config/claude/claude_desktop_config.json` (Linux/macOS) or equivalent:

```json
{
  "mcpServers": {
    "clorag": {
      "command": "uv",
      "args": ["--directory", "/path/to/clorag", "run", "clorag-mcp"]
    }
  }
}
```

On macOS, the config file is at `~/Library/Application Support/Claude/claude_desktop_config.json`.

## Available Tools

### Search Tools

| Tool | Description |
|------|-------------|
| `search` | Hybrid RAG search across all sources (docs, cases, custom) with reranking |
| `search_docs` | Search official Docusaurus documentation only |
| `search_cases` | Search past Gmail support cases |

### Camera Tools

| Tool | Description |
|------|-------------|
| `search_cameras` | FTS5 search across camera database |
| `get_camera` | Get camera details by ID |
| `find_related_cameras` | Find similar cameras by manufacturer/protocol/ports |
| `compare_cameras` | Side-by-side comparison of multiple cameras |
| `list_cameras` | List cameras with optional filtering |
| `get_camera_stats` | Database statistics |

### Document Tools

| Tool | Description |
|------|-------------|
| `list_documents` | List custom documents with category filtering |
| `get_document` | Get full document content by ID |
| `get_document_categories` | Get available document categories |

### Support Case Tools

| Tool | Description |
|------|-------------|
| `search_support_cases` | FTS5 search across support cases |
| `get_support_case` | Get full case details including resolution |
| `list_support_cases` | List cases with category/product filtering |
| `get_support_stats` | Support case statistics |

## Testing

Run the server standalone to verify it starts correctly:

```bash
uv run clorag-mcp
```

The server communicates via STDIO. You should see no output if everything is working.

## Environment Variables

The MCP server uses the same environment variables as the main CLORAG application:

- `ANTHROPIC_API_KEY` - Required for some operations
- `VOYAGE_API_KEY` - Required for embeddings and reranking
- `QDRANT_URL` - Vector database URL
- `DATABASE_PATH` - SQLite database path (cameras, support cases)

See the main CLORAG `.env.example` for all configuration options.

## Architecture

```
MCP Server (stdio)
    └── FastMCP
        ├── Search Tools
        │   └── MultiSourceRetriever
        │       ├── EmbeddingsClient (voyage-context-3)
        │       ├── SparseEmbeddingsClient (BM25)
        │       ├── VectorStore (Qdrant)
        │       └── RerankerClient (voyage rerank-2.5)
        ├── Camera Tools
        │   └── CameraDatabase (SQLite + FTS5)
        ├── Document Tools
        │   └── CustomDocumentService
        └── Support Tools
            └── SupportCaseDatabase (SQLite + FTS5)
```

## Development

To modify or add tools, see:

- `src/clorag/mcp/server.py` - Main server setup and lifespan
- `src/clorag/mcp/tools/` - Tool implementations by category
