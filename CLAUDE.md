# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (development)
uv run mcp dev server.py

# Run the MCP server directly
uv run code-memory-local

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .

# Build package
uv build

# Build standalone binary
pyinstaller --clean code-memory.spec
```

## Architecture

`code-memory` is a local, offline MCP server that provides semantic code search using tree-sitter AST parsing, sentence-transformers embeddings, and hybrid retrieval (BM25 + vector search) stored in SQLite.

### Tool Design ("Progressive Disclosure")

Three specialized search tools cover different question types:

| Tool | Question Type | Backend |
|------|--------------|---------|
| `search_code` | Where/What/How (code symbols) | BM25 (FTS5) + Dense Vector (sqlite-vec) with RRF fusion |
| `search_docs` | Architecture/Patterns (docs) | Semantic search over markdown/docstrings |
| `search_history` | Who/Why (git history) | Git + BM25 + Dense Vector |
| `index_codebase` | Setup/prepare | AST parser + sentence-transformers |

### Module Responsibilities

- **`server.py`** — FastMCP server, tool definitions, lazy model warmup
- **`parser.py`** — Tree-sitter AST parsing for Python/JS/TS/Java/Go/Rust/C/C++/Ruby/Kotlin; falls back to whole-file for unsupported languages; respects `.gitignore` and `.code-memoryignore`
- **`db.py`** — SQLite database with three layers: relational tables (files, symbols, references), FTS5 full-text index for BM25, and `sqlite-vec` virtual table for dense vector search. Hosts the lazy-loaded embedding model singleton
- **`queries.py`** — Hybrid retrieval with Reciprocal Rank Fusion (RRF, K=60); BM25 via FTS5; dense search via sqlite-vec
- **`doc_parser.py`** — Markdown section/heading parsing and docstring extraction with semantic chunking
- **`git_search.py`** — GitPython-based git operations (commits, blame, file history)
- **`api_types.py`** — TypedDict definitions for all MCP tool responses
- **`errors.py`** — Custom exception hierarchy
- **`validation.py`** — FTS query sanitization and input validation
- **`logging_config.py`** — Structured logging with RAM tracking

### Key Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace embedding model |
| `CODE_MEMORY_DEVICE` | auto | cuda/mps/cpu/auto |
| `CODE_MEMORY_MAX_WORKERS` | 4 | Parallel indexing threads |
| `CODE_MEMORY_BATCH_FILES` | 200 | Files per indexing batch |
| `CODE_MEMORY_LOG_LEVEL` | INFO | Logging verbosity |
| `CODE_MEMORY_LOG_FILE` | — | Optional log file path |
| `CODE_MEMORY_RERANK` | false | Enable cross-encoder reranking |
| `CODE_MEMORY_DRY_RUN` | — | Path to dump embedding inputs (skips model load) |
| `CODE_MEMORY_EXCLUDE` | — | Glob patterns to exclude from indexing |

### Data Flow

1. `index_codebase(directory)` → `parser.py` parses files with tree-sitter → symbols stored in SQLite (relational + FTS5 + sqlite-vec embeddings)
2. `search_code(query)` → `queries.py` runs parallel BM25 + vector search → RRF fusion → ranked results
3. Results are returned as structured TypedDicts defined in `api_types.py`

### Important Constraints

- Python `>=3.13` required
- Indexing must be called before any search tool (search returns empty results otherwise)
- The embedding model is loaded lazily on first use — first query is slow
- `CODE_MEMORY_DRY_RUN` skips model loading entirely (useful for testing parsers)
