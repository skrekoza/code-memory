"""
Database layer for code-memory.

Manages a local SQLite database with three storage layers:
  1. Relational tables (files, symbols, references)
  2. FTS5 full-text index (symbols_fts) for BM25 keyword search
  3. sqlite-vec virtual table (symbol_embeddings) for dense vector search

All writes use upsert semantics so re-indexing is idempotent.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import sqlite_vec
import xxhash

import logging_config

if TYPE_CHECKING:
    pass

logger = logging_config.setup_logging()

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_model = None
_embedding_dim = None
_remote_client = None  # openai.OpenAI singleton for remote provider

# Model identifier - can be overridden via EMBEDDING_MODEL environment variable
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Pin to a specific HuggingFace commit hash for reproducibility and supply-chain security.
# Set EMBEDDING_MODEL_REVISION to a full commit SHA (e.g. "abc1234...") to lock the model.
EMBEDDING_MODEL_REVISION = os.environ.get("EMBEDDING_MODEL_REVISION", None)

# Device selection - can be overridden via CODE_MEMORY_DEVICE environment variable
# Options: 'cuda', 'mps', 'cpu', or 'auto' (default)
CODE_MEMORY_DEVICE = os.environ.get("CODE_MEMORY_DEVICE", "auto")

# Cross-encoder reranking - enabled by default for improved precision
# Set CODE_MEMORY_RERANK=false to disable if latency is a concern
CODE_MEMORY_RERANK = os.environ.get("CODE_MEMORY_RERANK", "false").lower() in ("true", "1", "yes")

# Default cross-encoder model for reranking
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)
# Pin to a specific HuggingFace commit hash for the reranking model.
RERANK_MODEL_REVISION = os.environ.get("RERANK_MODEL_REVISION", None)

# ---------------------------------------------------------------------------
# Dry-run mode: capture embedding inputs without loading or calling the model
# ---------------------------------------------------------------------------
# Set CODE_MEMORY_DRY_RUN to a file path to enable dry-run mode.
# All embedding inputs are written to that file (JSONL, one record per call)
# and zero vectors of CODE_MEMORY_DRY_RUN_DIM dimensions are returned instead.
# The embedding model is never loaded in this mode.
DRY_RUN_OUTPUT_PATH: str = os.environ.get("CODE_MEMORY_DRY_RUN", "")
DRY_RUN_EMBEDDING_DIM: int = int(os.environ.get("CODE_MEMORY_DRY_RUN_DIM", "1024"))

# ---------------------------------------------------------------------------
# Remote embedding provider configuration (OpenAI-compatible APIs)
# ---------------------------------------------------------------------------
# Set EMBEDDING_PROVIDER=openai to route embedding calls to a remote server
# such as LM Studio, Ollama, or vLLM instead of loading a local SentenceTransformer.
EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER", "local")

# Base URL of the OpenAI-compatible embeddings endpoint.
# LM Studio default: http://localhost:1234/v1
EMBEDDING_API_BASE: str = os.environ.get("EMBEDDING_API_BASE", "http://localhost:1234/v1")

# API key forwarded in the Authorization header.
# LM Studio accepts any non-empty string; use your real key for cloud APIs.
EMBEDDING_API_KEY: str = os.environ.get("EMBEDDING_API_KEY", "lm-studio")

# Expected embedding dimension from the remote model.
# When 0 (default) the dimension is probed automatically on first use via a
# single test embedding call.
EMBEDDING_API_DIM: int = int(os.environ.get("EMBEDDING_API_DIM", "0"))

# Task-type prefix behaviour: "auto" | "true" | "false"
# "auto"  → prefix applied for local models (Qwen/Jina style), skipped for remote
# "true"  → always prepend "{task_type}: " to every input
# "false" → never prepend the prefix
EMBEDDING_TASK_PREFIX: str = os.environ.get("EMBEDDING_TASK_PREFIX", "auto")

_dry_run_lock = threading.Lock()

if DRY_RUN_OUTPUT_PATH:
    logging.getLogger(__name__).warning(
        "Dry-run mode enabled: embedding model calls are suppressed. "
        "Inputs will be written to '%s' (dim=%d).",
        DRY_RUN_OUTPUT_PATH,
        DRY_RUN_EMBEDDING_DIM,
    )


def _dry_run_record(call_type: str, task_type: str, texts: list[str]) -> None:
    """Append one embedding call record to the dry-run output file (JSONL)."""
    import datetime

    record = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "call": call_type,
        "task_type": task_type,
        "count": len(texts),
        "texts": texts,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _dry_run_lock:
        with open(DRY_RUN_OUTPUT_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)


# Check for bundled model (used in PyInstaller builds)
_BUNDLED_MODEL_PATH = None
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    _BUNDLED_MODEL_PATH = os.path.join(sys._MEIPASS, 'bundled_model')


def _should_use_task_prefix() -> bool:
    """Return True if task-type prefixes should be prepended to embedding inputs.

    Controlled by EMBEDDING_TASK_PREFIX env var:
      "auto"  → prefix for local models (Qwen/Jina expect it), skip for remote
      "true"  → always add prefix
      "false" → never add prefix
    """
    if EMBEDDING_TASK_PREFIX == "true":
        return True
    if EMBEDDING_TASK_PREFIX == "false":
        return False
    # "auto": local models use the Qwen-style task prefix; remote models generally don't
    return EMBEDDING_PROVIDER == "local"


def get_remote_client():
    """Lazy-load and cache the OpenAI-compatible HTTP client for remote embeddings.

    Requires the ``openai`` package (``uv add openai`` or ``pip install openai``).
    Controlled by EMBEDDING_API_BASE and EMBEDDING_API_KEY env vars.

    Raises:
        ImportError: if the ``openai`` package is not installed.
    """
    global _remote_client
    if _remote_client is None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required when EMBEDDING_PROVIDER=openai. "
                "Install it with: uv add openai"
            ) from exc
        _remote_client = _openai.OpenAI(
            base_url=EMBEDDING_API_BASE,
            api_key=EMBEDDING_API_KEY,
        )
        logger.info(
            "Remote embedding client initialised: base_url=%s, model=%s",
            EMBEDDING_API_BASE,
            EMBEDDING_MODEL_NAME,
        )
    return _remote_client


def _detect_device() -> str:
    """Detect the best available device for embedding computation.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    import torch

    device_override = CODE_MEMORY_DEVICE.lower()

    # Handle manual override
    if device_override in ('cuda', 'mps', 'cpu'):
        if device_override == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        if device_override == 'mps' and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return 'cpu'
        return device_override

    # Auto-detect (default behavior)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA GPU detected: {device_name}")
        return 'cuda'

    if torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) detected")
        return 'mps'

    return 'cpu'


def get_embedding_model(force_cpu: bool = False):
    """Lazy-load and cache the sentence-transformers model.

    Automatically uses GPU acceleration when available (CUDA or MPS).
    Set CODE_MEMORY_DEVICE env var to 'cuda', 'mps', 'cpu', or 'auto'.

    Args:
        force_cpu: If True, force the model to use CPU even if GPU is available.
                   This is useful when GPU memory is constrained (CUDA OOM).
                   If the model is already loaded on GPU, it will be moved to CPU.
    """
    global _model, _embedding_dim

    # Handle force_cpu after model is already loaded
    if _model is not None and force_cpu:
        current_device = str(_model.device)
        if 'cuda' in current_device or 'mps' in current_device:
            logger.info(f"Moving embedding model from {current_device} to CPU (force_cpu=True)")
            _model = _model.to('cpu')
        return _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        # Detect and use the best available device, or force CPU
        if force_cpu:
            device = 'cpu'
            logger.info("Using CPU for embedding computation (force_cpu=True)")
        else:
            device = _detect_device()

        # Use bundled model if available (PyInstaller build)
        model_path = _BUNDLED_MODEL_PATH if _BUNDLED_MODEL_PATH else EMBEDDING_MODEL_NAME
        revision = None if _BUNDLED_MODEL_PATH else EMBEDDING_MODEL_REVISION
        _model = SentenceTransformer(
            model_path, trust_remote_code=False, device=device, revision=revision
        )

        if device != 'cpu':
            logger.info(f"Embedding model loaded on {device.upper()} for acceleration")
        else:
            logger.info("Using CPU for embedding computation")

        # Cache the embedding dimension from the model
        _embedding_dim = _model.get_sentence_embedding_dimension()

        import logging_config
        logger.info(
            f"Loaded embedding model '{EMBEDDING_MODEL_NAME}' with dimension: {_embedding_dim} "
            f"[RAM peak: {logging_config.get_ram_mb():.0f} MB]"
        )
    return _model


def get_embedding_dim() -> int:
    """Get the embedding dimension.

    For local provider: loads the SentenceTransformer model if not already loaded.
    For remote provider: returns EMBEDDING_API_DIM if set, otherwise probes the
    endpoint with a single test embedding call to discover the dimension.
    In dry-run mode returns DRY_RUN_EMBEDDING_DIM without any model loading.
    """
    global _embedding_dim
    if DRY_RUN_OUTPUT_PATH:
        return DRY_RUN_EMBEDDING_DIM
    if _embedding_dim is not None:
        return _embedding_dim
    if EMBEDDING_PROVIDER == "openai":
        if EMBEDDING_API_DIM:
            _embedding_dim = EMBEDDING_API_DIM
            return _embedding_dim
        # Probe the remote endpoint to discover the vector dimension
        logger.info("Probing remote embedding endpoint for vector dimension...")
        client = get_remote_client()
        resp = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=["probe"])
        _embedding_dim = len(resp.data[0].embedding)
        logger.info("Remote embedding dimension detected: %d", _embedding_dim)
        return _embedding_dim
    # Local path: load model (sets _embedding_dim as a side effect)
    get_embedding_model()
    return _embedding_dim  # type: ignore[return-value]


def embed_text(text: str, task_type: str = "nl2code") -> list[float]:
    """Generate a dense vector embedding for *text*.

    Dispatches to the local SentenceTransformer model or a remote
    OpenAI-compatible endpoint depending on EMBEDDING_PROVIDER.

    Args:
        text: The text to embed.
        task_type: One of 'nl2code', 'code2code', 'code2nl', 'code2completion', 'qa'.
            Only prepended as a prefix when _should_use_task_prefix() is True.
    """
    if DRY_RUN_OUTPUT_PATH:
        prefixed_text = f"{task_type}: {text}" if _should_use_task_prefix() else text
        _dry_run_record("embed_text", task_type, [prefixed_text])
        return [0.0] * DRY_RUN_EMBEDDING_DIM

    if EMBEDDING_PROVIDER == "openai":
        client = get_remote_client()
        text_input = f"{task_type}: {text}" if _should_use_task_prefix() else text
        resp = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=[text_input])
        return resp.data[0].embedding

    model = get_embedding_model()
    prefixed_text = f"{task_type}: {text}"
    vec = model.encode(prefixed_text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


def embed_texts_batch(
    texts: list[str], batch_size: int = 32, task_type: str = "nl2code"
) -> np.ndarray:
    """Generate embeddings for multiple texts at once.

    This is significantly faster than calling embed_text() in a loop
    because sentence-transformers is optimized for batch processing.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to process per batch (default 32).
        task_type: One of 'nl2code', 'code2code', 'code2nl', 'code2completion', 'qa'.

    Returns:
        2-D float32 numpy array of shape (len(texts), embedding_dim).
        Returning numpy avoids the 7× memory overhead of converting each
        row to a Python list of Python floats.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    use_prefix = _should_use_task_prefix()

    if DRY_RUN_OUTPUT_PATH:
        prefixed_texts = [f"{task_type}: {t}" for t in texts] if use_prefix else list(texts)
        _dry_run_record("embed_texts_batch", task_type, prefixed_texts)
        return np.zeros((len(texts), DRY_RUN_EMBEDDING_DIM), dtype=np.float32)

    if EMBEDDING_PROVIDER == "openai":
        client = get_remote_client()
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = [f"{task_type}: {t}" for t in batch] if use_prefix else list(batch)
            resp = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=inputs)
            resp.data.sort(key=lambda e: e.index)
            all_vectors.extend(e.embedding for e in resp.data)
        result = np.array(all_vectors, dtype=np.float32)
        import logging_config
        logger.debug(
            "embed_texts_batch (remote): encoded %d texts [RAM peak: %.0f MB]",
            len(texts),
            logging_config.get_ram_mb(),
        )
        return result

    model = get_embedding_model()

    # Add task prefix to all texts
    prefixed_texts = [f"{task_type}: {text}" for text in texts]

    # Batch encode with normalization — returns a float32 numpy 2-D array
    vectors = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    import logging_config
    logger.debug(
        f"embed_texts_batch: encoded {len(texts)} texts "
        f"[RAM peak: {logging_config.get_ram_mb():.0f} MB]"
    )

    return vectors


def warmup_embedding_model(force_cpu: bool = False) -> None:
    """Pre-load and warm up the embedding model.

    Call this at server startup to avoid cold-start latency on first search.
    The warmup encodes a dummy string to initialize internal tensors.
    No-op in dry-run mode.

    Args:
        force_cpu: If True, force the model to use CPU even if GPU is available.
                   Useful when GPU memory is constrained (CUDA OOM).
    """
    if DRY_RUN_OUTPUT_PATH:
        logger.info("Dry-run mode: skipping embedding model warmup")
        return
    if EMBEDDING_PROVIDER == "openai":
        try:
            client = get_remote_client()
            client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=["warmup"])
            logger.info("Remote embedding endpoint reachable: %s", EMBEDDING_API_BASE)
        except Exception as exc:
            logger.warning("Remote embedding endpoint warmup failed: %s", exc)
        return
    model = get_embedding_model(force_cpu=force_cpu)
    # Warmup encode to initialize lazy-loaded components
    model.encode("nl2code: warmup", normalize_embeddings=True, show_progress_bar=False)
    logger.info("Embedding model warmed up")


# ---------------------------------------------------------------------------
# Cross-encoder reranking model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_rerank_model = None


def get_rerank_model():
    """Lazy-load and cache the cross-encoder reranking model.

    Only loads the model if CODE_MEMORY_RERANK is enabled.
    Uses the same device as the embedding model.

    Returns:
        CrossEncoder model instance, or None if reranking is disabled.
    """
    global _rerank_model

    if not CODE_MEMORY_RERANK:
        return None

    if _rerank_model is None:
        try:
            from sentence_transformers import CrossEncoder

            # Use the same device as the embedding model
            device = _detect_device() if _model is None else str(_model.device).split(':')[0]

            logger.info(f"Loading cross-encoder reranking model: {RERANK_MODEL_NAME}")
            _rerank_model = CrossEncoder(
                RERANK_MODEL_NAME, device=device, trust_remote_code=False,
                revision=RERANK_MODEL_REVISION,
            )
            logger.info(f"Cross-encoder model loaded on {device}")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model: {e}. Reranking disabled.")
            return None

    return _rerank_model


def rerank_results(query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
    """Rerank search results using cross-encoder for improved precision.

    Cross-encoders process query-document pairs jointly, producing more accurate
    relevance scores than bi-encoders alone.

    Args:
        query: The original search query.
        results: List of search result dicts, each containing at least 'source_text'
                 or 'content' field for reranking.
        top_k: Optional number of top results to return after reranking.
               If None, returns all results in new order.

    Returns:
        Reranked list of results sorted by cross-encoder relevance scores.
        If reranking fails or is disabled, returns original results unchanged.
    """
    if not results:
        return results

    model = get_rerank_model()
    if model is None:
        return results[:top_k] if top_k else results

    try:
        # Build query-document pairs for cross-encoder
        pairs = []
        for r in results:
            # Use source_text for code, content for documentation
            doc_text = r.get("source_text") or r.get("content", "")
            if doc_text:
                # Truncate long documents to avoid token limit issues
                if len(doc_text) > 2000:
                    doc_text = doc_text[:2000]
                pairs.append([query, doc_text])
            else:
                pairs.append([query, ""])  # Fallback for empty content

        # Get cross-encoder scores
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort by descending relevance
        for i, r in enumerate(results):
            r["_rerank_score"] = float(scores[i])

        reranked = sorted(results, key=lambda x: x.get("_rerank_score", 0), reverse=True)

        # Clean up internal score from results
        for r in reranked:
            r.pop("_rerank_score", None)

        return reranked[:top_k] if top_k else reranked

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original results.")
        return results[:top_k] if top_k else results


def is_reranking_enabled() -> bool:
    """Check if cross-encoder reranking is enabled and model is available."""
    return CODE_MEMORY_RERANK and get_rerank_model() is not None


# ---------------------------------------------------------------------------
# Transaction support
# ---------------------------------------------------------------------------


@contextmanager
def transaction(db: sqlite3.Connection):
    """Context manager for explicit transaction control.

    Disables autocommit, yields control, then commits on success.
    On exception, rolls back automatically.

    Example:
        with transaction(db):
            for item in items:
                upsert_symbol(db, ..., auto_commit=False)
        # Single commit here
    """
    # Disable autocommit by starting a transaction
    db.execute("BEGIN")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- 0. Metadata table for tracking index version and model info
CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 1. Tracked source files
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY,
    path          TEXT    UNIQUE NOT NULL,
    last_modified REAL   NOT NULL,
    file_hash     TEXT   NOT NULL
);

-- 2. Parsed AST symbols
CREATE TABLE IF NOT EXISTS symbols (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL,
    kind             TEXT    NOT NULL,
    file_id          INTEGER NOT NULL REFERENCES files(id),
    line_start       INTEGER NOT NULL,
    line_end         INTEGER NOT NULL,
    parent_symbol_id INTEGER,
    source_text      TEXT    NOT NULL,
    UNIQUE(file_id, name, kind, line_start)
);

-- 3. FTS5 content-sync'd to symbols (indexes name + source_text)
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    source_text,
    content=symbols,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with symbols table
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, source_text)
    VALUES (new.id, new.name, new.source_text);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, source_text)
    VALUES ('delete', old.id, old.name, old.source_text);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, source_text)
    VALUES ('delete', old.id, old.name, old.source_text);
    INSERT INTO symbols_fts(rowid, name, source_text)
    VALUES (new.id, new.name, new.source_text);
END;

-- 5. Cross-reference tracking
CREATE TABLE IF NOT EXISTS references_ (
    id          INTEGER PRIMARY KEY,
    symbol_name TEXT    NOT NULL,
    file_id     INTEGER NOT NULL REFERENCES files(id),
    line_number INTEGER NOT NULL,
    UNIQUE(symbol_name, file_id, line_number)
);

-- Index on symbol name for fast existence checks during reference insertion
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

-- ---------------------------------------------------------------------------
-- Documentation tables (Milestone 4)
-- ---------------------------------------------------------------------------

-- 6. Tracked documentation files
CREATE TABLE IF NOT EXISTS doc_files (
    id            INTEGER PRIMARY KEY,
    path          TEXT    UNIQUE NOT NULL,
    last_modified REAL   NOT NULL,
    file_hash     TEXT   NOT NULL,
    doc_type      TEXT   NOT NULL  -- 'markdown', 'readme', 'docstring'
);

-- 7. Chunked documentation content
CREATE TABLE IF NOT EXISTS doc_chunks (
    id            INTEGER PRIMARY KEY,
    doc_file_id   INTEGER NOT NULL REFERENCES doc_files(id),
    chunk_index   INTEGER NOT NULL,
    section_title TEXT,
    content       TEXT    NOT NULL,
    line_start    INTEGER NOT NULL,
    line_end      INTEGER NOT NULL,
    UNIQUE(doc_file_id, chunk_index)
);

-- 8. FTS5 for documentation chunks (BM25 keyword search)
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
    content,
    section_title,
    content=doc_chunks,
    content_rowid=id
);

-- Triggers to keep doc FTS5 in sync
CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
    INSERT INTO doc_chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;
"""


def get_db(project_dir: str) -> sqlite3.Connection:
    """Open (or create) the database, load sqlite-vec, and ensure schema.

    The database is stored as {project_dir}/code_memory.db to ensure each
    project has its own isolated index.

    If the embedding model has changed since the last index, all indexed data
    is automatically invalidated and the index will need to be rebuilt.

    Args:
        project_dir: The project directory where code_memory.db will be stored.

    Returns:
        A ready-to-use ``sqlite3.Connection`` with WAL mode and foreign keys.
    """
    import os
    db_path = os.path.join(os.path.abspath(project_dir), "code_memory.db")
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    db.executescript(_SCHEMA_SQL)

    # Check if the embedding model has changed
    stored_model = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_model'"
    ).fetchone()
    stored_dim = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_dim'"
    ).fetchone()

    # Only load the model if we don't have matching stored metadata yet
    if stored_model and stored_model[0] == EMBEDDING_MODEL_NAME and stored_dim:
        embedding_dim = int(stored_dim[0])
        model_changed = False
    else:
        # Get embedding dimension from the model (loads model if needed)
        embedding_dim = get_embedding_dim()
        model_changed = True

    if model_changed:
        if stored_model is not None:
            # Model changed - invalidate existing index
            logger.info(
                f"Embedding model changed from '{stored_model[0] if stored_model else 'none'}' "
                f"to '{EMBEDDING_MODEL_NAME}'. Invalidating index..."
            )
            _invalidate_index(db, embedding_dim)
        else:
            # New database - just create the embedding tables
            _create_embedding_tables(db, embedding_dim)

        # Store the current model info
        db.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('embedding_model', ?)",
            (EMBEDDING_MODEL_NAME,)
        )
        db.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('embedding_dim', ?)",
            (str(embedding_dim),)
        )
        db.commit()

    return db


def _invalidate_index(db: sqlite3.Connection, embedding_dim: int) -> None:
    """Invalidate the index by clearing all data and recreating embedding tables.

    This is called when the embedding model changes.
    """
    # Drop existing embedding virtual tables
    db.execute("DROP TABLE IF EXISTS symbol_embeddings")
    db.execute("DROP TABLE IF EXISTS doc_embeddings")

    # Clear all indexed data (cascades will handle related data via foreign keys,
    # but we need to be explicit since FK enforcement may vary)
    db.execute("DELETE FROM symbols")
    db.execute("DELETE FROM files")
    db.execute("DELETE FROM references_")
    db.execute("DELETE FROM doc_chunks")
    db.execute("DELETE FROM doc_files")

    # Recreate embedding tables with new dimension
    _create_embedding_tables(db, embedding_dim)
    logger.info("Index invalidated and embedding tables recreated")


def _create_embedding_tables(db: sqlite3.Connection, embedding_dim: int) -> None:
    """Create the embedding virtual tables with the specified dimension."""
    # sqlite-vec virtual table for code embeddings
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_embeddings
        USING vec0(
            symbol_id INTEGER PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
        """
    )

    # sqlite-vec virtual table for documentation embeddings
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings
        USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
        """
    )


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def file_hash(filepath: str) -> str:
    """Compute fast non-cryptographic hash of a file's contents.

    Uses xxHash (xxh64) which is ~10x faster than SHA-256 while still
    providing excellent collision resistance for change detection.

    Args:
        filepath: Path to the file to hash.

    Returns:
        Hexadecimal string representation of the 64-bit hash.
    """
    h = xxhash.xxh64()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_file(
    db: sqlite3.Connection,
    path: str,
    last_modified: float,
    fhash: str,
    auto_commit: bool = True,
) -> int:
    """Insert or update a file record. Returns the file_id."""
    db.execute(
        """
        INSERT INTO files (path, last_modified, file_hash)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            last_modified = excluded.last_modified,
            file_hash     = excluded.file_hash
        """,
        (path, last_modified, fhash),
    )
    if auto_commit:
        db.commit()
    # Fetch the id (needed because last_insert_rowid isn't reliable on update)
    row = db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_file_data(db: sqlite3.Connection, file_id: int, auto_commit: bool = True) -> None:
    """Remove all symbols, embeddings, and references for a file.

    This is called before re-indexing to guarantee idempotency.
    """
    # Collect symbol ids for embedding cleanup
    sym_ids = [
        r[0] for r in db.execute("SELECT id FROM symbols WHERE file_id = ?", (file_id,)).fetchall()
    ]
    if sym_ids:
        placeholders = ",".join("?" * len(sym_ids))
        db.execute(f"DELETE FROM symbol_embeddings WHERE symbol_id IN ({placeholders})", sym_ids)

    db.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
    db.execute("DELETE FROM references_ WHERE file_id = ?", (file_id,))
    if auto_commit:
        db.commit()


def upsert_symbol(
    db: sqlite3.Connection,
    name: str,
    kind: str,
    file_id: int,
    line_start: int,
    line_end: int,
    parent_symbol_id: int | None,
    source_text: str,
    auto_commit: bool = True,
) -> int:
    """Insert or update a symbol record. Returns the symbol_id."""
    db.execute(
        """
        INSERT INTO symbols (name, kind, file_id, line_start, line_end,
                             parent_symbol_id, source_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id, name, kind, line_start) DO UPDATE SET
            line_end         = excluded.line_end,
            parent_symbol_id = excluded.parent_symbol_id,
            source_text      = excluded.source_text
        """,
        (name, kind, file_id, line_start, line_end, parent_symbol_id, source_text),
    )
    if auto_commit:
        db.commit()
    row = db.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = ? AND kind = ? AND line_start = ?",
        (file_id, name, kind, line_start),
    ).fetchone()
    return row[0]


def upsert_reference(
    db: sqlite3.Connection,
    symbol_name: str,
    file_id: int,
    line_number: int,
    auto_commit: bool = True,
) -> None:
    """Insert a cross-reference record only if the symbol is defined in this codebase.

    The conditional INSERT filters out references to stdlib, external, and
    language-builtin symbols by requiring the name to exist in the symbols
    table.  This eliminates high-frequency noise entries (e.g. ``String``,
    ``List``, ``java``) that degrade search quality and inflate database size.

    Note: cross-file references are resolved against whatever symbols are
    already present in the DB at insertion time.  Re-indexing after a full
    initial index will pick up any references that were missed on the first
    pass due to ordering.
    """
    db.execute(
        """
        INSERT OR IGNORE INTO references_ (symbol_name, file_id, line_number)
        SELECT ?, ?, ?
        WHERE EXISTS (SELECT 1 FROM symbols WHERE name = ? LIMIT 1)
        """,
        (symbol_name, file_id, line_number, symbol_name),
    )
    if auto_commit:
        db.commit()


def upsert_embedding(
    db: sqlite3.Connection,
    symbol_id: int,
    embedding: np.ndarray | list[float],
    auto_commit: bool = True,
) -> None:
    """Insert or replace a symbol's dense vector embedding."""
    if isinstance(embedding, np.ndarray):
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
    else:
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)
    # sqlite-vec doesn't support ON CONFLICT, so delete-then-insert
    db.execute("DELETE FROM symbol_embeddings WHERE symbol_id = ?", (symbol_id,))
    db.execute(
        "INSERT INTO symbol_embeddings (symbol_id, embedding) VALUES (?, ?)",
        (symbol_id, blob),
    )
    if auto_commit:
        db.commit()


# ---------------------------------------------------------------------------
# Documentation upsert helpers (Milestone 4)
# ---------------------------------------------------------------------------


def upsert_doc_file(
    db: sqlite3.Connection,
    path: str,
    last_modified: float,
    fhash: str,
    doc_type: str,
    auto_commit: bool = True,
) -> int:
    """Insert or update a documentation file record. Returns doc_file_id."""
    db.execute(
        """
        INSERT INTO doc_files (path, last_modified, file_hash, doc_type)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            last_modified = excluded.last_modified,
            file_hash     = excluded.file_hash,
            doc_type      = excluded.doc_type
        """,
        (path, last_modified, fhash, doc_type),
    )
    if auto_commit:
        db.commit()
    row = db.execute("SELECT id FROM doc_files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_doc_file_data(db: sqlite3.Connection, doc_file_id: int, auto_commit: bool = True) -> None:
    """Remove all chunks and embeddings for a documentation file.

    This is called before re-indexing to guarantee idempotency.
    """
    # Collect chunk ids for embedding cleanup
    chunk_ids = [
        r[0]
        for r in db.execute(
            "SELECT id FROM doc_chunks WHERE doc_file_id = ?", (doc_file_id,)
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        db.execute(f"DELETE FROM doc_embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)

    db.execute("DELETE FROM doc_chunks WHERE doc_file_id = ?", (doc_file_id,))
    if auto_commit:
        db.commit()


def upsert_doc_chunk(
    db: sqlite3.Connection,
    doc_file_id: int,
    chunk_index: int,
    section_title: str | None,
    content: str,
    line_start: int,
    line_end: int,
    auto_commit: bool = True,
) -> int:
    """Insert or update a documentation chunk. Returns chunk_id."""
    db.execute(
        """
        INSERT INTO doc_chunks (doc_file_id, chunk_index, section_title,
                               content, line_start, line_end)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_file_id, chunk_index) DO UPDATE SET
            section_title = excluded.section_title,
            content       = excluded.content,
            line_start    = excluded.line_start,
            line_end      = excluded.line_end
        """,
        (doc_file_id, chunk_index, section_title, content, line_start, line_end),
    )
    if auto_commit:
        db.commit()
    row = db.execute(
        "SELECT id FROM doc_chunks WHERE doc_file_id = ? AND chunk_index = ?",
        (doc_file_id, chunk_index),
    ).fetchone()
    return row[0]


def upsert_doc_embedding(
    db: sqlite3.Connection,
    chunk_id: int,
    embedding: np.ndarray | list[float],
    auto_commit: bool = True,
) -> None:
    """Insert or replace a documentation chunk's dense vector embedding."""
    if isinstance(embedding, np.ndarray):
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
    else:
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)
    db.execute("DELETE FROM doc_embeddings WHERE chunk_id = ?", (chunk_id,))
    db.execute(
        "INSERT INTO doc_embeddings (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, blob),
    )
    if auto_commit:
        db.commit()


# ---------------------------------------------------------------------------
# Index Statistics
# ---------------------------------------------------------------------------

def get_index_stats(db: sqlite3.Connection, project_dir: str) -> dict:
    """Get comprehensive statistics about the index.

    Args:
        db: An open sqlite3.Connection.
        project_dir: The project directory path.

    Returns:
        Dictionary with index health metrics including:
        - Total symbols, files, doc chunks indexed
        - Index freshness (last indexed timestamps)
        - Embedding model info and dimension
        - Database size and WAL status
    """
    import os

    # Get counts
    symbols_count = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    files_count = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    doc_chunks_count = db.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
    doc_files_count = db.execute("SELECT COUNT(*) FROM doc_files").fetchone()[0]
    references_count = db.execute("SELECT COUNT(*) FROM references_").fetchone()[0]
    symbol_embeddings_count = db.execute("SELECT COUNT(*) FROM symbol_embeddings").fetchone()[0]
    doc_embeddings_count = db.execute("SELECT COUNT(*) FROM doc_embeddings").fetchone()[0]

    # Get symbol kinds distribution
    symbol_kinds = dict(db.execute(
        "SELECT kind, COUNT(*) FROM symbols GROUP BY kind ORDER BY COUNT(*) DESC"
    ).fetchall())

    # Get file types distribution (by extension)
    file_extensions = dict(db.execute(
        """SELECT substr(path, instr(path, '.')) as ext, COUNT(*) as cnt
           FROM files
           WHERE path LIKE '%.%'
           GROUP BY ext
           ORDER BY cnt DESC
           LIMIT 10"""
    ).fetchall())

    # Get last indexed timestamps
    last_file_indexed = db.execute(
        "SELECT MAX(last_modified) FROM files"
    ).fetchone()[0]
    last_doc_indexed = db.execute(
        "SELECT MAX(last_modified) FROM doc_files"
    ).fetchone()[0]

    # Get embedding model info
    embedding_model = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_model'"
    ).fetchone()
    embedding_dim = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_dim'"
    ).fetchone()

    # Database file size
    db_path = os.path.join(os.path.abspath(project_dir), "code_memory.db")
    db_size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    db_size_mb = round(db_size_bytes / (1024 * 1024), 2)

    # WAL status
    wal_path = db_path + "-wal"
    wal_exists = os.path.exists(wal_path)
    wal_size_mb = round(os.path.getsize(wal_path) / (1024 * 1024), 2) if wal_exists else 0

    # Check journal mode
    journal_mode = db.execute("PRAGMA journal_mode").fetchone()[0]

    return {
        "indexed": symbols_count > 0 or doc_chunks_count > 0,
        "counts": {
            "symbols": symbols_count,
            "files": files_count,
            "doc_chunks": doc_chunks_count,
            "doc_files": doc_files_count,
            "references": references_count,
            "symbol_embeddings": symbol_embeddings_count,
            "doc_embeddings": doc_embeddings_count,
        },
        "distributions": {
            "symbol_kinds": symbol_kinds,
            "file_extensions": file_extensions,
        },
        "freshness": {
            "last_file_indexed": last_file_indexed,
            "last_doc_indexed": last_doc_indexed,
        },
        "embedding": {
            "model": embedding_model[0] if embedding_model else None,
            "dimension": int(embedding_dim[0]) if embedding_dim else None,
            "device": (
                f"remote({EMBEDDING_API_BASE})"
                if EMBEDDING_PROVIDER == "openai"
                else (str(_model.device).split(':')[0] if _model is not None else "not_loaded")
            ),
        },
        "database": {
            "size_mb": db_size_mb,
            "journal_mode": journal_mode,
            "wal_exists": wal_exists,
            "wal_size_mb": wal_size_mb,
        },
    }
