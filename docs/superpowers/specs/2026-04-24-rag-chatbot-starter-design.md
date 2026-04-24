# RAG Chatbot Starter Stack — Design Spec

**Date:** 2026-04-24
**Status:** Approved
**Source guide:** https://coreconduit.com/techlounge/guides/offline-first-ai-stack.html

## Goal

Create the supporting source code for the offline-first AI stack guide. The repo should be clone-and-run for guide readers, properly packaged with tests and tooling, and include an optional upgrade script to evolve into a production dual-process architecture.

## Constraints

- Must match the guide's code (class names, method signatures, API endpoints)
- Runs on Raspberry Pi 5 (8GB) and Pi 4 (4GB) without cloud dependencies
- Requires Ollama running locally with `nomic-embed-text` pulled
- Python 3.13, no class components, no unnecessary abstractions
- Single Flask server for base; dual Flask+FastAPI after upgrade

## Project Structure

```
rag-chatbot/
├── pyproject.toml
├── Makefile
├── .env.example
├── src/
│   └── rag_chatbot/
│       ├── __init__.py
│       ├── __main__.py
│       ├── rag_engine.py
│       ├── document_loader.py
│       └── api_server.py
├── tests/
│   ├── conftest.py
│   ├── test_rag_engine.py
│   ├── test_document_loader.py
│   └── test_api_server.py
├── scripts/
│   └── upgrade.sh
├── documents/
│   └── .gitkeep
├── CLAUDE.md
├── README.md
├── LICENSE
└── .gitignore
```

## Module Specifications

### `src/rag_chatbot/__init__.py`

Exports `__version__ = "0.1.0"`.

### `src/rag_chatbot/rag_engine.py`

```python
class RAGEngine:
    def __init__(self, collection_name: str, chroma_path: str,
                 embed_model: str, gen_model: str) -> None: ...
    def embed_text(self, text: str) -> list[float]: ...
    def add_documents(self, texts: list[str], ids: list[str] | None = None) -> None: ...
    def query(self, question: str, n_results: int = 3) -> dict: ...
    def generate_answer(self, question: str) -> str: ...
```

- Uses `chromadb.PersistentClient` with cosine distance
- Embeddings via `ollama.embed(model=embed_model, input=text)`
- Generation via `ollama.generate(model=gen_model, prompt=...)`
- Constructor args sourced from environment in `api_server.py`; class itself is config-agnostic

### `src/rag_chatbot/document_loader.py`

```python
def load_text_files(directory: str, chunk_size: int = 500) -> tuple[list[str], list[str]]: ...
```

- Globs `*.txt` from the directory
- Chunks by character count (non-overlapping)
- Returns `(chunks, ids)` where ids are `"{filename}_{offset}"`

### `src/rag_chatbot/api_server.py`

Flask app with factory pattern:

```python
def create_app(rag: RAGEngine | None = None) -> Flask: ...
```

**Endpoints:**

| Method | Path      | Request Body                       | Response                                   |
|--------|-----------|------------------------------------|--------------------------------------------|
| GET    | /health   | —                                  | `{"status": "ok", "engine": "local"}`      |
| POST   | /ingest   | `{"texts": [...], "ids": [...]}`   | `{"indexed": <count>}`                     |
| POST   | /query    | `{"question": "...", "model": "..."}` | `{"question", "answer", "model"}`       |

- `model` field in `/query` is optional, defaults to configured `MODEL`
- Returns 400 with `{"error": "question required"}` when question missing
- Port from `PORT` env var, default `5000`

**Entrypoint:** `python -m rag_chatbot` runs the Flask server (via `__main__.py`).

### Environment Variables (`.env.example`)

```
PORT=5000
OLLAMA_HOST=http://localhost:11434
MODEL=qwen2.5-coder:3b
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./chroma_db
COLLECTION_NAME=documents
```

All optional — defaults match the guide's hardcoded values.

### `pyproject.toml`

```toml
[project]
name = "rag-chatbot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "chromadb",
    "ollama",
    "flask",
    "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
upgrade = ["fastapi", "uvicorn[standard]", "pymupdf"]
```

Build backend: `hatchling`.

### `Makefile`

| Target         | Command                                              |
|----------------|------------------------------------------------------|
| `make install` | `python3 -m venv venv && venv/bin/pip install -e ".[dev]"` |
| `make run`     | `venv/bin/python -m rag_chatbot`                     |
| `make test`    | `venv/bin/pytest -q`                                 |
| `make lint`    | `venv/bin/ruff check src/ tests/`                    |
| `make format`  | `venv/bin/ruff format src/ tests/`                   |

### Tests

**Strategy:** Mock `ollama.embed` and `ollama.generate` so tests run without a live Ollama instance. ChromaDB uses a temp directory per test session via `tmp_path` fixture.

**`conftest.py`:**
- Fixture `rag_engine` — creates a `RAGEngine` with a temp chroma dir
- Fixture `client` — Flask test client via `create_app(rag_engine)`
- Monkeypatch `ollama.embed` to return a deterministic 768-dim vector
- Monkeypatch `ollama.generate` to return a canned response

**`test_rag_engine.py`:**
- `test_embed_text` — returns list of floats, correct length
- `test_add_and_query` — add docs, query returns relevant results
- `test_generate_answer` — returns string response
- `test_auto_ids` — omitting ids generates `doc_0`, `doc_1`, ...

**`test_document_loader.py`:**
- `test_load_text_files` — loads from temp dir with `.txt` files
- `test_chunking` — verifies chunk size boundaries
- `test_empty_directory` — returns empty lists

**`test_api_server.py`:**
- `test_health` — GET /health returns 200 with status ok
- `test_ingest` — POST /ingest returns indexed count
- `test_query` — POST /query returns answer
- `test_query_missing_question` — returns 400

## Upgrade Script (`scripts/upgrade.sh`)

### What it creates

**New files:**

| File | Purpose |
|------|---------|
| `src/rag_chatbot/fastapi_app.py` | FastAPI backend on port 7860 — async `/api/health`, `/api/ingest`, `/api/query`, `/api/stream` (SSE) |
| `src/rag_chatbot/pdf_loader.py` | PDF ingestion via `pymupdf` — `load_pdf(path, chunk_size=500) -> (chunks, ids)` |
| `src/rag_chatbot/chat_history.py` | SQLite session/message storage — `ChatHistory` class with `create_session()`, `add_message()`, `get_messages()` |
| `src/rag_chatbot/templates/chat.html` | Minimal chat UI template for Flask |
| `services/ai-stack-api.service` | systemd unit for FastAPI backend |
| `services/ai-stack-web.service` | systemd unit for Flask web UI |

**Modified files:**

| File | Change |
|------|--------|
| `src/rag_chatbot/api_server.py` | Add `/chat` route serving the chat template, wire up chat history |
| `.env.example` | Add `API_PORT=7860`, `WEB_PORT=5050`, `DB_PATH=./chat.db` |

### Behavior

- Idempotent — checks for existing files before writing
- Installs `[upgrade]` optional dependencies
- Creates a git commit: `feat: upgrade to production architecture`
- Prints summary of what was added and next steps (systemd install commands)

### FastAPI streaming endpoint

`POST /api/stream` accepts `{"question": "..."}` and returns SSE with `ollama.generate(stream=True)`. Each event is `data: {"token": "..."}` with a final `data: {"done": true}`.

### Chat history schema

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

## Out of Scope

- Authentication / multi-user (extensibility mention in guide, not implemented)
- Web scraping loader
- React/Vue frontend (guide mentions as extensibility, upgrade uses server-rendered template)
- Docker packaging
- CI/CD pipeline
