# RAG Chatbot Starter Stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the supporting source code for the offline-first AI stack guide — a clone-and-run RAG chatbot with Flask API, ChromaDB vector store, and Ollama inference, plus an upgrade script for production architecture.

**Architecture:** Single Flask server with factory pattern (`create_app()`). RAGEngine class wraps ChromaDB + Ollama. All config via env vars with sensible defaults matching the guide. Upgrade script adds FastAPI backend, streaming, PDF support, chat history, and systemd services.

**Tech Stack:** Python 3.13, Flask, ChromaDB, Ollama, python-dotenv, pytest, ruff, hatchling

**Spec:** `docs/superpowers/specs/2026-04-24-rag-chatbot-starter-design.md`

---

## File Map

| File | Responsibility | Task |
|------|---------------|------|
| `pyproject.toml` | Package metadata, dependencies, build config | 1 |
| `Makefile` | Dev commands: install, run, test, lint, format | 1 |
| `.env.example` | Documented env var defaults | 1 |
| `documents/.gitkeep` | Default document directory placeholder | 1 |
| `src/rag_chatbot/__init__.py` | Version export | 1 |
| `src/rag_chatbot/__main__.py` | `python -m rag_chatbot` entrypoint | 1 |
| `tests/conftest.py` | Shared fixtures, ollama mocks | 2 |
| `src/rag_chatbot/rag_engine.py` | RAGEngine: embed, add, query, generate | 2 |
| `tests/test_rag_engine.py` | RAGEngine unit tests | 2 |
| `src/rag_chatbot/document_loader.py` | Text file chunking + loading | 3 |
| `tests/test_document_loader.py` | Document loader tests | 3 |
| `src/rag_chatbot/api_server.py` | Flask app factory + endpoints | 4 |
| `tests/test_api_server.py` | Flask endpoint tests | 4 |
| `scripts/upgrade.sh` | Production upgrade script | 5 |
| `CLAUDE.md` | Update with actual commands | 6 |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `.env.example`
- Create: `documents/.gitkeep`
- Create: `src/rag_chatbot/__init__.py`
- Create: `src/rag_chatbot/__main__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rag-chatbot"
version = "0.1.0"
description = "Offline-first RAG chatbot starter stack for Raspberry Pi"
readme = "README.md"
license = "MIT"
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

[tool.ruff]
target-version = "py311"
line-length = 88
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: install run test lint format clean

install:
	python3 -m venv venv
	venv/bin/pip install -e ".[dev]"

run:
	venv/bin/python -m rag_chatbot

test:
	venv/bin/pytest -q

lint:
	venv/bin/ruff check src/ tests/

format:
	venv/bin/ruff format src/ tests/

clean:
	rm -rf venv/ chroma_db/ *.egg-info src/*.egg-info
```

- [ ] **Step 3: Create .env.example**

```
PORT=5000
OLLAMA_HOST=http://localhost:11434
MODEL=qwen2.5-coder:3b
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./chroma_db
COLLECTION_NAME=documents
```

- [ ] **Step 4: Create documents/.gitkeep**

Empty file.

- [ ] **Step 5: Create src/rag_chatbot/__init__.py**

```python
"""RAG Chatbot — offline-first starter stack."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create src/rag_chatbot/__main__.py**

```python
"""Entrypoint for python -m rag_chatbot."""

from rag_chatbot.api_server import create_app

import os

from dotenv import load_dotenv

load_dotenv()

app = create_app()
app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
```

- [ ] **Step 7: Install dependencies**

Run: `make install`
Expected: venv created, all deps installed, `pip list` shows chromadb, ollama, flask, pytest, ruff

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml Makefile .env.example documents/.gitkeep src/rag_chatbot/__init__.py src/rag_chatbot/__main__.py
git commit -m "chore: add project scaffolding"
```

---

### Task 2: RAG Engine + Tests

**Files:**
- Create: `src/rag_chatbot/rag_engine.py`
- Create: `tests/conftest.py`
- Create: `tests/test_rag_engine.py`

- [ ] **Step 1: Create tests/conftest.py with ollama mocks**

```python
from __future__ import annotations

import hashlib

import pytest

from rag_chatbot.rag_engine import RAGEngine


def _deterministic_embedding(text: str) -> list[float]:
    """Generate a deterministic 768-dim vector from text for testing."""
    h = hashlib.sha256(text.encode()).digest()
    base = [b / 255.0 for b in h]
    return (base * 24)[:768]


@pytest.fixture(autouse=True)
def _mock_ollama(monkeypatch):
    """Mock ollama.embed and ollama.generate for all tests."""

    def mock_embed(model: str, input: str) -> dict:
        return {"embeddings": [_deterministic_embedding(input)]}

    def mock_generate(model: str, prompt: str, **kwargs) -> dict:
        return {"response": f"Mock answer for: {prompt[:50]}"}

    import ollama as _ollama

    monkeypatch.setattr(_ollama, "embed", mock_embed)
    monkeypatch.setattr(_ollama, "generate", mock_generate)


@pytest.fixture()
def rag_engine(tmp_path) -> RAGEngine:
    """Create a RAGEngine with a temp ChromaDB directory."""
    return RAGEngine(
        collection_name="test_docs",
        chroma_path=str(tmp_path / "chroma"),
        embed_model="nomic-embed-text",
        gen_model="qwen2.5-coder:3b",
    )
```

- [ ] **Step 2: Create tests/test_rag_engine.py with failing tests**

```python
from __future__ import annotations

from rag_chatbot.rag_engine import RAGEngine


def test_embed_text(rag_engine: RAGEngine) -> None:
    result = rag_engine.embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_add_and_query(rag_engine: RAGEngine) -> None:
    rag_engine.add_documents(
        texts=["The sky is blue", "Grass is green", "Water is wet"],
        ids=["doc_0", "doc_1", "doc_2"],
    )
    results = rag_engine.query("What color is the sky?", n_results=2)
    assert "documents" in results
    assert len(results["documents"][0]) == 2


def test_generate_answer(rag_engine: RAGEngine) -> None:
    rag_engine.add_documents(texts=["Pi 5 has a quad-core CPU"], ids=["pi_0"])
    answer = rag_engine.generate_answer("What CPU does Pi 5 have?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_auto_ids(rag_engine: RAGEngine) -> None:
    rag_engine.add_documents(texts=["first", "second", "third"])
    results = rag_engine.query("first")
    assert len(results["documents"][0]) > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_rag_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_chatbot.rag_engine'` (file not created yet)

- [ ] **Step 4: Create src/rag_chatbot/rag_engine.py**

```python
"""Local RAG pipeline using ChromaDB + Ollama."""

from __future__ import annotations

import chromadb
import ollama


class RAGEngine:
    """Local RAG pipeline using ChromaDB + Ollama."""

    def __init__(
        self,
        collection_name: str = "documents",
        chroma_path: str = "./chroma_db",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "qwen2.5-coder:3b",
    ) -> None:
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embeddings via Ollama."""
        response = ollama.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]

    def add_documents(
        self, texts: list[str], ids: list[str] | None = None
    ) -> None:
        """Index documents into ChromaDB."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = [self.embed_text(t) for t in texts]
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
        )

    def query(self, question: str, n_results: int = 3) -> dict:
        """Retrieve relevant documents for a question."""
        query_embedding = self.embed_text(question)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def generate_answer(self, question: str) -> str:
        """Full RAG pipeline: retrieve context, generate answer."""
        results = self.query(question)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""Answer based on the following context:
Context: {context}
Question: {question}
Answer:"""

        response = ollama.generate(model=self.gen_model, prompt=prompt)
        return response["response"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_rag_engine.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/rag_chatbot/rag_engine.py tests/conftest.py tests/test_rag_engine.py
git commit -m "feat: add RAGEngine with ChromaDB + Ollama"
```

---

### Task 3: Document Loader + Tests

**Files:**
- Create: `src/rag_chatbot/document_loader.py`
- Create: `tests/test_document_loader.py`

- [ ] **Step 1: Create tests/test_document_loader.py with failing tests**

```python
from __future__ import annotations

from pathlib import Path

from rag_chatbot.document_loader import load_text_files


def test_load_text_files(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("Hello world this is a test file.")
    chunks, ids = load_text_files(str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0] == "Hello world this is a test file."
    assert ids[0] == "hello.txt_0"


def test_chunking(tmp_path: Path) -> None:
    text = "A" * 1200  # Should produce 3 chunks at chunk_size=500
    (tmp_path / "big.txt").write_text(text)
    chunks, ids = load_text_files(str(tmp_path), chunk_size=500)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 200
    assert ids[0] == "big.txt_0"
    assert ids[1] == "big.txt_500"
    assert ids[2] == "big.txt_1000"


def test_empty_directory(tmp_path: Path) -> None:
    chunks, ids = load_text_files(str(tmp_path))
    assert chunks == []
    assert ids == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_document_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create src/rag_chatbot/document_loader.py**

```python
"""Load and chunk text files for RAG ingestion."""

from __future__ import annotations

from pathlib import Path


def load_text_files(
    directory: str, chunk_size: int = 500
) -> tuple[list[str], list[str]]:
    """Load and chunk text files from a directory.

    Args:
        directory: Path to directory containing .txt files.
        chunk_size: Maximum characters per chunk.

    Returns:
        Tuple of (chunks, ids) where ids are "{filename}_{offset}".
    """
    chunks: list[str] = []
    ids: list[str] = []

    for file_path in sorted(Path(directory).glob("*.txt")):
        text = file_path.read_text()
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            ids.append(f"{file_path.name}_{i}")

    return chunks, ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_document_loader.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_chatbot/document_loader.py tests/test_document_loader.py
git commit -m "feat: add document loader with text chunking"
```

---

### Task 4: Flask API Server + Tests

**Files:**
- Create: `src/rag_chatbot/api_server.py`
- Create: `tests/test_api_server.py`
- Modify: `tests/conftest.py` (add Flask client fixture)

- [ ] **Step 1: Add Flask client fixture to tests/conftest.py**

Append to the end of `tests/conftest.py`:

```python
from flask.testing import FlaskClient

from rag_chatbot.api_server import create_app


@pytest.fixture()
def client(rag_engine: RAGEngine) -> FlaskClient:
    """Flask test client wired to a test RAGEngine."""
    app = create_app(rag=rag_engine)
    app.config["TESTING"] = True
    return app.test_client()
```

- [ ] **Step 2: Create tests/test_api_server.py with failing tests**

```python
from __future__ import annotations

from flask.testing import FlaskClient


def test_health(client: FlaskClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["engine"] == "local"


def test_ingest(client: FlaskClient) -> None:
    resp = client.post("/ingest", json={
        "texts": ["Raspberry Pi 5 has 8GB RAM"],
        "ids": ["pi5_ram"],
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["indexed"] == 1


def test_query(client: FlaskClient) -> None:
    client.post("/ingest", json={
        "texts": ["The sky is blue"],
        "ids": ["sky_0"],
    })
    resp = client.post("/query", json={"question": "What color is the sky?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert data["question"] == "What color is the sky?"
    assert "model" in data


def test_query_missing_question(client: FlaskClient) -> None:
    resp = client.post("/query", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "question required"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_api_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Create src/rag_chatbot/api_server.py**

```python
"""Flask API server for the RAG chatbot."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from rag_chatbot.rag_engine import RAGEngine


def create_app(rag: RAGEngine | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        rag: Optional RAGEngine instance. If None, creates one from env vars.
    """
    app = Flask(__name__)

    if rag is None:
        rag = RAGEngine(
            collection_name=os.getenv("COLLECTION_NAME", "documents"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            gen_model=os.getenv("MODEL", "qwen2.5-coder:3b"),
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "engine": "local"})

    @app.route("/ingest", methods=["POST"])
    def ingest():
        data = request.get_json()
        texts = data.get("texts", [])
        ids = data.get("ids")
        rag.add_documents(texts, ids)
        return jsonify({"indexed": len(texts)})

    @app.route("/query", methods=["POST"])
    def query():
        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({"error": "question required"}), 400
        answer = rag.generate_answer(question)
        return jsonify({
            "question": question,
            "answer": answer,
            "model": rag.gen_model,
        })

    return app
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_server.py -v`
Expected: 4 passed

- [ ] **Step 6: Run full test suite**

Run: `venv/bin/pytest -q`
Expected: 11 passed

- [ ] **Step 7: Lint**

Run: `venv/bin/ruff check src/ tests/`
Expected: All checks passed (fix any issues before committing)

- [ ] **Step 8: Commit**

```bash
git add src/rag_chatbot/api_server.py tests/conftest.py tests/test_api_server.py
git commit -m "feat: add Flask API server with health, ingest, query endpoints"
```

---

### Task 5: Upgrade Script

**Files:**
- Create: `scripts/upgrade.sh`

The upgrade script is a shell script that generates Python files, systemd units, and a chat template. It must be idempotent.

- [ ] **Step 1: Create scripts/upgrade.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src/rag_chatbot"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
skip()  { echo -e "${YELLOW}[skip]${NC} $1 already exists"; }

echo "============================================"
echo "  RAG Chatbot — Production Upgrade"
echo "============================================"
echo

# --- Install upgrade dependencies ---
info "Installing upgrade dependencies..."
if [ -d "$PROJECT_DIR/venv" ]; then
    "$PROJECT_DIR/venv/bin/pip" install -e "$PROJECT_DIR[upgrade]" -q
else
    pip install -e "$PROJECT_DIR[upgrade]" -q
fi

# --- FastAPI backend ---
FASTAPI_FILE="$SRC_DIR/fastapi_app.py"
if [ -f "$FASTAPI_FILE" ]; then
    skip "fastapi_app.py"
else
    info "Creating FastAPI backend..."
    cat > "$FASTAPI_FILE" << 'PYEOF'
"""FastAPI backend for production RAG chatbot."""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import ollama as ollama_client

from rag_chatbot.rag_engine import RAGEngine

app = FastAPI(title="RAG Chatbot API")

rag = RAGEngine(
    collection_name=os.getenv("COLLECTION_NAME", "documents"),
    chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
    embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    gen_model=os.getenv("MODEL", "qwen2.5-coder:3b"),
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "local", "mode": "production"}


@app.post("/api/ingest")
async def ingest(request: Request):
    data = await request.json()
    texts = data.get("texts", [])
    ids = data.get("ids")
    rag.add_documents(texts, ids)
    return {"indexed": len(texts)}


@app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "question required"}
    answer = rag.generate_answer(question)
    return {"question": question, "answer": answer, "model": rag.gen_model}


@app.post("/api/stream")
async def stream(request: Request):
    """SSE endpoint for streaming token-by-token responses."""
    data = await request.json()
    question = data.get("question", "")

    results = rag.query(question)
    context = "\n\n".join(results["documents"][0]) if results["documents"][0] else ""

    prompt = f"""Answer based on the following context:
Context: {context}
Question: {question}
Answer:"""

    def event_stream():
        for chunk in ollama_client.generate(
            model=rag.gen_model, prompt=prompt, stream=True
        ):
            token = chunk.get("response", "")
            done = chunk.get("done", False)
            if done:
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
PYEOF
fi

# --- PDF loader ---
PDF_FILE="$SRC_DIR/pdf_loader.py"
if [ -f "$PDF_FILE" ]; then
    skip "pdf_loader.py"
else
    info "Creating PDF loader..."
    cat > "$PDF_FILE" << 'PYEOF'
"""Load and chunk PDF files for RAG ingestion."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def load_pdf(
    path: str, chunk_size: int = 500
) -> tuple[list[str], list[str]]:
    """Load and chunk a PDF file.

    Args:
        path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.

    Returns:
        Tuple of (chunks, ids) where ids are "{filename}_p{page}_{offset}".
    """
    chunks: list[str] = []
    ids: list[str] = []
    file_name = Path(path).name

    doc = pymupdf.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i : i + chunk_size]
        chunks.append(chunk)
        ids.append(f"{file_name}_{i}")

    return chunks, ids
PYEOF
fi

# --- Chat history ---
CHAT_FILE="$SRC_DIR/chat_history.py"
if [ -f "$CHAT_FILE" ]; then
    skip "chat_history.py"
else
    info "Creating chat history module..."
    cat > "$CHAT_FILE" << 'PYEOF'
"""SQLite-backed chat session and message storage."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone


class ChatHistory:
    """Manages chat sessions and messages in SQLite."""

    def __init__(self, db_path: str = "./chat.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def create_session(self) -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, now),
            )
        return session_id

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Add a message to a session."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )

    def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session, ordered by creation time."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
PYEOF
fi

# --- Chat template ---
TMPL_DIR="$SRC_DIR/templates"
CHAT_TMPL="$TMPL_DIR/chat.html"
if [ -f "$CHAT_TMPL" ]; then
    skip "templates/chat.html"
else
    info "Creating chat UI template..."
    mkdir -p "$TMPL_DIR"
    cat > "$CHAT_TMPL" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; background: #0d1421; color: #e9edf2; height: 100vh; display: flex; flex-direction: column; }
        header { background: #0d1b2e; padding: 1rem; text-align: center; border-bottom: 3px solid; border-image: linear-gradient(90deg, #2b7de9, #e07018) 1; }
        header h1 { font-size: 1.2rem; }
        #messages { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
        .msg { padding: 0.75rem 1rem; border-radius: 8px; max-width: 80%; line-height: 1.5; }
        .msg.user { background: #2b7de9; align-self: flex-end; }
        .msg.assistant { background: #1a2332; align-self: flex-start; border: 1px solid #2a3a4a; }
        #input-bar { display: flex; gap: 0.5rem; padding: 1rem; background: #0d1b2e; }
        #input-bar input { flex: 1; padding: 0.75rem; border-radius: 6px; border: 1px solid #2a3a4a; background: #1a2332; color: #e9edf2; font-size: 1rem; }
        #input-bar button { padding: 0.75rem 1.5rem; border-radius: 6px; border: none; background: #2b7de9; color: #fff; cursor: pointer; font-size: 1rem; }
        #input-bar button:hover { background: #1a6dd4; }
    </style>
</head>
<body>
    <header><h1>RAG Chat<span style="color:#e07018">bot</span></h1></header>
    <div id="messages"></div>
    <div id="input-bar">
        <input type="text" id="question" placeholder="Ask a question..." autocomplete="off" />
        <button onclick="send()">Send</button>
    </div>
    <script>
        const messages = document.getElementById('messages');
        const input = document.getElementById('question');
        input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

        async function send() {
            const q = input.value.trim();
            if (!q) return;
            addMsg('user', q);
            input.value = '';
            try {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                addMsg('assistant', data.answer || data.error);
            } catch (err) {
                addMsg('assistant', 'Error: ' + err.message);
            }
        }

        function addMsg(role, text) {
            const div = document.createElement('div');
            div.className = 'msg ' + role;
            div.textContent = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
HTMLEOF
fi

# --- Patch api_server.py with /chat route ---
if grep -q '/chat' "$SRC_DIR/api_server.py" 2>/dev/null; then
    skip "/chat route in api_server.py"
else
    info "Adding /chat route to api_server.py..."
    # Insert before the final "return app" line
    sed -i '/^    return app$/i\
    @app.route("/chat", methods=["GET"])\
    def chat():\
        from flask import render_template\
        return render_template("chat.html")\
' "$SRC_DIR/api_server.py"
fi

# --- Update .env.example ---
if grep -q 'API_PORT' "$PROJECT_DIR/.env.example" 2>/dev/null; then
    skip ".env.example already has production vars"
else
    info "Updating .env.example with production vars..."
    cat >> "$PROJECT_DIR/.env.example" << 'EOF'

# Production (added by upgrade script)
API_PORT=7860
WEB_PORT=5050
DB_PATH=./chat.db
EOF
fi

# --- systemd services ---
SERVICES_DIR="$PROJECT_DIR/services"
mkdir -p "$SERVICES_DIR"

API_SERVICE="$SERVICES_DIR/ai-stack-api.service"
if [ -f "$API_SERVICE" ]; then
    skip "ai-stack-api.service"
else
    info "Creating FastAPI systemd service..."
    cat > "$API_SERVICE" << EOF
[Unit]
Description=RAG Chatbot FastAPI Backend
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/uvicorn rag_chatbot.fastapi_app:app --host 0.0.0.0 --port 7860
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
fi

WEB_SERVICE="$SERVICES_DIR/ai-stack-web.service"
if [ -f "$WEB_SERVICE" ]; then
    skip "ai-stack-web.service"
else
    info "Creating Flask web UI systemd service..."
    cat > "$WEB_SERVICE" << EOF
[Unit]
Description=RAG Chatbot Flask Web UI
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/python -m rag_chatbot
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
fi

# --- Git commit ---
echo
info "Committing upgrade files..."
cd "$PROJECT_DIR"
git add -A
git commit -m "feat: upgrade to production architecture" || warn "Nothing to commit"

echo
echo "============================================"
echo "  Upgrade complete!"
echo "============================================"
echo
echo "New files added:"
echo "  - src/rag_chatbot/fastapi_app.py   (FastAPI backend, port 7860)"
echo "  - src/rag_chatbot/pdf_loader.py    (PDF ingestion via pymupdf)"
echo "  - src/rag_chatbot/chat_history.py  (SQLite chat sessions)"
echo "  - src/rag_chatbot/templates/chat.html (Chat UI)"
echo "  - services/ai-stack-api.service    (systemd for FastAPI)"
echo "  - services/ai-stack-web.service    (systemd for Flask)"
echo
echo "Next steps:"
echo "  1. Copy services:  sudo cp services/*.service /etc/systemd/system/"
echo "  2. Reload systemd: sudo systemctl daemon-reload"
echo "  3. Enable:         sudo systemctl enable --now ai-stack-api ai-stack-web"
echo "  4. Test:           curl http://localhost:7860/api/health"
echo "  5. Chat UI:        http://localhost:5000/chat"
echo
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/upgrade.sh`

- [ ] **Step 3: Run lint on all Python files embedded in the script**

Visually verify: all Python code in the heredocs uses parameterized SQL, has type hints, follows ruff formatting.

- [ ] **Step 4: Commit**

```bash
git add scripts/upgrade.sh
git commit -m "feat: add production upgrade script"
```

---

### Task 6: Update CLAUDE.md + Final Polish

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with actual commands**

Replace the placeholder commands section and architecture section with real content now that the code exists:

```markdown
## Commands

- `make install` — create venv, install with dev deps
- `make run` — start Flask server (`python -m rag_chatbot`)
- `make test` — run tests (`pytest -q`)
- `make lint` — lint (`ruff check src/ tests/`)
- `make format` — format (`ruff format src/ tests/`)
- `pytest tests/test_rag_engine.py::test_embed_text -v` — run a single test

## Architecture

Single-process Flask API server with factory pattern (`create_app()`).

- `src/rag_chatbot/rag_engine.py` — RAGEngine class: wraps ChromaDB (vector store) + Ollama (embeddings + generation). Config-agnostic — receives all settings via constructor.
- `src/rag_chatbot/document_loader.py` — `load_text_files()`: globs *.txt, chunks by character count, returns (chunks, ids).
- `src/rag_chatbot/api_server.py` — Flask app factory. Endpoints: GET /health, POST /ingest, POST /query. Reads env vars to construct RAGEngine when no instance is injected.
- `src/rag_chatbot/__main__.py` — Entrypoint: loads .env, creates app, runs on PORT (default 5000).

Tests mock `ollama.embed` and `ollama.generate` in `conftest.py` so they run without a live Ollama instance. ChromaDB uses tmp_path per test.

## Upgrade Path

`bash scripts/upgrade.sh` adds FastAPI backend (port 7860), streaming SSE, PDF ingestion, SQLite chat history, chat UI template, and systemd service files.
```

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: 11 passed

- [ ] **Step 3: Run lint**

Run: `make lint`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with actual commands and architecture"
```

- [ ] **Step 5: Push**

```bash
git push origin main
```
