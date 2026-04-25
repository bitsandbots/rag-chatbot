#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════╗
# ║  DEPRECATION NOTICE                                         ║
# ║                                                              ║
# ║  This upgrade script is kept for educational purposes —      ║
# ║  it demonstrates how to extend the starter stack with        ║
# ║  FastAPI, chat history, and a web UI.                        ║
# ║                                                              ║
# ║  For production use, deploy CoreChat instead:                ║
# ║  https://github.com/coreconduit/corechat                    ║
# ║                                                              ║
# ║  CoreChat includes everything this script generates,         ║
# ║  plus: hybrid retrieval, multi-language support,             ║
# ║  a polished Flask UI, Docker, and systemd services.          ║
# ╚══════════════════════════════════════════════════════════════╝

echo ""
echo "NOTE: This script is for learning purposes."
echo "  For production, use CoreChat: https://github.com/coreconduit/corechat"
echo ""

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
    gen_model=os.getenv("MODEL", "qwen3:1.7b"),
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "engine": "local", "mode": "production"}


@app.post("/api/ingest")
async def ingest(request: Request) -> dict:
    data = await request.json()
    texts = data.get("texts", [])
    ids = data.get("ids")
    rag.add_documents(texts, ids)
    return {"indexed": len(texts)}


@app.post("/api/query")
async def query(request: Request) -> dict:
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "question required"}
    answer = rag.generate_answer(question)
    return {"question": question, "answer": answer, "model": rag.gen_model}


@app.post("/api/stream")
async def stream(request: Request) -> StreamingResponse:
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
        Tuple of (chunks, ids) where ids are "{filename}_{offset}".
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

# --- Install systemd services ---
echo
info "Installing systemd services..."

install_service() {
    local src="$1"
    local name
    name="$(basename "$src")"
    local dst="/etc/systemd/system/$name"

    if [ "$(id -u)" -eq 0 ]; then
        cp "$src" "$dst"
        info "Installed $name"
    else
        warn "Not root — skipping systemd install for $name"
    fi
}

# Stop and disable the base service if it exists (upgrading from starter)
if systemctl is-active --quiet rag-chatbot.service 2>/dev/null; then
    if [ "$(id -u)" -eq 0 ]; then
        info "Stopping base rag-chatbot.service (replaced by split services)..."
        systemctl stop rag-chatbot.service
        systemctl disable rag-chatbot.service
        rm -f /etc/systemd/system/rag-chatbot.service
    else
        warn "rag-chatbot.service is running — stop it manually before starting split services"
    fi
fi

install_service "$API_SERVICE"
install_service "$WEB_SERVICE"

if [ "$(id -u)" -eq 0 ]; then
    systemctl daemon-reload
    systemctl enable ai-stack-api.service ai-stack-web.service
    info "Services enabled. Start with:"
    echo "    sudo systemctl start ai-stack-api ai-stack-web"
else
    echo
    warn "Run the following to install services manually:"
    echo "    sudo cp services/*.service /etc/systemd/system/"
    echo "    sudo systemctl daemon-reload"
    echo "    sudo systemctl enable --now ai-stack-api ai-stack-web"
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
echo "Services:"
echo "  - ai-stack-api:  FastAPI on port 7860"
echo "  - ai-stack-web:  Flask + chat UI on port 5000"
echo
echo "Test:"
echo "  curl http://localhost:7860/api/health"
echo "  open http://localhost:5000/chat"
echo
