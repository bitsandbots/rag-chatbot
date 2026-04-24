# Setup Guide

## Prerequisites

### 1. Install Ollama

Follow the official instructions at https://ollama.com/download or on Raspberry Pi:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify Ollama is running:

```bash
ollama list
```

### 2. Pull the required models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
```

Both pulls require network access. After that, all inference is local.

## Install

### Clone and set up the virtual environment

```bash
git clone https://github.com/bitsandbots/rag-chatbot.git
cd rag-chatbot
make install
```

`make install` creates `venv/` and runs `pip install -e ".[dev]"`.

### Configure environment

```bash
cp .env.example .env
```

Default `.env.example` values:

```
PORT=5000
OLLAMA_HOST=http://localhost:11434
MODEL=qwen2.5-coder:3b
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./chroma_db
COLLECTION_NAME=documents
```

Edit `.env` to change the model, port, or ChromaDB path. The file is loaded automatically at startup via `python-dotenv`.

## Running the Server

```bash
make run
```

This runs `venv/bin/python -m rag_chatbot`, which loads `.env` and starts Flask on `PORT` (default 5000).

The server binds to `0.0.0.0` so it is reachable from other devices on the local network.

## Testing the API

### Health check

```bash
curl http://localhost:5000/health
```

Expected response:

```json
{"engine": "local", "status": "ok"}
```

### Ingest documents

```bash
curl -s -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The Raspberry Pi 5 has an 8GB RAM option.", "Ollama runs LLMs locally without cloud services."]}'
```

Expected response:

```json
{"indexed": 2}
```

### Query

```bash
curl -s -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How much RAM does the Raspberry Pi 5 have?"}'
```

Expected response:

```json
{
  "answer": "...",
  "model": "qwen2.5-coder:3b",
  "question": "How much RAM does the Raspberry Pi 5 have?"
}
```

### Ingest from text files

Place `.txt` files in the `documents/` directory, then use `load_text_files()` in a script:

```python
from rag_chatbot.document_loader import load_text_files
from rag_chatbot.rag_engine import RAGEngine
import requests

chunks, ids = load_text_files("documents/")
requests.post("http://localhost:5000/ingest", json={"texts": chunks, "ids": ids})
```

## Running Tests

```bash
make test
```

Runs `pytest -q` against all 43 tests in `tests/`. Ollama and ChromaDB are mocked in `conftest.py` — no live models required.

## Linting

```bash
make lint       # ruff check src/ tests/
make format     # ruff format src/ tests/
```

## Upgrading to Production

Run the upgrade script once to extend the starter stack to a dual-process production setup:

```bash
bash scripts/upgrade.sh
```

The script is idempotent — safe to re-run. It adds:

- FastAPI backend (`src/rag_chatbot/fastapi_app.py`) on port 7860
- PDF ingestion support (`src/rag_chatbot/pdf_loader.py`)
- SQLite chat history (`src/rag_chatbot/chat_history.py`)
- Browser chat UI at `GET /chat`
- systemd service files in `services/`

After running the upgrade script, install the systemd services:

```bash
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-stack-api ai-stack-web
```

Verify:

```bash
curl http://localhost:7860/api/health
curl http://localhost:5000/health
```
