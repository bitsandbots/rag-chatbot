# Tech Stack

## Hardware

| Component | Recommended | Minimum |
|---|---|---|
| Board | Raspberry Pi 5 (8GB) | Raspberry Pi 4 (4GB) |
| Storage | 32GB microSD or USB SSD | 16GB microSD |
| OS | Raspberry Pi OS 64-bit | Ubuntu Server 24.04 LTS |

Models and ChromaDB require significant RAM. The 4GB Pi 4 is workable with a small model (`qwen3:1.7b`) but will be slow on large corpora. The 8GB Pi 5 is the recommended target.

## Runtime Dependencies

| Package | Purpose | Notes |
|---|---|---|
| Python | Language runtime | `>=3.11` required (match `pyproject.toml`) |
| Ollama | Local LLM inference | Must be installed separately; see setup |
| `chromadb` | Vector database | Persists to disk at `CHROMA_PATH` |
| `ollama` (Python) | Ollama API client | Wraps embed and generate calls |
| `flask` | HTTP API server | Dev server on port 5000 |
| `python-dotenv` | Environment variable loading | Reads `.env` at startup |

## Development Dependencies

Installed via `pip install -e ".[dev]"`:

| Package | Purpose |
|---|---|
| `pytest` | Test runner (43 tests across 5 files) |
| `ruff` | Linter and formatter |
| `hatchling` | Build backend for packaging |

## Upgrade Dependencies

Installed by `scripts/upgrade.sh` via `pip install -e ".[upgrade]"`:

| Package | Purpose |
|---|---|
| `fastapi` | Production async API backend (port 7860) |
| `uvicorn[standard]` | ASGI server for FastAPI |
| `pymupdf` | PDF text extraction for `pdf_loader.py` |

## Ollama Models

| Model | Role | Pull command |
|---|---|---|
| `nomic-embed-text` | Embeddings (768-dim) | `ollama pull nomic-embed-text` |
| `qwen3:1.7b` | Text generation | `ollama pull qwen3:1.7b` |

Both models are configured via `.env` and can be swapped for any Ollama-compatible model.

## Build System

`pyproject.toml` uses `hatchling` as the build backend. The package is named `rag-chatbot`, version `0.1.0`, and exposes the `rag_chatbot` Python package under `src/`.

Ruff is configured for Python 3.11 target, 88-character line length, with `E`, `F`, `I`, `N`, `W` rule sets enabled.
