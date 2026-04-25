# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG Chatbot Starter Stack — a self-hosted, offline-first retrieval-augmented generation chatbot built on Raspberry Pi 5.

## Stack

- Python 3.13, Flask (web UI), FastAPI (API backend)
- Ollama for local LLM inference (qwen3, llama3)
- nomic-embed-text for embeddings
- ChromaDB for vector storage
- SQLite for metadata/session persistence
- systemd for process management

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

## Conventions

- Package name: `rag-chatbot`
- Config via `.env` loaded through python-dotenv
- All SQL uses parameterized queries
- Type hints on all function signatures
- Google-style docstrings on public APIs
- Ruff for linting and formatting
