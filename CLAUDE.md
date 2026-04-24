# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG Chatbot Starter Stack — a self-hosted, offline-first retrieval-augmented generation chatbot built on Raspberry Pi 5.

## Stack

- Python 3.13, Flask (web UI), FastAPI (API backend)
- Ollama for local LLM inference (qwen2.5-coder, llama3)
- nomic-embed-text for embeddings
- ChromaDB for vector storage
- SQLite for metadata/session persistence
- systemd for process management

## Commands

No build system configured yet. When set up, this section should document:
- `pip install -e ".[dev]"` — install with dev dependencies
- `pytest -q` — run tests (use `-q` not `-v` unless debugging)
- `pytest tests/path/test_file.py::test_name` — run a single test
- `ruff check .` — lint
- `ruff format .` — format

## Architecture (Planned)

Dual-process design:
- **Flask web UI** (port 5050) — chat interface, document management, prompt wizard
- **FastAPI backend** (port 7860) — RAG pipeline API, embedding, retrieval, generation

RAG pipeline flow: ingest documents -> chunk -> embed via Ollama -> store in ChromaDB -> query retrieval -> LLM generation with context.

## Conventions

- Package name: TBD (predecessor was `corechat`)
- Config via `.env` loaded through environment variables
- All SQL uses parameterized queries
- Type hints on all function signatures
- Google-style docstrings on public APIs
- Black formatting, ruff for linting
