# rag-chatbot

A self-hosted, offline-first RAG (Retrieval-Augmented Generation) chatbot stack for Raspberry Pi. Clone, install, and query your own documents with local AI — no cloud required.

**Guide:** [Building an Offline-First AI Stack on Raspberry Pi](https://coreconduit.com/techlounge/guides/offline-first-ai-stack.html)

## Stack

- **Ollama** — Local LLM inference (qwen2.5-coder, llama3, phi3)
- **ChromaDB** — Vector database for semantic search
- **Flask** — API server with health, ingest, and query endpoints
- **nomic-embed-text** — Local embedding model

## Quick Start

```bash
# Prerequisites: Ollama installed with nomic-embed-text pulled
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b

# Clone and install
git clone https://github.com/bitsandbots/rag-chatbot.git
cd rag-chatbot
make install

# Run
make run
```

## API

```bash
# Health check
curl http://localhost:5000/health

# Ingest documents
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your document text here"], "ids": ["doc_0"]}'

# Query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say?"}'
```

## Development

```bash
make test     # Run tests
make lint     # Lint with ruff
make format   # Format with ruff
```

## Upgrade to Production

An optional upgrade script adds FastAPI backend, streaming responses, PDF ingestion, chat history, and systemd services:

```bash
bash scripts/upgrade.sh
```

## Hardware

- Raspberry Pi 5 (8GB recommended) or Pi 4 (4GB minimum)
- 64GB+ microSD or USB SSD
- Active cooling

## License

MIT
