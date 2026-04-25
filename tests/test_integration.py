"""Integration tests exercising the full RAG pipeline end-to-end."""

from __future__ import annotations

from pathlib import Path

from flask.testing import FlaskClient

from rag_chatbot.document_loader import load_text_files
from rag_chatbot.rag_engine import RAGEngine


def test_load_ingest_query_pipeline(rag_engine: RAGEngine, tmp_path: Path) -> None:
    """Full pipeline: load text files → ingest → query → generate answer."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "pi.txt").write_text(
        "The Raspberry Pi 5 features a quad-core ARM Cortex-A76 CPU at 2.4GHz. "
        "It has 8GB of RAM and supports PCIe for NVMe storage."
    )
    (doc_dir / "ollama.txt").write_text(
        "Ollama is a local inference engine that runs LLMs on consumer hardware. "
        "It supports models like llama3, qwen2.5-coder, and mistral."
    )

    chunks, ids = load_text_files(str(doc_dir))
    assert len(chunks) == 2

    rag_engine.add_documents(chunks, ids)
    results = rag_engine.query("What CPU does Pi 5 have?", n_results=2)
    assert len(results["documents"][0]) == 2

    answer = rag_engine.generate_answer("What CPU does Pi 5 have?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_api_ingest_multiple_then_query(client: FlaskClient) -> None:
    """API round trip: ingest several docs, then query returns relevant answer."""
    docs = [
        "SQLite is a lightweight embedded database",
        "ChromaDB stores vector embeddings for semantic search",
        "Flask is a micro web framework for Python",
        "Ollama runs large language models locally",
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]

    resp = client.post("/ingest", json={"texts": docs, "ids": ids})
    assert resp.status_code == 200
    assert resp.get_json()["indexed"] == 4

    resp = client.post("/query", json={"question": "What stores embeddings?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert data["question"] == "What stores embeddings?"
    assert data["model"] == "qwen3:1.7b"


def test_chunked_file_ingestion_via_api(client: FlaskClient, tmp_path: Path) -> None:
    """Load a large file, chunk it, ingest via API, and query."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    # Create a file large enough to be chunked
    content = (
        "Python is a versatile programming language. " * 20
        + "Raspberry Pi runs Linux natively. " * 20
    )
    (doc_dir / "knowledge.txt").write_text(content)

    chunks, ids = load_text_files(str(doc_dir), chunk_size=200)
    assert len(chunks) > 1  # Should be chunked

    resp = client.post("/ingest", json={"texts": chunks, "ids": ids})
    assert resp.status_code == 200
    assert resp.get_json()["indexed"] == len(chunks)

    resp = client.post("/query", json={"question": "What runs Linux?"})
    assert resp.status_code == 200
    assert "answer" in resp.get_json()


def test_empty_collection_query(rag_engine: RAGEngine) -> None:
    """Querying an empty collection should return empty results, not crash."""
    results = rag_engine.query("anything at all")
    assert results["documents"] == [[]]
    assert results["ids"] == [[]]


def test_generate_answer_empty_collection(rag_engine: RAGEngine) -> None:
    """generate_answer on empty collection should still return a string."""
    answer = rag_engine.generate_answer("hello?")
    assert isinstance(answer, str)
