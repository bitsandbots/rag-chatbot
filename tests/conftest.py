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
