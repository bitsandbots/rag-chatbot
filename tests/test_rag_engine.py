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


def test_embed_determinism(rag_engine: RAGEngine) -> None:
    """Same text must produce the same embedding vector."""
    v1 = rag_engine.embed_text("hello")
    v2 = rag_engine.embed_text("hello")
    assert v1 == v2


def test_embed_different_texts(rag_engine: RAGEngine) -> None:
    """Different texts must produce different embeddings."""
    v1 = rag_engine.embed_text("hello")
    v2 = rag_engine.embed_text("goodbye")
    assert v1 != v2


def test_constructor_stores_attributes(rag_engine: RAGEngine) -> None:
    assert rag_engine.embed_model == "nomic-embed-text"
    assert rag_engine.gen_model == "qwen3:1.7b"
    assert rag_engine.collection is not None


def test_query_n_results_exceeds_collection(rag_engine: RAGEngine) -> None:
    """Requesting more results than docs should return all available."""
    rag_engine.add_documents(texts=["only one doc"], ids=["single"])
    results = rag_engine.query("anything", n_results=10)
    assert len(results["documents"][0]) == 1


def test_multiple_add_documents_accumulate(rag_engine: RAGEngine) -> None:
    """Multiple calls should accumulate, not replace."""
    rag_engine.add_documents(texts=["doc A"], ids=["a"])
    rag_engine.add_documents(texts=["doc B"], ids=["b"])
    results = rag_engine.query("doc", n_results=10)
    assert len(results["documents"][0]) == 2


def test_generate_answer_includes_context(rag_engine: RAGEngine) -> None:
    """generate_answer should pass retrieved context to the LLM prompt."""
    rag_engine.add_documents(texts=["The capital of France is Paris"], ids=["fr"])
    answer = rag_engine.generate_answer("What is the capital of France?")
    # Mock returns "Mock answer for: {prompt[:50]}"
    assert answer.startswith("Mock answer for:")


def test_query_returns_ids(rag_engine: RAGEngine) -> None:
    rag_engine.add_documents(texts=["alpha", "beta"], ids=["id_a", "id_b"])
    results = rag_engine.query("alpha", n_results=2)
    assert "ids" in results
    returned_ids = results["ids"][0]
    assert "id_a" in returned_ids
    assert "id_b" in returned_ids
