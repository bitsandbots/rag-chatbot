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
