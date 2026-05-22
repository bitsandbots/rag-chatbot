"""Flask API server for the RAG chatbot."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from rag_chatbot.rag_engine import RAGEngine


def create_app(rag: RAGEngine | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        rag: Optional RAGEngine instance. If None, creates one from env vars.
    """
    app = Flask(__name__)

    if rag is None:
        rag = RAGEngine(
            collection_name=os.getenv("COLLECTION_NAME", "documents"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            gen_model=os.getenv("MODEL", "qwen3:1.7b"),
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "engine": "local"})

    @app.route("/ingest", methods=["POST"])
    def ingest():
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "texts must be a non-empty list of strings"}), 400
        if not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "texts must be a non-empty list of strings"}), 400
        ids = data.get("ids")
        if ids is not None and len(ids) != len(texts):
            return jsonify({"error": "ids length must match texts length"}), 400
        rag.add_documents(texts, ids)
        return jsonify({"indexed": len(texts)})

    @app.route("/query", methods=["POST"])
    def query():
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        if not question:
            return jsonify({"error": "question required"}), 400
        answer = rag.generate_answer(question)
        return jsonify(
            {
                "question": question,
                "answer": answer,
                "model": rag.gen_model,
            }
        )

    @app.route("/documents", methods=["GET"])
    def list_documents():
        docs = rag.list_documents()
        return jsonify({"documents": docs, "count": len(docs)})

    @app.route("/documents", methods=["DELETE"])
    def delete_documents():
        data = request.get_json(silent=True) or {}
        ids = data.get("ids")
        if not isinstance(ids, list) or not ids:
            return jsonify({"error": "ids must be a non-empty list of strings"}), 400
        if not all(isinstance(i, str) for i in ids):
            return jsonify({"error": "ids must be a non-empty list of strings"}), 400
        deleted = rag.delete_documents(ids)
        return jsonify({"deleted": deleted})

    return app
