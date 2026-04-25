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
        data = request.get_json()
        texts = data.get("texts", [])
        ids = data.get("ids")
        rag.add_documents(texts, ids)
        return jsonify({"indexed": len(texts)})

    @app.route("/query", methods=["POST"])
    def query():
        data = request.get_json()
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

    return app
