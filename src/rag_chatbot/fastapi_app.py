"""FastAPI backend for production RAG chatbot."""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import ollama as ollama_client

from rag_chatbot.rag_engine import RAGEngine

app = FastAPI(title="RAG Chatbot API")

rag = RAGEngine(
    collection_name=os.getenv("COLLECTION_NAME", "documents"),
    chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
    embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    gen_model=os.getenv("MODEL", "qwen3:1.7b"),
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "engine": "local", "mode": "production"}


@app.post("/api/ingest")
async def ingest(request: Request) -> dict:
    data = await request.json()
    texts = data.get("texts", [])
    ids = data.get("ids")
    rag.add_documents(texts, ids)
    return {"indexed": len(texts)}


@app.post("/api/query")
async def query(request: Request) -> dict:
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "question required"}
    answer = rag.generate_answer(question)
    return {"question": question, "answer": answer, "model": rag.gen_model}


@app.post("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    """SSE endpoint for streaming token-by-token responses."""
    data = await request.json()
    question = data.get("question", "")

    results = rag.query(question)
    context = "\n\n".join(results["documents"][0]) if results["documents"][0] else ""

    prompt = f"""Answer based on the following context:
Context: {context}
Question: {question}
Answer:"""

    def event_stream():
        for chunk in ollama_client.generate(
            model=rag.gen_model, prompt=prompt, stream=True
        ):
            token = chunk.get("response", "")
            done = chunk.get("done", False)
            if done:
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/documents")
async def list_documents() -> dict:
    docs = rag.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/api/documents")
async def delete_documents(request: Request) -> dict:
    from fastapi import HTTPException

    data = await request.json()
    ids = data.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="ids must be a non-empty list of strings")
    if not all(isinstance(i, str) for i in ids):
        raise HTTPException(status_code=400, detail="ids must be a non-empty list of strings")
    deleted = rag.delete_documents(ids)
    return {"deleted": deleted}
