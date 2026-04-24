"""Local RAG pipeline using ChromaDB + Ollama."""

from __future__ import annotations

import chromadb
import ollama


class RAGEngine:
    """Local RAG pipeline using ChromaDB + Ollama."""

    def __init__(
        self,
        collection_name: str = "documents",
        chroma_path: str = "./chroma_db",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "qwen2.5-coder:3b",
    ) -> None:
        """Initialize the RAG engine with ChromaDB and Ollama.

        Args:
            collection_name: Name of the ChromaDB collection.
            chroma_path: Path to persist ChromaDB data.
            embed_model: Ollama embedding model name.
            gen_model: Ollama generation model name.
        """
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embeddings via Ollama.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = ollama.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]

    def add_documents(self, texts: list[str], ids: list[str] | None = None) -> None:
        """Index documents into ChromaDB.

        Args:
            texts: List of document strings to index.
            ids: Optional list of IDs; auto-generated as doc_N if omitted.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = [self.embed_text(t) for t in texts]
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
        )

    def query(self, question: str, n_results: int = 3) -> dict:
        """Retrieve relevant documents for a question.

        Args:
            question: The query string.
            n_results: Number of results to return.

        Returns:
            ChromaDB query result dict with 'documents', 'ids', etc.
        """
        query_embedding = self.embed_text(question)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def generate_answer(self, question: str) -> str:
        """Full RAG pipeline: retrieve context, generate answer.

        Args:
            question: The user's question.

        Returns:
            Generated answer string from the language model.
        """
        results = self.query(question)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""Answer based on the following context:
Context: {context}
Question: {question}
Answer:"""

        response = ollama.generate(model=self.gen_model, prompt=prompt)
        return response["response"]
