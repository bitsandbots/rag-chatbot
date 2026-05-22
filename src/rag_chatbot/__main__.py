"""Entrypoint for python -m rag_chatbot and the rag-chatbot CLI command."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from rag_chatbot.api_server import create_app


def main() -> None:
    """Start the Flask server. Called by python -m rag_chatbot and the CLI."""
    load_dotenv()
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


if __name__ == "__main__":
    main()
