"""Entrypoint for python -m rag_chatbot."""

import os

from dotenv import load_dotenv

from rag_chatbot.api_server import create_app

load_dotenv()

app = create_app()
app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
