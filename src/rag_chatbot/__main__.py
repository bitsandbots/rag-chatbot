"""Entrypoint for python -m rag_chatbot."""

from rag_chatbot.api_server import create_app

import os

from dotenv import load_dotenv

load_dotenv()

app = create_app()
app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
