---
name: verify-chat-ui
description: Launch the rag-chatbot Flask app and drive its /chat UI end-to-end with a headless browser to confirm it renders and answers questions. Use when asked to run, verify, or screenshot this project's chat UI.
---

# Verify the rag-chatbot chat UI

The chat UI is a single self-contained page (`src/rag_chatbot/templates/chat.html`)
served by the Flask app at `GET /chat`. Its JS calls relative `POST /query` on the
same Flask server — no FastAPI (port 7860) or separate frontend build involved.

## 1. Launch

```bash
venv/bin/python -m rag_chatbot
```

Default `PORT=5000`. If that port is already taken (`Address already in use` in
the log), pick another: `PORT=5099 venv/bin/python -m rag_chatbot`. Run it in the
background (`nohup ... & disown`, or `run_in_background`) and poll `/health`
instead of sleeping a fixed time:

```bash
timeout 20 bash -c 'until curl -sf http://localhost:$PORT/health >/dev/null; do sleep 1; done'
```

`/health` and `/chat` both come up without a live Ollama daemon — the RAGEngine
only touches ChromaDB (a local path, `chroma_db/`) at startup. Ollama is only
needed when a question is actually submitted via `/query`.

## 2. Warm the model (avoid a flaky first request)

Ollama unloads idle models from memory, and reloading one can take well over
90s on constrained hardware — measured one cold load exceeding a 90s wait
during testing. Force the load before driving the UI so the browser-driven
request is fast and reliable:

```bash
curl -s -X POST http://localhost:$PORT/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "warmup"}' >/dev/null
```

## 3. Drive it

No `chromium-cli` or `playwright` package is preinstalled here, but the system
Chromium binary (`/usr/bin/chromium`) and npm registry access are available.
`scripts/check.js` in this skill drives the page with `playwright-core` pointed
at that binary (no browser download needed). One-time setup, then run:

```bash
cd .claude/skills/verify-chat-ui/scripts
npm install --no-audit --no-fund   # first run only, reads package.json
PORT=5099 node check.js
```

It navigates to `/chat`, reads the header text, submits a question through the
real input/button flow, waits for `.msg.assistant` to appear, and prints the
resulting messages plus any browser console errors. Screenshots land next to
the script as `1-loaded.png` and `2-response.png`.

**Gotcha:** generation latency is genuinely inconsistent even after warmup —
observed one `/query` take ~122s end-to-end with no client or server error,
just slow token generation. The script's `waitForSelector('.msg.assistant',
...)` timeout is set to 180s to absorb this; don't shorten it without
re-testing. A click on Send reliably fires the `POST /query` request every
time (verified with request-level tracing) — if a run does time out, that
means generation itself is unusually slow or stuck, not that the UI failed to
submit. Check `curl http://localhost:11434/api/tags` (is the model pulled?)
and `ollama ps` (is it still loaded / actually working) before concluding the
UI is broken.

## 4. Clean up

```bash
pkill -f rag_chatbot
ss -ltnp | grep $PORT   # confirm the port is freed
```
