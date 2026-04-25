# API Reference

## Starter Stack (Flask, port 5000)

### GET /health

Returns server status.

**Response**

```json
{"engine": "local", "status": "ok"}
```

**Status codes**: `200 OK`

**curl**

```bash
curl http://localhost:5000/health
```

---

### POST /ingest

Index document text into ChromaDB.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `texts` | `string[]` | Yes | List of document strings to embed and store |
| `ids` | `string[]` | No | Document IDs; auto-generated as `doc_0`, `doc_1`, ... if omitted |

**Response**

```json
{"indexed": 2}
```

`indexed` is the count of documents added.

**Status codes**: `200 OK`

**curl**

```bash
curl -s -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "The Raspberry Pi 5 features a 2.4GHz quad-core ARM Cortex-A76 CPU.",
      "Ollama supports running models offline without any cloud dependency."
    ],
    "ids": ["pi5-cpu", "ollama-offline"]
  }'
```

---

### POST /query

Run the full RAG pipeline: embed the question, retrieve relevant chunks, generate an answer.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | `string` | Yes | The user's question |

**Response**

```json
{
  "answer": "The Raspberry Pi 5 has a quad-core ARM Cortex-A76 CPU.",
  "model": "qwen3:1.7b",
  "question": "What CPU does the Raspberry Pi 5 use?"
}
```

**Status codes**

| Code | Condition |
|---|---|
| `200 OK` | Answer generated successfully |
| `400 Bad Request` | `question` field missing from request body |

**Error response (400)**

```json
{"error": "question required"}
```

**curl**

```bash
curl -s -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What CPU does the Raspberry Pi 5 use?"}'
```

---

## After Upgrade (FastAPI, port 7860)

Running `scripts/upgrade.sh` adds a FastAPI backend with additional endpoints.

### GET /api/health

```bash
curl http://localhost:7860/api/health
```

**Response**

```json
{"engine": "local", "mode": "production", "status": "ok"}
```

---

### POST /api/ingest

Same request/response contract as `POST /ingest` above.

```bash
curl -s -X POST http://localhost:7860/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["..."]}'
```

---

### POST /api/stream

Streaming token-by-token response via Server-Sent Events (SSE).

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | `string` | Yes | The user's question |

**Response format (SSE)**

Each event is a line starting with `data: ` followed by a JSON object, terminated by a blank line:

```
data: {"token": "The "}

data: {"token": "Raspberry "}

data: {"token": "Pi 5 "}

data: {"done": true}

```

The stream ends when `{"done": true}` is received.

**Content-Type**: `text/event-stream`

**curl**

```bash
curl -s -X POST http://localhost:7860/api/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What CPU does the Raspberry Pi 5 use?"}' \
  --no-buffer
```

---

### GET /chat

After upgrade, the Flask server also serves a browser-based chat UI.

```
http://localhost:5000/chat
```

The UI connects to `POST /query` and renders a streaming-compatible chat interface.
