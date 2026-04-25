from __future__ import annotations

from flask.testing import FlaskClient


def test_health(client: FlaskClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["engine"] == "local"


def test_ingest(client: FlaskClient) -> None:
    resp = client.post(
        "/ingest",
        json={
            "texts": ["Raspberry Pi 5 has 8GB RAM"],
            "ids": ["pi5_ram"],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["indexed"] == 1


def test_query(client: FlaskClient) -> None:
    client.post(
        "/ingest",
        json={
            "texts": ["The sky is blue"],
            "ids": ["sky_0"],
        },
    )
    resp = client.post("/query", json={"question": "What color is the sky?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert data["question"] == "What color is the sky?"
    assert "model" in data


def test_query_missing_question(client: FlaskClient) -> None:
    resp = client.post("/query", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "question required"


def test_query_empty_string_question(client: FlaskClient) -> None:
    """Empty string should also trigger 400."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "question required"


def test_ingest_auto_ids(client: FlaskClient) -> None:
    """Omitting ids should still succeed (engine auto-generates)."""
    resp = client.post("/ingest", json={"texts": ["auto id doc"]})
    assert resp.status_code == 200
    assert resp.get_json()["indexed"] == 1


def test_ingest_multiple_documents(client: FlaskClient) -> None:
    resp = client.post(
        "/ingest",
        json={
            "texts": ["doc one", "doc two", "doc three"],
            "ids": ["d1", "d2", "d3"],
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["indexed"] == 3


def test_query_returns_model_name(client: FlaskClient) -> None:
    """Model field should match the engine's configured gen_model."""
    client.post("/ingest", json={"texts": ["test data"], "ids": ["t0"]})
    resp = client.post("/query", json={"question": "test?"})
    data = resp.get_json()
    assert data["model"] == "qwen3:1.7b"


def test_ingest_then_query_roundtrip(client: FlaskClient) -> None:
    """Full API round trip: ingest docs, then query should find them."""
    client.post(
        "/ingest",
        json={
            "texts": [
                "Python was created by Guido van Rossum",
                "JavaScript was created by Brendan Eich",
            ],
            "ids": ["py", "js"],
        },
    )
    resp = client.post("/query", json={"question": "Who created Python?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


def test_health_method_not_allowed(client: FlaskClient) -> None:
    """POST to /health should return 405."""
    resp = client.post("/health")
    assert resp.status_code == 405


def test_create_app_env_var_fallback(tmp_path, monkeypatch) -> None:
    """create_app() without rag arg should build from env vars."""
    from rag_chatbot.api_server import create_app

    monkeypatch.setenv("COLLECTION_NAME", "env_test")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("MODEL", "test-model")

    app = create_app()  # No rag arg — should use env vars
    assert app is not None
    # Verify the app works
    with app.test_client() as c:
        resp = c.get("/health")
        assert resp.status_code == 200
