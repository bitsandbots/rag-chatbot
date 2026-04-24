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
