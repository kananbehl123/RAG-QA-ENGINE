import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_db_type" in data
    assert "stored_chunks_count" in data


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "p50_retrieval_latency_ms" in data


def test_query_endpoint():
    payload = {
        "question": "What vector database is used?",
        "top_k": 3
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "total_latency_ms" in data
    assert "retrieved_chunks" in data
