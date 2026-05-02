"""Tests de l'API FastAPI avec TestClient + mock du RAGChain."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Construit un TestClient avec un RAGChain mocké et un vector_store factice non-vide."""
    from app import main as app_main

    fake_vs = MagicMock()
    fake_vs.size = 42

    fake_rag = MagicMock()
    fake_rag.ask.return_value = {
        "answer": "Je recommande le concert de jazz au Vauban samedi.",
        "sources": [
            {
                "title": "Concert Jazz au Vauban",
                "url": "https://example.com/event/123",
                "daterange": "Samedi 18h",
                "location_name": "Cabaret Vauban",
                "image": None,
                "score": 87.5,
            }
        ],
    }

    # Patch le state global initialisé par lifespan
    monkeypatch.setitem(app_main.state, "vector_store", fake_vs)
    monkeypatch.setitem(app_main.state, "rag_chain", fake_rag)

    # On utilise le client sans déclencher lifespan (pour ne pas écraser le state)
    with TestClient(app_main.app) as c:
        # Re-patch après le startup lifespan (qui a remplacé le state)
        app_main.state["vector_store"] = fake_vs
        app_main.state["rag_chain"] = fake_rag
        yield c


def test_health_returns_index_size(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_size"] == 42
    assert "model" in body


def test_root_returns_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "name" in body and "docs" in body


def test_ask_with_valid_question(client):
    r = client.post("/ask", json={"question": "Quels concerts à Paris ?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "sources" in body
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["title"] == "Concert Jazz au Vauban"


def test_ask_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # Pydantic min_length=1


def test_ask_503_when_index_empty(monkeypatch):
    """Si l'index est vide, /ask doit renvoyer 503."""
    from app import main as app_main

    empty_vs = MagicMock()
    empty_vs.size = 0
    monkeypatch.setitem(app_main.state, "vector_store", empty_vs)
    with TestClient(app_main.app) as c:
        app_main.state["vector_store"] = empty_vs  # re-patch après lifespan
        r = c.post("/ask", json={"question": "test"})
    assert r.status_code == 503


def test_rebuild_calls_indexer(monkeypatch, client):
    """Mock run_indexing pour ne pas refetch l'API."""
    from app import main as app_main

    monkeypatch.setattr(app_main, "run_indexing", lambda **kwargs: 100)

    fake_vs = MagicMock()
    fake_vs.size = 100
    monkeypatch.setattr(app_main, "VectorStoreManager", lambda: fake_vs)

    r = client.post("/rebuild", json={"use_snapshot": True})
    assert r.status_code == 200
    body = r.json()
    assert body["rebuilt"] is True
    assert body["n_chunks"] == 100
