"""Tests pour utils.vector_store (mock du client Mistral)."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from utils import config as cfg
from utils.openagenda_loader import events_to_documents, load_events_from_snapshot
from utils.vector_store import VectorStoreManager

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "events_sample.json"
EMB_DIM = 1024  # mistral-embed = 1024


def _make_mock_client(seed: int = 42) -> MagicMock:
    """Mock du client Mistral qui renvoie des embeddings aléatoires déterministes."""
    rng = np.random.default_rng(seed)

    def fake_embed(model: str, inputs: list[str]):
        response = MagicMock()
        response.data = [
            MagicMock(embedding=rng.standard_normal(EMB_DIM).astype("float32").tolist())
            for _ in inputs
        ]
        return response

    client = MagicMock()
    client.embeddings.create.side_effect = fake_embed
    return client


@pytest.fixture
def isolated_vector_db(tmp_path, monkeypatch):
    """Redirige les chemins de l'index vers un répertoire temporaire."""
    monkeypatch.setattr(cfg, "VECTOR_DB_DIR", tmp_path)
    monkeypatch.setattr(cfg, "FAISS_INDEX_FILE", tmp_path / "faiss_index.idx")
    monkeypatch.setattr(cfg, "DOCUMENT_CHUNKS_FILE", tmp_path / "document_chunks.pkl")
    # Réimporter aussi les constantes captées dans vector_store
    from utils import vector_store as vs

    monkeypatch.setattr(vs, "VECTOR_DB_DIR", tmp_path)
    monkeypatch.setattr(vs, "FAISS_INDEX_FILE", tmp_path / "faiss_index.idx")
    monkeypatch.setattr(vs, "DOCUMENT_CHUNKS_FILE", tmp_path / "document_chunks.pkl")
    return tmp_path


@pytest.fixture
def documents() -> list[dict]:
    events = load_events_from_snapshot(FIXTURE_PATH)
    return events_to_documents(events)


def test_build_index_creates_vectors(isolated_vector_db, documents):
    mgr = VectorStoreManager(mistral_client=_make_mock_client())
    mgr.build_index(documents)
    assert mgr.index is not None
    assert mgr.index.ntotal == len(mgr.document_chunks)
    assert mgr.size > 0


def test_build_index_persists_files(isolated_vector_db, documents):
    mgr = VectorStoreManager(mistral_client=_make_mock_client())
    mgr.build_index(documents)
    assert (isolated_vector_db / "faiss_index.idx").exists()
    assert (isolated_vector_db / "document_chunks.pkl").exists()


def test_index_reload_roundtrip(isolated_vector_db, documents):
    mgr1 = VectorStoreManager(mistral_client=_make_mock_client())
    mgr1.build_index(documents)
    n_before = mgr1.size
    chunks_before = len(mgr1.document_chunks)

    # Nouveau manager → doit recharger depuis disque
    mgr2 = VectorStoreManager(mistral_client=_make_mock_client())
    assert mgr2.size == n_before
    assert len(mgr2.document_chunks) == chunks_before


def test_search_returns_top_k(isolated_vector_db, documents):
    mgr = VectorStoreManager(mistral_client=_make_mock_client())
    mgr.build_index(documents)
    results = mgr.search("concert jazz", k=3)
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    for r in results:
        assert "text" in r and "metadata" in r and "score" in r
        assert isinstance(r["score"], float)


def test_search_on_empty_index(isolated_vector_db):
    mgr = VectorStoreManager(mistral_client=_make_mock_client())
    assert mgr.search("anything", k=3) == []


def test_build_index_handles_empty_documents(isolated_vector_db):
    mgr = VectorStoreManager(mistral_client=_make_mock_client())
    mgr.build_index([])
    assert mgr.size == 0
