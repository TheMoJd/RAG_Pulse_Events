"""Tests pour utils.openagenda_loader."""
import json
from pathlib import Path

import pytest

from utils.openagenda_loader import (
    _build_where_clause,
    event_to_document,
    events_to_documents,
    fetch_city_events,
    load_events_from_snapshot,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "events_sample.json"


@pytest.fixture(scope="module")
def sample_events() -> list[dict]:
    return load_events_from_snapshot(FIXTURE_PATH)


def test_fixture_exists_and_non_empty(sample_events):
    assert len(sample_events) >= 5, "Fixture trop petite pour des tests utiles."


def test_build_where_clause_format():
    clause = _build_where_clause("Paris", lookahead_days=30)
    assert 'location_city="Paris"' in clause
    assert "lastdate_end >= date'" in clause      # borne basse : pas terminé
    assert "firstdate_begin <= date'" in clause   # borne haute : commence avant horizon


def test_event_to_document_basic_shape(sample_events):
    doc = event_to_document(sample_events[0])
    assert doc is not None
    assert "page_content" in doc
    assert "metadata" in doc
    assert isinstance(doc["page_content"], str) and doc["page_content"].strip()
    assert "Titre:" in doc["page_content"]


def test_event_to_document_metadata_fields(sample_events):
    doc = event_to_document(sample_events[0])
    md = doc["metadata"]
    for key in ("uid", "title", "url", "daterange", "date_begin", "city", "source"):
        assert key in md, f"Métadonnée manquante: {key}"
    assert md["source"] == "openagenda"


def test_event_to_document_returns_none_when_no_title():
    doc = event_to_document({"title_fr": "", "description_fr": "x"})
    assert doc is None
    doc2 = event_to_document({"title_fr": None})
    assert doc2 is None


def test_event_to_document_handles_null_fields():
    minimal = {
        "title_fr": "Concert test",
        "description_fr": None,
        "longdescription_fr": None,
        "keywords_fr": None,
        "location_city": None,
        "uid": "abc",
        "canonicalurl": None,
    }
    doc = event_to_document(minimal)
    assert doc is not None
    assert "Concert test" in doc["page_content"]


def test_events_to_documents_filters_invalid(sample_events):
    docs = events_to_documents(sample_events)
    assert len(docs) <= len(sample_events)
    assert all(d["page_content"] for d in docs)


def test_load_events_from_snapshot_roundtrip(tmp_path, sample_events):
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(sample_events[:3]), encoding="utf-8")
    loaded = load_events_from_snapshot(snap_path)
    assert len(loaded) == 3
    assert loaded[0]["uid"] == sample_events[0]["uid"]


@pytest.mark.live
def test_fetch_city_events_smoke():
    """Test live (nécessite internet). Lance avec: pytest -m live"""
    events = fetch_city_events(city="Paris", lookahead_days=30, page_size=5, save_snapshot=False)
    # Paris a beaucoup d'events, on devrait au moins en trouver quelques-uns
    assert isinstance(events, list)
