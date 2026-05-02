"""Pytest fixtures partagées."""
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_events() -> list[dict]:
    """10 événements réels sauvegardés depuis Open Agenda (fixture pour tests de parsing)."""
    with open(FIXTURES_DIR / "events_sample.json", "r", encoding="utf-8") as f:
        return json.load(f)
