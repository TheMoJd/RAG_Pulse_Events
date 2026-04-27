"""
Loader pour les événements Open Agenda (via dataset public Opendatasoft).

Endpoint: https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records
- Pas d'authentification requise
- ODSQL pour le filtrage (where, select, etc.)
- Pagination via limit (max 100) + offset
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from .config import (
    OPENAGENDA_BASE_URL,
    RAW_DATA_DIR,
    SINCE_DAYS,
    TARGET_CITY,
)

logger = logging.getLogger(__name__)

PAGE_SIZE = 100  # max autorisé par Opendatasoft v2.1
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def _build_where_clause(city: str, since_date: str) -> str:
    """Construit la clause ODSQL pour filtrer par ville et date de début >= since_date."""
    return f'location_city="{city}" AND firstdate_begin >= date\'{since_date}\''


def fetch_brest_events(
    city: str = TARGET_CITY,
    since_days: int = SINCE_DAYS,
    page_size: int = PAGE_SIZE,
    save_snapshot: bool = True,
    snapshot_dir: Path = RAW_DATA_DIR,
) -> list[dict]:
    """
    Récupère tous les événements Open Agenda pour une ville sur les `since_days` derniers jours
    et le futur. Pagine jusqu'à épuisement.

    Args:
        city: nom de la ville (filtre exact sur location_city).
        since_days: nombre de jours à rétrocéder pour la borne basse.
        page_size: taille de page (max 100 sur Opendatasoft).
        save_snapshot: si True, écrit data/raw/events_<city>_<date>.json.
        snapshot_dir: répertoire pour le snapshot.

    Returns:
        Liste brute des records Open Agenda.
    """
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d")
    where = _build_where_clause(city, since_date)

    all_events: list[dict] = []
    offset = 0
    total_count: Optional[int] = None

    logger.info(f"Récupération des événements pour '{city}' depuis {since_date}...")

    while True:
        params = {
            "select": "*",
            "where": where,
            "limit": page_size,
            "offset": offset,
            "order_by": "firstdate_begin",
        }
        data = _request_with_retries(OPENAGENDA_BASE_URL, params)
        if data is None:
            logger.error(f"Échec de la requête à l'offset {offset}, arrêt.")
            break

        if total_count is None:
            total_count = data.get("total_count", 0)
            logger.info(f"  total_count = {total_count}")

        results = data.get("results", [])
        if not results:
            break

        all_events.extend(results)
        offset += len(results)
        logger.info(f"  récupéré {len(all_events)}/{total_count}")

        if total_count is not None and offset >= total_count:
            break
        # Garde-fou: Opendatasoft plafonne souvent l'offset à 10000
        if offset >= 10000:
            logger.warning("Plafond d'offset Opendatasoft atteint (10000). Arrêt anticipé.")
            break

    logger.info(f"✅ {len(all_events)} événements récupérés pour {city}.")

    if save_snapshot and all_events:
        path = _save_snapshot(all_events, city, snapshot_dir)
        logger.info(f"📁 Snapshot écrit: {path}")

    return all_events


def _request_with_retries(url: str, params: dict) -> Optional[dict]:
    """GET avec retry simple sur 429/5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                wait = 2**attempt
                logger.warning(
                    f"  HTTP {response.status_code} (tentative {attempt}/{MAX_RETRIES}), "
                    f"retry dans {wait}s..."
                )
                time.sleep(wait)
                continue
            logger.error(f"  HTTP {response.status_code}: {response.text[:200]}")
            return None
        except requests.RequestException as e:
            logger.warning(f"  Erreur réseau (tentative {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(2**attempt)
    return None


def _save_snapshot(events: list[dict], city: str, snapshot_dir: Path) -> Path:
    """Écrit la liste des events en JSON dans snapshot_dir."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = snapshot_dir / f"events_{city.lower()}_{today}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    return path


def load_events_from_snapshot(path: Path) -> list[dict]:
    """Charge un snapshot JSON depuis le disque (tests / démo offline)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_snapshot(snapshot_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """Charge le snapshot le plus récent dans snapshot_dir, ou [] si aucun."""
    if not snapshot_dir.exists():
        return []
    snapshots = sorted(snapshot_dir.glob("events_*.json"), reverse=True)
    if not snapshots:
        return []
    logger.info(f"Chargement du snapshot le plus récent: {snapshots[0]}")
    return load_events_from_snapshot(snapshots[0])


def event_to_document(event: dict) -> Optional[dict]:
    """
    Convertit un event Open Agenda brut en un document compatible avec VectorStoreManager:
        {"page_content": str, "metadata": dict}

    Renvoie None si le document n'a pas assez de contenu (pas de titre).
    """
    title = (event.get("title_fr") or "").strip()
    if not title:
        return None

    description = (event.get("description_fr") or "").strip()
    long_description = (event.get("longdescription_fr") or "").strip()
    keywords = event.get("keywords_fr") or []
    if isinstance(keywords, list):
        keywords_str = ", ".join(str(k) for k in keywords if k)
    else:
        keywords_str = str(keywords)

    daterange = (event.get("daterange_fr") or "").strip()
    location_name = (event.get("location_name") or "").strip()
    location_address = (event.get("location_address") or "").strip()
    location_city = (event.get("location_city") or "").strip()
    location_postalcode = (event.get("location_postalcode") or "").strip()

    # Compose un texte enrichi pour l'embedding
    parts = [f"Titre: {title}"]
    if description:
        parts.append(f"Description: {description}")
    if long_description and long_description != description:
        parts.append(f"Détails: {long_description}")
    if daterange:
        parts.append(f"Dates: {daterange}")
    if location_name or location_address or location_city:
        loc = ", ".join(p for p in [location_name, location_address, location_postalcode, location_city] if p)
        parts.append(f"Lieu: {loc}")
    if keywords_str:
        parts.append(f"Mots-clés: {keywords_str}")

    page_content = "\n".join(parts)

    metadata = {
        "uid": event.get("uid"),
        "title": title,
        "url": event.get("canonicalurl"),
        "image": event.get("image"),
        "daterange": daterange,
        "date_begin": event.get("firstdate_begin"),
        "date_end": event.get("lastdate_end"),
        "city": location_city,
        "location_name": location_name,
        "address": location_address,
        "source": "openagenda",
    }

    return {"page_content": page_content, "metadata": metadata}


def events_to_documents(events: list[dict]) -> list[dict]:
    """Convertit une liste d'events en documents indexables (filtre les invalides)."""
    docs = []
    for ev in events:
        doc = event_to_document(ev)
        if doc:
            docs.append(doc)
    logger.info(f"{len(docs)}/{len(events)} événements convertis en documents.")
    return docs
