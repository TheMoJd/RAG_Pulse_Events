"""
CLI d'indexation: récupère les événements Open Agenda pour Brest, les vectorise via Mistral
et construit l'index FAISS persisté dans `vector_db/`.

Usage:
    python indexer.py                       # fetch live + index
    python indexer.py --use-snapshot        # repart du dernier snapshot data/raw/
    python indexer.py --since-days 180      # fenêtre temporelle personnalisée
    python indexer.py --city "Rennes"       # autre ville (override .env)
"""
import argparse
import logging
import sys
from pathlib import Path

from utils.config import RAW_DATA_DIR, SINCE_DAYS, TARGET_CITY
from utils.openagenda_loader import (
    events_to_documents,
    fetch_brest_events,
    load_latest_snapshot,
)
from utils.vector_store import VectorStoreManager


def run_indexing(city: str, since_days: int, use_snapshot: bool) -> int:
    """Pipeline complet: events → documents → chunks → embeddings → FAISS."""
    if use_snapshot:
        events = load_latest_snapshot(RAW_DATA_DIR)
        if not events:
            logging.error(f"Aucun snapshot trouvé dans {RAW_DATA_DIR}. Lance sans --use-snapshot.")
            return 0
    else:
        events = fetch_brest_events(city=city, since_days=since_days, save_snapshot=True)

    if not events:
        logging.warning("Aucun événement à indexer.")
        return 0

    documents = events_to_documents(events)
    if not documents:
        logging.error("Aucun document valide après conversion.")
        return 0

    vs = VectorStoreManager()
    vs.build_index(documents)
    return vs.size


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Indexer Puls-Events RAG")
    parser.add_argument("--city", default=TARGET_CITY, help="Ville cible (par défaut: Brest)")
    parser.add_argument(
        "--since-days",
        type=int,
        default=SINCE_DAYS,
        help="Nombre de jours d'historique (par défaut: 365)",
    )
    parser.add_argument(
        "--use-snapshot",
        action="store_true",
        help="Repart du snapshot le plus récent au lieu de fetch l'API",
    )
    args = parser.parse_args()

    logging.info(f"=== Indexation Puls-Events ===")
    logging.info(f"  ville: {args.city}")
    logging.info(f"  fenêtre: {args.since_days} jours")
    logging.info(f"  source: {'snapshot local' if args.use_snapshot else 'API live'}")

    n = run_indexing(args.city, args.since_days, args.use_snapshot)
    if n > 0:
        logging.info(f"✅ Indexation terminée: {n} chunks dans l'index FAISS.")
        return 0
    logging.error("❌ Indexation échouée ou vide.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
