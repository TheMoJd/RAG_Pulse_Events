"""Configuration centralisée du POC RAG Puls-Events."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Clé API Mistral ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️  MISTRAL_API_KEY non défini dans .env — embeddings et chat indisponibles.")

# --- Modèles Mistral ---
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
EMBEDDING_MODEL = "mistral-embed"

# --- Open Agenda (Opendatasoft public dataset) ---
OPENAGENDA_BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
TARGET_CITY = os.getenv("TARGET_CITY", "Brest")
SINCE_DAYS = int(os.getenv("SINCE_DAYS", "365"))

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index.idx"
DOCUMENT_CHUNKS_FILE = VECTOR_DB_DIR / "document_chunks.pkl"

# --- Indexation ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32

# --- Recherche ---
SEARCH_K = 5

# --- App ---
APP_TITLE = "Puls-Events — Assistant culturel"
