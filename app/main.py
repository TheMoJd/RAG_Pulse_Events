"""API FastAPI exposant le RAG Puls-Events."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag_chain import RAGChain
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RebuildRequest,
    RebuildResponse,
)
from indexer import run_indexing
from utils.config import APP_TITLE, LOOKAHEAD_DAYS, MODEL_NAME, TARGET_CITY
from utils.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- État applicatif (chargé au startup, partagé entre requêtes) ---
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage de l'API Puls-Events")
    vs = VectorStoreManager()
    if vs.size == 0:
        logger.warning(
            "⚠️  Index FAISS vide. Lance `python indexer.py` ou POST /rebuild "
            "pour le construire."
        )
    state["vector_store"] = vs
    state["rag_chain"] = RAGChain(vector_store=vs)
    yield
    logger.info("🛑 Arrêt de l'API")


app = FastAPI(
    title=APP_TITLE,
    description=f"API RAG pour recommander des événements culturels à venir à {TARGET_CITY} "
    f"(fenêtre de {LOOKAHEAD_DAYS} jours) à partir des données Open Agenda.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — ouvert pour le POC (frontend en local + futur VPS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health() -> HealthResponse:
    vs: VectorStoreManager = state.get("vector_store")
    return HealthResponse(
        status="ok",
        index_size=vs.size if vs else 0,
        model=MODEL_NAME,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(req: AskRequest) -> AskResponse:
    """Pose une question et renvoie une réponse augmentée + sources."""
    vs: VectorStoreManager = state.get("vector_store")
    if vs is None or vs.size == 0:
        raise HTTPException(
            status_code=503,
            detail="Index FAISS vide ou non chargé. Appelle POST /rebuild d'abord.",
        )
    rag: RAGChain = state["rag_chain"]
    result = rag.ask(req.question)
    return AskResponse(**result)


@app.post("/rebuild", response_model=RebuildResponse, tags=["Admin"])
def rebuild(req: RebuildRequest = RebuildRequest()) -> RebuildResponse:
    """Reconstruit l'index FAISS à partir d'Open Agenda (ou d'un snapshot local)."""
    try:
        n_chunks = run_indexing(
            city=TARGET_CITY,
            lookahead_days=LOOKAHEAD_DAYS,
            use_snapshot=req.use_snapshot,
        )
    except Exception as e:
        logger.exception("Erreur pendant /rebuild")
        raise HTTPException(status_code=500, detail=f"Erreur reconstruction: {e}")

    if n_chunks == 0:
        raise HTTPException(status_code=500, detail="L'indexation n'a produit aucun chunk.")

    # Recharge le manager pour refléter le nouvel index
    new_vs = VectorStoreManager()
    state["vector_store"] = new_vs
    state["rag_chain"] = RAGChain(vector_store=new_vs)

    return RebuildResponse(
        rebuilt=True,
        n_chunks=n_chunks,
        message=f"Index reconstruit avec {n_chunks} chunks.",
    )


@app.get("/", tags=["Système"])
def root() -> dict:
    return {
        "name": APP_TITLE,
        "docs": "/docs",
        "endpoints": ["/health", "/ask (POST)", "/rebuild (POST)"],
    }
