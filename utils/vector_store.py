"""
Gestion de l'index FAISS + embeddings Mistral.

Adapté de SimpleRAGMistral pour mistralai>=1.x (interface `Mistral` au lieu de `MistralClient`).
Index: cosine similarity via IndexFlatIP + normalisation L2.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    MISTRAL_API_KEY,
    VECTOR_DB_DIR,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index FAISS."""

    def __init__(self, mistral_client: Optional[Mistral] = None):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: list[dict] = []
        self.mistral_client = mistral_client or (
            Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        )
        self._load_index_and_chunks()

    # -- chargement / sauvegarde --

    def _load_index_and_chunks(self) -> None:
        if FAISS_INDEX_FILE.exists() and DOCUMENT_CHUNKS_FILE.exists():
            try:
                logger.info(f"Chargement de l'index FAISS depuis {FAISS_INDEX_FILE}")
                self.index = faiss.read_index(str(FAISS_INDEX_FILE))
                with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                    self.document_chunks = pickle.load(f)
                logger.info(
                    f"  {self.index.ntotal} vecteurs / {len(self.document_chunks)} chunks chargés."
                )
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'index: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logger.warning("Aucun index FAISS existant. État: vide.")

    def _save_index_and_chunks(self) -> None:
        if self.index is None or not self.document_chunks:
            logger.warning("Sauvegarde ignorée: index ou chunks vides.")
            return
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
            pickle.dump(self.document_chunks, f)
        logger.info(f"Index et chunks sauvegardés dans {VECTOR_DB_DIR}.")

    # -- pipeline d'indexation --

    def _split_documents_to_chunks(self, documents: list[dict]) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        all_chunks: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            lc_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = splitter.split_documents([lc_doc])
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "id": f"{doc_idx}_{chunk_idx}",
                        "text": chunk.page_content,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_id_in_doc": chunk_idx,
                        },
                    }
                )
        logger.info(f"{len(all_chunks)} chunks créés à partir de {len(documents)} documents.")
        return all_chunks

    def _generate_embeddings(self, chunks: list[dict]) -> Optional[np.ndarray]:
        if self.mistral_client is None:
            logger.error("Client Mistral indisponible (clé API manquante).")
            return None
        if not chunks:
            return None

        all_embeddings: list[list[float]] = []
        n_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            logger.info(f"  Embeddings lot {batch_num}/{n_batches} ({len(texts)} chunks)")
            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL, inputs=texts
                )
                all_embeddings.extend(d.embedding for d in response.data)
            except Exception as e:
                logger.error(f"  Échec lot {batch_num}: {e}")
                return None

        arr = np.array(all_embeddings, dtype="float32")
        logger.info(f"Embeddings générés: shape={arr.shape}")
        return arr

    def build_index(self, documents: list[dict]) -> None:
        """Construit l'index FAISS à partir d'une liste de documents `{page_content, metadata}`."""
        if not documents:
            logger.warning("Aucun document fourni pour l'indexation.")
            return

        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logger.error("Découpage vide, indexation abandonnée.")
            return

        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logger.error("Mismatch embeddings/chunks, indexation abandonnée.")
            self.document_chunks = []
            self.index = None
            return

        # Cosine via IndexFlatIP + normalisation L2
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"Index FAISS construit: {self.index.ntotal} vecteurs (dim={dim}).")

        self._save_index_and_chunks()

    # -- recherche --

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index is None or not self.document_chunks:
            logger.warning("Recherche impossible: index vide.")
            return []
        if self.mistral_client is None:
            logger.error("Recherche impossible: client Mistral indisponible.")
            return []

        try:
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL, inputs=[query]
            )
            query_emb = np.array([response.data[0].embedding], dtype="float32")
            faiss.normalize_L2(query_emb)
            scores, indices = self.index.search(query_emb, k)
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return []

        results: list[dict] = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                results.append(
                    {
                        "score": float(scores[0][rank]) * 100,  # cosine en %
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    }
                )
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0
