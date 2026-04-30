"""Logique RAG: search FAISS → prompt Mistral → réponse + sources."""
import logging
from typing import Optional

from mistralai import Mistral

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """

Role : Tu es l'assistant culturel de Puls-Events pour la ville de Brest.

Objectif : Ta mission est de recommander des événements culturels (concerts, 
expositions, spectacles, festivals…) en t'appuyant **exclusivement** sur le CONTEXTE fourni.

Règles:
1. Base-toi UNIQUEMENT sur les événements du CONTEXTE. N'invente rien.
2. Si plusieurs événements correspondent, présente-en 2 à 4 avec leur titre, date et lieu.
3. Si le CONTEXTE ne contient pas d'événement pertinent, dis-le poliment et invite l'utilisateur à reformuler ou élargir sa recherche.
4. Reste chaleureux et concis. N'utilise pas de jargon technique.
5. Si une URL est disponible dans les métadonnées, mentionne-la pour permettre à l'utilisateur d'en savoir plus.
6. Réponds en français.

CONTEXTE:
---
{context}
---

QUESTION: {question}

RÉPONSE:"""


def _format_context(results: list[dict]) -> str:
    """Concatène les chunks récupérés en un bloc lisible pour le LLM."""
    if not results:
        return "Aucun événement pertinent trouvé dans la base."
    blocks = []
    for i, r in enumerate(results, 1):
        md = r.get("metadata", {})
        header_parts = [f"[Événement {i}]"]
        if md.get("title"):
            header_parts.append(f"Titre: {md['title']}")
        if md.get("daterange"):
            header_parts.append(f"Dates: {md['daterange']}")
        if md.get("location_name") or md.get("address"):
            loc = md.get("location_name") or md.get("address")
            header_parts.append(f"Lieu: {loc}")
        if md.get("url"):
            header_parts.append(f"URL: {md['url']}")
        header = " | ".join(header_parts)
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def _format_sources(results: list[dict]) -> list[dict]:
    """Convertit les résultats en payloads de sources pour l'API (dédupliqués par URL)."""
    seen: set[str] = set()
    sources: list[dict] = []
    for r in results:
        md = r.get("metadata", {})
        key = md.get("url") or md.get("uid") or md.get("title") or ""
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": md.get("title"),
                "url": md.get("url"),
                "daterange": md.get("daterange"),
                "location_name": md.get("location_name") or md.get("address"),
                "image": md.get("image"),
                "score": float(r.get("score", 0.0)),
            }
        )
    return sources


class RAGChain:
    """Encapsule la chaîne RAG. Réutilisée à chaque requête (clients persistants)."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        mistral_client: Optional[Mistral] = None,
        model: str = MODEL_NAME,
    ):
        self.vector_store = vector_store
        self.client = mistral_client or (
            Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        )
        self.model = model

    def ask(self, question: str, k: int = SEARCH_K) -> dict:
        """Renvoie {"answer": str, "sources": list[dict]}."""
        if not question or not question.strip():
            return {"answer": "Veuillez poser une question.", "sources": []}

        results = self.vector_store.search(question, k=k)
        context = _format_context(results)
        sources = _format_sources(results)

        if self.client is None:
            return {
                "answer": "Service indisponible (clé Mistral manquante).",
                "sources": sources,
            }

        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.exception("Erreur appel Mistral chat.complete")
            return {
                "answer": f"Erreur lors de la génération de la réponse: {e}",
                "sources": sources,
            }

        return {"answer": answer, "sources": sources}
