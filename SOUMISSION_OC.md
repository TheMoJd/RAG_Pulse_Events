# Soumission projet OC — Puls-Events

**Assistant intelligent de recommandation d'événements culturels (POC RAG)**

|  |  |
|---|---|
| Candidat | TheMoJd |
| Email | moetez@polaria.ai |
| Projet | Puls-Events — POC RAG (LangChain + Mistral + FAISS) |
| Dépôt GitHub | **https://github.com/TheMoJd/RAG_Pulse_Events** |
| Branche évaluée | `main` |
| Date de soumission | 2026-05-03 |

---

## Résumé

POC d'un assistant conversationnel qui répond en langage naturel à des questions sur les événements culturels à venir à Paris (concerts, expositions, spectacles…), en s'appuyant sur les données publiques de l'API Open Agenda.

L'architecture combine **recherche sémantique** (embeddings Mistral + index FAISS, ~1 564 chunks) et **génération ancrée** (`mistral-medium-latest`), avec une triple protection contre les événements passés (filtre ODSQL côté API, filtre Python au retrieve, prompt système strict). L'ensemble est exposé par une **API REST FastAPI** documentée par Swagger, conteneurisée via Docker, et démontrée par un frontend React minimal.

L'évaluation de la qualité des réponses est automatisée via **Ragas** (4 métriques) et a fait l'objet d'une itération méthodologique v1 → v2 documentée.

---

## Mapping livrables ⇄ fichiers du dépôt

Tous les liens pointent vers la branche `main`.

| Livrable demandé par l'énoncé | Où le trouver |
|---|---|
| **Système RAG fonctionnel (LangChain, Mistral, FAISS)** | [`app/rag_chain.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/app/rag_chain.py) · [`utils/vector_store.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/utils/vector_store.py) · [`utils/openagenda_loader.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/utils/openagenda_loader.py) |
| **Script de reconstruction de l'index vectoriel** | [`indexer.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/indexer.py) (CLI : `python indexer.py`) |
| **API REST avec endpoints `/ask` et `/rebuild`** | [`app/main.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/app/main.py) — endpoints `/ask`, `/rebuild`, `/health`, Swagger sur `/docs` |
| **Rapport technique** | [`RAPPORT_TECHNIQUE.md`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/RAPPORT_TECHNIQUE.md) (10 sections, ~750 lignes) |
| **README clair (objectifs, structure, instructions)** | [`README.md`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/README.md) |
| **Jeu de test annoté (questions/réponses de référence)** | [`tests/qa_dataset.json`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/tests/qa_dataset.json) — 15 questions, 4 catégories |
| **Tests unitaires** | [`tests/`](https://github.com/TheMoJd/RAG_Pulse_Events/tree/main/tests) — 20 tests (`test_api.py`, `test_vector_store.py`, `test_openagenda_loader.py`) lancés via `pytest -m "not live"` |
| **Automatisation des métriques d'évaluation (Ragas)** | [`evaluate_rag.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/evaluate_rag.py) — pipeline Ragas 4 métriques, juge `mistral-medium-latest` |
| **Conteneurisation Docker** | [`Dockerfile.api`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/Dockerfile.api) · [`docker-compose.yml`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/docker-compose.yml) (services `api` + `frontend`) |
| **Présentation PowerPoint (10-15 slides)** | [`PRESENTATION.pptx`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/PRESENTATION.pptx) — **14 slides** natives éditables |
| **Fichier de dépendances** | [`requirements.txt`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/requirements.txt) · [`.env.example`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/.env.example) |

### Bonus livrés

| Élément | Pourquoi |
|---|---|
| [`docs/walkthroughs/`](https://github.com/TheMoJd/RAG_Pulse_Events/tree/main/docs/walkthroughs) (8 fichiers) | Walkthroughs pédagogiques par module pour aider l'évaluateur à parcourir le code. |
| [`frontend/`](https://github.com/TheMoJd/RAG_Pulse_Events/tree/main/frontend) (React + TS + Tailwind) | Interface chat avec rendu Markdown (`react-markdown`), cartes événements cliquables, suggestions contextuelles. |
| [`regenerate_qa_dataset.py`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/regenerate_qa_dataset.py) | Outil reproduisant la méthodologie v1 → v2 (model-generated GT + validation humaine). |
| [`evaluation_results.json`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/evaluation_results.json) + [`evaluation_results_v2.json`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/evaluation_results_v2.json) | Snapshots des deux passes Ragas, preuves de l'amélioration. |
| [`data/raw/events_paris_2026-05-02.json`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/data/raw/events_paris_2026-05-02.json) | Snapshot daté des données Open Agenda → reproductibilité offline. |

---

## Comment reproduire et tester (5 minutes)

```bash
# 1. Cloner
git clone https://github.com/TheMoJd/RAG_Pulse_Events.git
cd RAG_Pulse_Events

# 2. Configurer la clé Mistral
cp .env.example .env
# éditer .env et coller MISTRAL_API_KEY=...

# 3. Lancer via Docker
docker compose up --build

# 4. Tester l'API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels concerts à Paris ce week-end ?"}'

# 5. (optionnel) Frontend chat
# http://localhost:5173
```

L'index vectoriel est reconstructible à la demande :
```bash
curl -X POST http://localhost:8000/rebuild
# ou en CLI hors Docker :
python indexer.py
```

Lancer les tests unitaires :
```bash
pytest -m "not live"   # 20 tests offline avec mocks Mistral
```

Lancer l'évaluation Ragas complète :
```bash
python evaluate_rag.py   # nécessite la clé Mistral, ~5 min
```

---

## Résultats d'évaluation (Ragas, v2)

Méthode : 15 questions annotées, juge `mistral-medium-latest` (T=0).

| Métrique | v2 | Δ vs v1 |
|---|---|---|
| `faithfulness` | **0.50** | -15 pts (à creuser : voir RAPPORT §7) |
| `answer_relevancy` | **0.63** | +5 pts |
| `context_precision` | **0.46** | **+35 pts** |
| `context_recall` | **0.66** | **+59 pts** |

L'écart v1 → v2 sur `context_recall` (×9) illustre l'importance de calibrer le jeu d'évaluation sur la **distribution réelle des données indexées**, pas sur des intuitions a priori. La méthodologie est documentée dans [RAPPORT_TECHNIQUE.md §7](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/RAPPORT_TECHNIQUE.md#7-évaluation-du-système).

---

## Démonstration prévue en soutenance (3 scénarios)

| # | Question | Comportement attendu |
|---|---|---|
| 1 ✅ | « Quels concerts à Paris ce week-end ? » | 3 events datés et géolocalisés, format markdown (titre / 📅 / 📍 / 🔗), cartes cliquables côté frontend. |
| 2 ⚠️ | « Y a-t-il un concert de rap à Paris cet été ? » | Refus poli + suggestion d'élargir (rule 2) ; sources masquées par `is_refusal()` pour cohérence UX. |
| 3 🚫 | « Quelle est la météo à Paris demain ? » | Refus hors-domaine (rule 3) ; pas de cartes affichées. |

---

## Architecture en une phrase

`Frontend React → POST /ask → embed(question) → FAISS top-k → filter past → contexte XML → Mistral chat → réponse markdown + sources` — détails complets dans [`RAPPORT_TECHNIQUE.md §2`](https://github.com/TheMoJd/RAG_Pulse_Events/blob/main/RAPPORT_TECHNIQUE.md#2-architecture-du-système).

---

**Fin du document de soumission — TheMoJd, 2026-05-03.**
