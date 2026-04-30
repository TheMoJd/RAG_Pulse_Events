# Puls-Events — Assistant RAG pour événements culturels

POC d'un chatbot intelligent qui répond aux questions sur les **événements culturels à Brest** en s'appuyant sur les données publiques **Open Agenda**. Le système combine **recherche vectorielle FAISS** et **génération Mistral** orchestrées par **LangChain**, exposé par une **API REST FastAPI** et consommé par un **frontend React** esthétique.

> Mission OpenClassrooms — *Développez un assistant pour la recommandation d'événements culturels* — pour Puls-Events (cliente fictive).

## Démo en images

```
┌──────────────────────────────────────────────────────────┐
│  P   Puls-Events                          ●  590 chunks  │
│      Assistant culturel — Brest              indexés     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   👤 Quels concerts à Brest ce week-end ?               │
│                                                          │
│   🤖 Voici quelques idées :                              │
│      1. Un imaginaire, créer en sons (Vendredi 30/01)    │
│      2. Instants baroques (Temple Protestant)            │
│      3. Concert pop/rock avec Stellen                    │
│                                                          │
│      ┌──────────────────┐  ┌──────────────────┐          │
│      │ 🎵 Concert Jazz  │  │ 🎭 Festival Yves │          │
│      │ 📅 Sam 18h       │  │ 📅 14-24 mai     │          │
│      │ 📍 Cabaret Vaub. │  │ 📍 Vivre La Rue  │          │
│      │ 87% pertinent    │  │ 79% pertinent    │          │
│      └──────────────────┘  └──────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Architecture

```
┌──────────────┐    HTTP    ┌────────────┐    embed/search    ┌────────────┐
│  React (UI)  │ ─────────► │  FastAPI   │ ─────────────────► │  FAISS     │
│  Vite + TS   │ ◄───────── │  /ask      │                    │  IndexFlatIP│
│  Tailwind    │            │  /rebuild  │                    │  + cosine  │
└──────────────┘            │  /health   │                    └────────────┘
                            └────┬───────┘                          ▲
                                 │ chat                       index │
                                 ▼                                  │
                            ┌────────────┐                          │
                            │  Mistral   │                          │
                            │  small +   │                          │
                            │  embed     │                          │
                            └────────────┘                          │
                                                                    │
              ┌────────────────────┐    fetch     ┌─────────────────┴──────┐
              │ Open Agenda public │ ───────────► │ indexer.py             │
              │ (Opendatasoft v2.1)│              │ chunks → embeddings    │
              └────────────────────┘              └────────────────────────┘
```

### Choix technologiques

| Composant | Choix | Justification |
|---|---|---|
| **Source de données** | Dataset public Opendatasoft `evenements-publics-openagenda` | Pas de clé API, données fraîches d'Open Agenda, ODSQL puissant pour filtrer |
| **Filtre temporel** | `firstdate_begin >= now() - 365j` | Conformité avec l'énoncé (1 an passé + à venir) |
| **Embeddings** | `mistral-embed` (1024 dim) | Imposé par le brief, qualité multilingue FR |
| **Index vectoriel** | FAISS `IndexFlatIP` + L2-norm | Cosine exacte, suffisant pour ~600 chunks ; `IVF`/`HNSW` over-engineering ici |
| **Chunking** | RecursiveCharacterTextSplitter (1500 c / 150 c overlap) | Compromis taille texte / précision sémantique |
| **LLM** | `mistral-small-latest` (T=0.2) | Bon ratio qualité/coût pour le POC |
| **Backend** | FastAPI + Uvicorn | Swagger auto, types Pydantic, async natif |
| **Frontend** | Vite + React + TS + Tailwind | DX rapide, build minimal, esthétique |
| **Évaluation** | Ragas (faithfulness, answer_relevancy, context_precision/recall) | Standard de l'éval RAG, juge LLM Mistral |
| **Conteneurisation** | Docker Compose 2 services | Déployable en local et sur VPS |

---

## 2. Structure du projet

```
RAG_Pulse_Events/
├── app/                              # API FastAPI
│   ├── main.py                       # /ask, /rebuild, /health, CORS, lifespan
│   ├── schemas.py                    # Pydantic models
│   └── rag_chain.py                  # Logique RAG (retrieve + prompt + generate)
├── utils/
│   ├── config.py                     # Variables centralisées (.env)
│   ├── openagenda_loader.py          # Fetch/parse Open Agenda
│   └── vector_store.py               # FAISS + embeddings Mistral
├── frontend/                         # Vite + React + TS + Tailwind
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   └── components/
│   │       ├── ChatBox.tsx
│   │       ├── MessageBubble.tsx
│   │       └── EventCard.tsx
│   ├── Dockerfile                    # Multi-stage build → nginx
│   └── nginx.conf                    # Proxy /api → api:8000
├── data/raw/                         # Snapshots JSON Open Agenda (committés)
├── vector_db/                        # Index FAISS + chunks (généré, gitignored)
├── tests/
│   ├── test_openagenda_loader.py     # 9 tests
│   ├── test_vector_store.py          # 6 tests (mocks Mistral)
│   ├── test_api.py                   # 6 tests FastAPI
│   ├── fixtures/events_sample.json   # 10 events Brest
│   └── qa_dataset.json               # 15 Q/R annotées
├── indexer.py                        # CLI: fetch → chunks → embed → FAISS
├── evaluate_rag.py                   # Pipeline Ragas
├── Dockerfile.api                    # Image FastAPI
├── docker-compose.yml                # api + frontend
├── requirements.txt
├── .env.example
└── pytest.ini
```

---

## 3. Démarrage rapide

### Prérequis
- Python ≥ 3.10
- Node.js ≥ 18 (pour le frontend en dev)
- Docker + Docker Compose (optionnel)
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### A. En local (dev)

```bash
# 1. Cloner et configurer
git clone <repo-url> && cd RAG_Pulse_Events
cp .env.example .env             # éditer avec ta clé MISTRAL_API_KEY

# 2. Environnement Python
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# 3. Construire l'index (fetch live + embeddings)
python indexer.py
# → ~30s, ~590 chunks, sauvegardé dans vector_db/

# 4. Tests
pytest -m "not live"             # 20 tests offline
# pytest -m live                 # 1 test réseau (Open Agenda)

# 5. Lancer l'API (terminal A)
uvicorn app.main:app --reload
# → http://localhost:8000/docs

# 6. Lancer le frontend (terminal B)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### B. Avec Docker (démo / VPS)

```bash
# Construire l'index AVANT le up (volume monté)
python indexer.py

# Lancer la stack
docker compose up --build
# → http://localhost     (frontend)
# → http://localhost:8000/docs  (Swagger)
```

### C. Évaluation Ragas

```bash
python evaluate_rag.py
# → tableau récapitulatif + evaluation_results.json
```

---

## 4. API

Documentation interactive : `http://localhost:8000/docs`

### `GET /health`
```json
{ "status": "ok", "index_size": 590, "model": "mistral-small-latest" }
```

### `POST /ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quels concerts à Brest ce week-end ?"}'
```
```json
{
  "answer": "Voici quelques idées de concerts à Brest cette semaine : ...",
  "sources": [
    {
      "title": "La Nuit des conservatoires à Brest",
      "url": "https://openagenda.com/.../la-nuit-des-conservatoires-a-brest",
      "daterange": "Vendredi 30 janvier, 17h30",
      "location_name": "Conservatoire à rayonnement régional de Brest",
      "image": "https://cdn.openagenda.com/main/...",
      "score": 80.32
    }
  ]
}
```

### `POST /rebuild`
Reconstruit l'index FAISS depuis l'API Open Agenda (live) ou le dernier snapshot local.
```bash
curl -X POST http://localhost:8000/rebuild \
  -H 'Content-Type: application/json' \
  -d '{"use_snapshot": false}'
```

---

## 5. Pipeline de données

```
1. fetch_brest_events()
   → GET /catalog/datasets/evenements-publics-openagenda/records
     ?where=location_city="Brest" AND firstdate_begin >= date'2025-04-27'
     &limit=100&offset=…
   → ~506 events
   → snapshot data/raw/events_brest_<YYYY-MM-DD>.json

2. event_to_document()
   → page_content = "Titre: …\nDescription: …\nDates: …\nLieu: …\nMots-clés: …"
   → metadata = {uid, title, url, daterange, date_begin, city, …}

3. RecursiveCharacterTextSplitter
   → 590 chunks (1500c, overlap 150c)

4. mistralai.embeddings.create(inputs=[…])
   → 590 vecteurs × 1024 dim (par lots de 32)

5. faiss.normalize_L2 + IndexFlatIP.add
   → vector_db/faiss_index.idx + document_chunks.pkl
```

---

## 6. Évaluation

Jeu d'évaluation : **15 questions annotées** dans [tests/qa_dataset.json](tests/qa_dataset.json), couvrant :
- **Lieux spécifiques** (galerie Comoedia, musée de la Marine, médiathèque Capucins, château)
- **Thématiques** (concerts, expositions, festivals, visites guidées, événements maritimes)
- **Recommandations libres** ("que faire pour un amateur d'art ?")
- **Hors-domaine** (test du refus poli)

Métriques Ragas calculées (juge = `mistral-small-latest`) :

| Métrique | Description |
|---|---|
| **faithfulness** | La réponse reste fidèle au contexte récupéré (pas d'hallucination) |
| **answer_relevancy** | Pertinence de la réponse vis-à-vis de la question |
| **context_precision** | Pertinence du contexte récupéré pour la question |
| **context_recall** | Couverture du ground_truth par le contexte |

Lancer l'éval : `python evaluate_rag.py` → tableau + `evaluation_results.json`.

---

## 7. Tests unitaires

```bash
pytest -m "not live"
```

| Module | Tests offline | Tests live | Couverture |
|---|---|---|---|
| `test_openagenda_loader.py` | 8 | 1 | parsing, filtre, conversion event→doc, snapshot roundtrip |
| `test_vector_store.py` | 6 | 0 | build, persistence, search, gestion vide (mock Mistral) |
| `test_api.py` | 6 | 0 | /health, /ask, /rebuild, validation, erreurs 503 |

**20 tests offline** (pas d'appel réseau, pas de coût Mistral) + 1 test live (`-m live`, smoke fetch Open Agenda).

---

## 8. Variables d'environnement

Voir [.env.example](.env.example).

| Variable | Défaut | Rôle |
|---|---|---|
| `MISTRAL_API_KEY` | *(requis)* | Clé API Mistral |
| `MISTRAL_MODEL` | `mistral-small-latest` | Modèle de génération |
| `TARGET_CITY` | `Brest` | Ville cible (filtre `location_city`) |
| `SINCE_DAYS` | `365` | Profondeur historique en jours |

---

## 9. Déploiement VPS (phase 2)

```bash
# Sur le VPS
git clone <repo>
cd RAG_Pulse_Events
cp .env.example .env && nano .env       # ajouter la clé
python indexer.py                        # ou docker run … indexer.py
docker compose up -d --build
```

Pour **HTTPS automatique** : ajouter un service Caddy en front du compose :
```yaml
  caddy:
    image: caddy:alpine
    ports: ["80:80","443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
```
avec `Caddyfile` :
```
mondomaine.fr {
    reverse_proxy frontend:80
}
```

---

## 10. Pistes d'amélioration

- **Filtrage métier** : exclure les events "France Travail" non culturels — ils représentent ~65% du dataset Brest (lieux *Agence Brest Europe/Marine/Iroise*, mots-clés `recrutement`, `1 jeune 1 solution`) — via un filtre sur `keywords_fr` ou `originagenda_uid`
- **Reranking** : ajouter un cross-encoder pour réordonner les top-k FAISS
- **Hybrid search** : combiner BM25 + dense pour mieux gérer les noms propres (lieux, artistes)
- **Streaming** : utiliser `client.chat.stream()` pour afficher la réponse en streaming dans le frontend
- **Cache de réponses** : Redis/in-memory sur les questions fréquentes
- **Refresh automatique** : cron quotidien `/rebuild` ou diff incrémental sur `updatedat`
- **Multi-villes** : paramètre `city` dans la requête + index multi-tenant
- **Mémoire de conversation** : si exigé par les utilisateurs (hors scope POC)
- **Auth `/rebuild`** : header admin token pour la prod

---

## 11. Crédits

- **Données** : [Open Agenda](https://openagenda.com) via [public.opendatasoft.com](https://public.opendatasoft.com)
- **LLM** : [Mistral AI](https://mistral.ai)
- **Stack** : FastAPI, LangChain, FAISS, Ragas, Vite, React, Tailwind
- **Inspiration** : squelette `SimpleRAGMistral` du cours OpenClassrooms *Mettez en place un RAG pour un LLM*

---

## Licence

POC éducatif — usage non commercial.
