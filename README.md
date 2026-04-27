# Puls-Events — Assistant RAG pour événements culturels

POC d'un chatbot de recommandation d'événements culturels à **Brest**, appuyé sur un système RAG (FAISS + Mistral via LangChain) et exposé par une API REST FastAPI consommée par un frontend React.

## Architecture

```
┌──────────────┐    HTTP    ┌────────────┐    embed/search    ┌────────────┐
│  React (UI)  │ ─────────► │  FastAPI   │ ─────────────────► │  FAISS     │
│  Vite + TS   │ ◄───────── │  /ask      │                    │  (cosine)  │
└──────────────┘            │  /rebuild  │                    └────────────┘
                            │  /health   │                          ▲
                            └────┬───────┘                          │
                                 │ chat                       index │
                                 ▼                                  │
                            ┌────────────┐                          │
                            │  Mistral   │                          │
                            │  API       │                          │
                            └────────────┘                          │
                                                                    │
              ┌────────────────────┐    fetch     ┌─────────────────┴──────┐
              │ Open Agenda public │ ───────────► │ indexer.py             │
              │ (Opendatasoft v2.1)│              │ (chunks → embeddings)  │
              └────────────────────┘              └────────────────────────┘
```

## Démarrage rapide

```bash
# 1. Environnement
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.example .env  # ajouter ta clé MISTRAL_API_KEY

# 2. Construction de l'index
python indexer.py

# 3. API
uvicorn app.main:app --reload
# → http://localhost:8000/docs

# 4. Frontend (autre terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

Plus de détails à venir au fil de l'implémentation.
