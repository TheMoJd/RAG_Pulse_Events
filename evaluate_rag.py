"""
Évaluation automatique du système RAG via Ragas.

Métriques calculées:
- faithfulness        : la réponse reste fidèle au contexte récupéré (pas d'hallucination)
- answer_relevancy    : pertinence de la réponse vis-à-vis de la question
- context_precision   : pertinence du contexte récupéré pour la question
- context_recall      : couverture du ground_truth par le contexte

Usage:
    python evaluate_rag.py [--output evaluation_results.json]

Sortie:
    - tableau récapitulatif imprimé
    - JSON détaillé écrit sur disque

Note: Ragas appelle un LLM "juge" pour scorer (utilise OpenAI par défaut).
On le configure ici sur Mistral via langchain-mistralai pour rester homogène.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QA_DATASET_PATH = Path(__file__).parent / "tests" / "qa_dataset.json"
DEFAULT_OUTPUT = Path(__file__).parent / "evaluation_results.json"


def collect_predictions(qa_items: list[dict]) -> list[dict]:
    """Pour chaque question, exécute la chaîne RAG et collecte answer + retrieved_contexts."""
    from app.rag_chain import RAGChain
    from utils.vector_store import VectorStoreManager

    vs = VectorStoreManager()
    if vs.size == 0:
        logger.error("Index FAISS vide. Lance `python indexer.py` d'abord.")
        sys.exit(1)

    rag = RAGChain(vector_store=vs)

    predictions = []
    for i, item in enumerate(qa_items, 1):
        q = item["question"]
        logger.info(f"[{i}/{len(qa_items)}] {q}")
        # Récupère le contexte ET la réponse
        contexts_raw = vs.search(q, k=5)
        contexts = [c["text"] for c in contexts_raw]
        result = rag.ask(q)
        predictions.append(
            {
                "id": item.get("id"),
                "question": q,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "category": item.get("category"),
            }
        )
    return predictions


def run_ragas(predictions: list[dict]) -> dict:
    """Lance l'évaluation Ragas sur les prédictions."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.error(f"Ragas/datasets non installé: {e}")
        logger.error("→ pip install ragas datasets")
        sys.exit(1)

    ds = Dataset.from_list(
        [
            {
                "question": p["question"],
                "answer": p["answer"],
                "contexts": p["contexts"],
                "ground_truth": p["ground_truth"],
            }
            for p in predictions
        ]
    )

    # Configuration LLM/embeddings via Mistral pour rester cohérent
    try:
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from utils.config import MISTRAL_API_KEY, MODEL_NAME

        judge_llm = LangchainLLMWrapper(
            ChatMistralAI(model=MODEL_NAME, mistral_api_key=MISTRAL_API_KEY, temperature=0)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_emb,
        )
    except ImportError:
        logger.warning(
            "langchain-mistralai non installé, fallback sur les LLM par défaut de Ragas (OpenAI)."
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

    # Convertit le résultat Ragas en dict simple
    scores = {}
    try:
        df = result.to_pandas()
        for col in df.columns:
            if col not in ("question", "answer", "contexts", "ground_truth"):
                values = df[col].dropna()
                if len(values) > 0 and values.dtype.kind in "fi":
                    scores[col] = float(values.mean())
    except Exception as e:
        logger.warning(f"Conversion résultats Ragas: {e}")
        scores = dict(result)

    return scores


def print_summary(scores: dict, predictions: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS D'ÉVALUATION RAGAS")
    print("=" * 60)
    print(f"  Échantillon: {len(predictions)} questions\n")
    for metric, value in scores.items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {metric:25s} {bar} {value:.3f}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Fichier de sortie JSON")
    parser.add_argument(
        "--qa-dataset",
        default=str(QA_DATASET_PATH),
        help="Fichier JSON avec les questions/ground_truth annotés",
    )
    args = parser.parse_args()

    # Chargement du dataset
    with open(args.qa_dataset, encoding="utf-8") as f:
        qa = json.load(f)
    qa_items = qa["dataset"] if isinstance(qa, dict) and "dataset" in qa else qa
    logger.info(f"Chargé {len(qa_items)} questions depuis {args.qa_dataset}")

    # Prédictions RAG
    predictions = collect_predictions(qa_items)

    # Évaluation Ragas
    scores = run_ragas(predictions)

    # Sauvegarde
    output = {
        "n_questions": len(predictions),
        "metrics": scores,
        "predictions": predictions,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Résultats écrits dans {args.output}")

    print_summary(scores, predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
