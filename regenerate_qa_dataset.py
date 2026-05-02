"""
Régénère un draft de qa_dataset.json à partir des réponses réelles du bot.

Pourquoi ce script :
    Les ground_truths écrits "à la main" mentionnent souvent des entités absentes
    de l'index (lieux célèbres, événements connus). Cela pénalise artificiellement
    les métriques Ragas `context_precision` et `context_recall`.

    Au lieu de cela, on prend les réponses réelles du bot comme base de
    ground_truths, puis on les valide humainement. C'est la méthodologie
    "model-generated GT with human validation" — standard en éval RAG.

Workflow:
    1. Lance `python evaluate_rag.py` (produit evaluation_results.json)
    2. Lance `python regenerate_qa_dataset.py` (produit tests/qa_dataset.draft.json)
    3. Édite manuellement le draft : pour chaque question, vérifie que le ground_truth
       décrit la réponse IDÉALE. Corrige si le bot s'est trompé.
    4. Supprime les champs "_review_*" une fois validé.
    5. Renomme : `tests/qa_dataset.draft.json` → `tests/qa_dataset.json`.
    6. Re-lance `python evaluate_rag.py` pour la v2 des métriques.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_FILE = PROJECT_ROOT / "evaluation_results.json"
ORIGINAL_QA_FILE = PROJECT_ROOT / "tests" / "qa_dataset.json"
DEFAULT_DRAFT_FILE = PROJECT_ROOT / "tests" / "qa_dataset.draft.json"


def strip_markdown_to_paragraph(answer: str) -> str:
    """
    Convertit la réponse markdown du bot en un paragraphe descriptif neutre,
    plus adapté comme ground_truth Ragas (qui décompose en faits atomiques).

    Stratégie:
    - Conserve les noms propres (titres, lieux, dates).
    - Retire les emojis de structure (📅 📍 🔗) et le markdown bold.
    - Conserve les liens URL (utiles pour le LLM juge).
    - Compacte les sauts de ligne multiples.
    """
    text = answer
    # Retire le markdown bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Retire les emojis de structure
    text = re.sub(r"[📅📍🔗]", "", text)
    # Retire les puces "1. " "2. " en début de ligne
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Compacte les espaces et sauts de ligne
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Génère un qa_dataset.draft.json depuis les réponses du bot")
    parser.add_argument(
        "--predictions",
        default=str(PREDICTIONS_FILE),
        help="Fichier JSON produit par evaluate_rag.py",
    )
    parser.add_argument(
        "--original",
        default=str(ORIGINAL_QA_FILE),
        help="Fichier qa_dataset.json original (pour récupérer les categories)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DRAFT_FILE),
        help="Fichier draft de sortie",
    )
    parser.add_argument(
        "--keep-markdown",
        action="store_true",
        help="Conserve le markdown brut au lieu de le simplifier",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"❌ {pred_path} introuvable. Lance d'abord `python evaluate_rag.py`.")
        return 1

    with open(pred_path, encoding="utf-8") as f:
        results = json.load(f)

    with open(args.original, encoding="utf-8") as f:
        original_qa = json.load(f)

    # Indexer les questions originales par id pour conserver les categories
    original_by_id = {item["id"]: item for item in original_qa["dataset"]}

    new_dataset = []
    for pred in results["predictions"]:
        qid = pred.get("id", "")
        original = original_by_id.get(qid, {})
        bot_answer = pred["answer"]
        gt = bot_answer if args.keep_markdown else strip_markdown_to_paragraph(bot_answer)

        new_dataset.append(
            {
                "id": qid,
                "question": pred["question"],
                "ground_truth": gt,
                "category": original.get("category", "uncategorized"),
                # Champs d'aide à la review humaine — à supprimer après validation
                "_review_status": "DRAFT",
                "_review_instructions": "Vérifier que ground_truth décrit la réponse IDÉALE. Corriger si le bot s'est trompé. Puis supprimer ce champ et `_original_ground_truth`.",
                "_original_ground_truth": original.get("ground_truth", ""),
            }
        )

    output = {
        "description": (
            "DRAFT de qa_dataset généré depuis les réponses du bot. "
            "À RÉVISER MANUELLEMENT avant remplacement de qa_dataset.json. "
            "Voir _review_instructions sur chaque entrée."
        ),
        "dataset": new_dataset,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Draft écrit : {out_path}")
    print(f"   {len(new_dataset)} questions traitées.\n")
    print("📝 Prochaines étapes :")
    print(f"   1. Ouvre {out_path.relative_to(PROJECT_ROOT)} dans VS Code")
    print("   2. Pour chaque entrée, vérifie le `ground_truth` :")
    print("      - Décrit-il la réponse IDÉALE à la question ?")
    print("      - Le bot s'est-il trompé ? Si oui, corrige.")
    print("      - Trop verbeux ? Raccourcis (1-3 phrases suffisent).")
    print("   3. Supprime les champs `_review_status`, `_review_instructions`, `_original_ground_truth`.")
    print("   4. Renomme :")
    print("      Move-Item tests/qa_dataset.json tests/qa_dataset.v1_brest_style.json")
    print(f"      Move-Item tests/qa_dataset.draft.json tests/qa_dataset.json")
    print("   5. Re-lance : python evaluate_rag.py")
    print("\n💡 Astuce : la question hors-domaine (q15 météo) a probablement un ground_truth")
    print("   bizarre car le bot a refusé. Tu peux la garder telle quelle (la formule de refus)")
    print("   ou la remplacer par : 'Cette question est hors-domaine. Refus poli attendu.'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
