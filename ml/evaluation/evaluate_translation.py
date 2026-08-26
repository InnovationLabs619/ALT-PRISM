"""
PRISM Translation Offline Evaluation Engine
============================================
Evaluates self-hosted translation performance using BLEU (sacrebleu) and chrF.
Stratifies scores across languages and domain categories.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import sacrebleu

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


def evaluate_translation_predictions(
    references: List[str],
    hypotheses: List[str],
    source_languages: List[str],
    target_languages: List[str],
    output_dir: str = "ml/reports/translation",
) -> Dict:
    """Calculates corpus-level and language-level BLEU and chrF scores."""
    os.makedirs(output_dir, exist_ok=True)

    # SacreBLEU requires references as list of lists
    ref_list = [[r] for r in references]

    # Global BLEU and chrF
    bleu = sacrebleu.corpus_bleu(hypotheses, [[r for r in references]])
    chrf = sacrebleu.corpus_chrf(hypotheses, [[r for r in references]])

    # Language pair metrics
    pair_data: Dict[str, Dict[str, List[str]]] = {}
    for ref, hyp, src, tgt in zip(references, hypotheses, source_languages, target_languages):
        pair = f"{src}-{tgt}"
        if pair not in pair_data:
            pair_data[pair] = {"refs": [], "hyps": []}
        pair_data[pair]["refs"].append(ref)
        pair_data[pair]["hyps"].append(hyp)

    pair_metrics = {}
    for pair, items in pair_data.items():
        p_bleu = sacrebleu.corpus_bleu(items["hyps"], [[r for r in items["refs"]]])
        p_chrf = sacrebleu.corpus_chrf(items["hyps"], [[r for r in items["refs"]]])
        pair_metrics[pair] = {
            "sample_count": len(items["refs"]),
            "bleu": round(float(p_bleu.score), 2),
            "chrf": round(float(p_chrf.score), 2),
        }

    # Write CSV
    csv_path = os.path.join(output_dir, "translation_language_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Language Pair", "Sample Count", "BLEU", "chrF"])
        for pair, m in pair_metrics.items():
            writer.writerow([pair, m["sample_count"], m["bleu"], m["chrf"]])

    # Write JSON
    report = {
        "global_metrics": {
            "total_samples": len(references),
            "global_bleu": round(float(bleu.score), 2),
            "global_chrf": round(float(chrf.score), 2),
        },
        "pair_metrics": pair_metrics,
    }

    json_path = os.path.join(output_dir, "translation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[PRISM Eval] Saved Translation metrics to {json_path} and {csv_path}")
    return report


if __name__ == "__main__":
    sample_refs = [
        "My phone was stolen near the railway station yesterday evening.",
        "My black color motorcycle was stolen from near the bus stand.",
        "A stranger snatched my gold chain near the market and ran away.",
        "My younger brother left Vijayawada at 9:30 PM and has not returned yet.",
        "50,000 rupees were transferred from my bank account through an online fraud.",
    ]
    sample_hyps = [
        "My phone was stolen near the railway station yesterday evening.",
        "My black color motorcycle was stolen near the bus stand.",
        "A stranger snatched my gold chain near the market and ran away.",
        "My younger brother left Vijayawada at 9:30 PM and has not returned yet.",
        "50000 rupees were transferred from my bank account by an online fraud.",
    ]
    sample_src = ["te", "hi", "ta", "kn", "mr"]
    sample_tgt = ["en", "en", "en", "en", "en"]

    report = evaluate_translation_predictions(sample_refs, sample_hyps, sample_src, sample_tgt)
    print("Translation Evaluation Summary:")
    print(json.dumps(report, indent=2))
