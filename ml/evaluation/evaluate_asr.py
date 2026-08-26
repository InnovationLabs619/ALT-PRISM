"""
PRISM ASR Offline Evaluation Engine
===================================
Calculates Word Error Rate (WER) and Character Error Rate (CER) per language,
speaker, and scenario using jiwer. Generates structured evaluation reports.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import jiwer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


def normalize_text_for_eval(text: str) -> str:
    """Standardizes punctuation and whitespace for ASR evaluation."""
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return transform(text)


def evaluate_asr_predictions(
    references: List[str],
    hypotheses: List[str],
    languages: List[str],
    categories: List[str],
    output_dir: str = "ml/reports/asr",
) -> Dict:
    """Calculates WER, CER, and language-stratified error rates."""
    os.makedirs(output_dir, exist_ok=True)

    norm_refs = [normalize_text_for_eval(r) for r in references]
    norm_hyps = [normalize_text_for_eval(h) for h in hypotheses]

    # Global WER and CER
    global_wer = jiwer.wer(norm_refs, norm_hyps)
    global_cer = jiwer.cer(norm_refs, norm_hyps)

    # Per-language metrics
    lang_data: Dict[str, Dict[str, List[str]]] = {}
    for ref, hyp, lang in zip(norm_refs, norm_hyps, languages):
        if lang not in lang_data:
            lang_data[lang] = {"refs": [], "hyps": []}
        lang_data[lang]["refs"].append(ref)
        lang_data[lang]["hyps"].append(hyp)

    lang_metrics = {}
    for lang, items in lang_data.items():
        l_wer = jiwer.wer(items["refs"], items["hyps"])
        l_cer = jiwer.cer(items["refs"], items["hyps"])
        lang_metrics[lang] = {
            "sample_count": len(items["refs"]),
            "wer": round(float(l_wer), 4),
            "cer": round(float(l_cer), 4),
            "accuracy_word": round(max(0.0, 1.0 - float(l_wer)), 4),
        }

    # Save CSV report
    csv_path = os.path.join(output_dir, "asr_language_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Language", "Sample Count", "WER", "CER", "Word Accuracy"])
        for lang, m in lang_metrics.items():
            writer.writerow([lang, m["sample_count"], m["wer"], m["cer"], m["accuracy_word"]])

    # Save JSON report
    report = {
        "global_metrics": {
            "total_samples": len(references),
            "global_wer": round(float(global_wer), 4),
            "global_cer": round(float(global_cer), 4),
            "global_word_accuracy": round(max(0.0, 1.0 - float(global_wer)), 4),
        },
        "language_metrics": lang_metrics,
    }

    json_path = os.path.join(output_dir, "asr_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[PRISM Eval] Saved ASR metrics to {json_path} and {csv_path}")
    return report


if __name__ == "__main__":
    # Test evaluation with representative multi-lingual domain baseline samples
    sample_refs = [
        "na phone ninna evening railway station daggara theft ayyindi",
        "mera black color ka motorcycle bus stand ke paas se chori ho gaya",
        "enoda gold chain market kitta oru stranger snatch pannittu odipoyittan",
        "nanna thamma vijayawada indha 930 pm ge horatu innu baralilla",
        "majhya bank account madhun online fraud dwara 50000 rupees transfer jhale",
        "an unknown person called me pretending to be a bank manager and took my otp",
    ]
    sample_hyps = [
        "na phone ninna evening railway station daggara theft ayyindi",
        "mera black color motorcycle bus stand ke paas se chori ho gaya",
        "enoda gold chain market kitta oru stranger snatch panitu odipoyittan",
        "nanna thamma vijayawada indha 9:30 pm ge horatu innu baralilla",
        "majhya bank account madhun online fraud dwara 50000 rupees transfer jhale",
        "an unknown person called me pretending to be a bank manager and took my otp",
    ]
    sample_langs = ["te", "hi", "ta", "kn", "mr", "en"]
    sample_cats = ["theft", "vehicle_theft", "robbery", "missing_person", "cyber_fraud", "fraud"]

    report = evaluate_asr_predictions(sample_refs, sample_hyps, sample_langs, sample_cats)
    print("ASR Evaluation Summary:")
    print(json.dumps(report, indent=2))
