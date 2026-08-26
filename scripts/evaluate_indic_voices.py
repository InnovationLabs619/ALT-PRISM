"""
PRISM IndicVoices Evaluation Benchmark
=======================================
Runs real IndicVoices speech dataset samples against PRISM's self-hosted
Inference Engine and reports WER, CER, translation quality, and latency.

Usage:
    python scripts/evaluate_indic_voices.py --lang telugu
    python scripts/evaluate_indic_voices.py --manifest data/manifests/indic_voices_telugu.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.evaluation.evaluate_asr import calculate_wer, calculate_cer
from ml.inference.engine import InferenceEngine


def evaluate_manifest(manifest_path: Path):
    if not manifest_path.exists():
        print(f"[Error] Manifest file not found: {manifest_path}")
        print("Please run `scripts/fetch_indic_voices.py --lang <language>` first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("[Error] Manifest is empty.")
        return

    print(f"\n=======================================================")
    print(f"  PRISM IndicVoices Benchmark: {manifest_path.name}")
    print(f"  Total Samples: {len(records)}")
    print(f"=======================================================\n")

    engine = InferenceEngine()
    engine.initialize_models()

    total_wer = 0.0
    total_cer = 0.0
    total_latency = 0.0
    valid_count = 0

    results = []

    for idx, rec in enumerate(records):
        audio_file = BASE_DIR / rec["filepath"]
        ref_transcript = rec.get("transcript", "")
        lang = rec.get("language", "te")

        if not audio_file.exists():
            print(f"[{idx+1}/{len(records)}] Audio file missing: {audio_file}")
            continue

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        t0 = time.time()
        res = engine.process_speech(
            audio_input=audio_bytes,
            session_id=f"IV-EVAL-{idx:04d}",
            language_hint=lang,
        )
        latency = round(time.time() - t0, 3)

        wer = calculate_wer(ref_transcript, res.transcript) if ref_transcript else 0.0
        cer = calculate_cer(ref_transcript, res.transcript) if ref_transcript else 0.0

        total_wer += wer
        total_cer += cer
        total_latency += latency
        valid_count += 1

        print(f"[{idx+1}/{len(records)}] ID: {rec['id']} | Lang: {lang.upper()} | Latency: {latency}s")
        print(f"   Ref:  {ref_transcript[:60]}")
        print(f"   Hyp:  {res.transcript[:60]}")
        print(f"   Eng:  {res.english_translation[:60]}")
        print(f"   WER:  {wer*100:.1f}% | CER: {cer*100:.1f}%")
        print("-" * 55)

        results.append({
            "id": rec["id"],
            "language": lang,
            "ref_transcript": ref_transcript,
            "hyp_transcript": res.transcript,
            "english_translation": res.english_translation,
            "wer": wer,
            "cer": cer,
            "latency_sec": latency,
        })

    if valid_count > 0:
        avg_wer = (total_wer / valid_count) * 100
        avg_cer = (total_cer / valid_count) * 100
        avg_lat = total_latency / valid_count

        print("\n=======================================================")
        print(f"  INDICVOICES BENCHMARK SUMMARY")
        print(f"  - Average WER: {avg_wer:.2f}%")
        print(f"  - Average CER: {avg_cer:.2f}%")
        print(f"  - Average Latency: {avg_lat:.3f}s")
        print(f"=======================================================\n")

        # Save report
        report_path = BASE_DIR / "ml" / "reports" / "indic_voices_evaluation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "manifest": str(manifest_path),
                "samples_count": valid_count,
                "avg_wer": round(avg_wer, 2),
                "avg_cer": round(avg_cer, 2),
                "avg_latency": round(avg_lat, 3),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PRISM against IndicVoices speech dataset.")
    parser.add_argument("--lang", type=str, default=None, help="Language name (e.g. telugu, hindi, assamese)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to JSONL manifest")

    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.lang:
        manifest_path = BASE_DIR / "data" / "manifests" / f"indic_voices_{args.lang.lower()}.jsonl"
    else:
        manifest_path = BASE_DIR / "data" / "manifests" / "indic_voices_telugu.jsonl"

    evaluate_manifest(manifest_path)


if __name__ == "__main__":
    main()
