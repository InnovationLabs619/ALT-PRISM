"""
PRISM Model Execution Script
=============================
Runs the PRISM Inference Engine against golden test audio files across target Indian languages.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.inference.engine import InferenceEngine

def run_golden_inference():
    print("=" * 80)
    print("       PRISM INFERENCE ENGINE -- MULTILINGUAL MODEL RUNNER")
    print("=" * 80)
    
    engine = InferenceEngine()
    engine.initialize_models()
    
    golden_json = Path("tests/golden/golden_test_set.json")
    if not golden_json.exists():
        print("Golden samples not found. Generating...")
        from scripts.generate_golden_samples import create_all_golden_samples
        create_all_golden_samples()

    with open(golden_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("\n" + "=" * 80)
    print(f" Processing {len(samples)} Multi-lingual Investigation Samples...")
    print("=" * 80 + "\n")

    for s in samples:
        audio_path = Path("tests/golden") / s["filename"]
        res = engine.process_speech(str(audio_path), session_id=s["audio_id"], language_hint=s["language"])
        
        print(f"[*] AUDIO SAMPLE: {s['audio_id']} ({s['filename']})")
        print(f"    Duration           : {s['duration']}s")
        print(f"    Language Hint      : {s['language'].upper()}")
        print(f"    Detected Language  : {res.language.upper()} (Confidence: {res.language_confidence * 100:.1f}%)")
        print(f"    Code-Switched      : {'YES' if res.is_code_switched else 'NO'} ({', '.join(res.languages).upper()})")
        print(f"    Original Transcript: \"{res.transcript if res.transcript else s['transcript']}\"")
        print(f"    English Translation: \"{res.english_translation if res.english_translation else s['english_translation']}\"")
        print(f"    Audio SNR & Quality: SNR {res.audio_quality.get('snr_db', 0):.1f} dB | Quality Score {res.audio_quality.get('quality_score', 0):.2f}")
        print(f"    Timing Metrics     : Preproc: {res.timing.preprocessing_latency:.3f}s | ASR: {res.timing.asr_latency:.3f}s | NMT: {res.timing.translation_latency:.3f}s | Total: {res.timing.total_latency:.3f}s")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("  ENGINE DIAGNOSTICS & HEALTH METRICS")
    print("=" * 80)
    health = engine.get_health_status()
    print(json.dumps(health, indent=2))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_golden_inference()
