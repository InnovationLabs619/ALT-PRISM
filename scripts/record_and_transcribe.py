"""
PRISM Terminal Live Microphone Recording & Real-Time Speech Extraction
========================================================================
Records live voice audio directly from your system microphone, processes it
through PRISM's self-hosted ASR and NMT models, and extracts text & English translation.

Usage:
    python scripts/record_and_transcribe.py [--duration 5] [--lang te]
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sounddevice as sd

from ml.inference.engine import InferenceEngine
from ml.inference.language_detector import language_detector


def record_audio(duration_sec: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Records audio from microphone for specified duration in seconds."""
    print(f"\n[MIC] 🎙️  Recording live microphone audio for {duration_sec:.1f} seconds...")
    print("[MIC]     Speak into your microphone now!\n")

    num_samples = int(duration_sec * sample_rate)
    recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")

    # Animated progress bar during recording
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_sec:
            break
        progress = min(1.0, elapsed / duration_sec)
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r[RECORDING] [{bar}] {elapsed:.1f}s / {duration_sec:.1f}s")
        sys.stdout.flush()
        time.sleep(0.1)

    sd.wait()  # Wait for recording to complete
    sys.stdout.write(f"\r[RECORDING] [------------------------------] {duration_sec:.1f}s / {duration_sec:.1f}s COMPLETE!\n")
    sys.stdout.flush()

    # Flatten audio to 1D float32 array
    audio_flat = recording.flatten()
    return audio_flat


def main():
    parser = argparse.ArgumentParser(description="Record live microphone audio and extract transcript & English translation.")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--lang", type=str, default=None, help="Optional language hint (e.g. te, hi, ta, kn, mr, bn, ml, en)")
    args = parser.parse_args()

    print("=" * 80)
    print("      PRISM TERMINAL LIVE VOICE RECORDING & TRANSLATION PIPELINE")
    print("=" * 80)

    # Initialize PRISM Self-Hosted AI Engine
    engine = InferenceEngine()
    engine.initialize_models()

    # Step 1: Record Voice Audio from Terminal Microphone
    audio_array = record_audio(duration_sec=args.duration, sample_rate=16000)

    # Step 2: Process Recorded Audio through PRISM Speech Pipeline
    print("\n[AI ENGINE] Processing recorded speech (ASR + Code-Switching + Translation)...")
    t0 = time.time()
    res = engine.process_speech(
        audio_input=audio_array,
        session_id="MIC-LIVE-01",
        language_hint=args.lang,
        target_language="en",
    )
    total_latency = time.time() - t0
    rtf = round(total_latency / max(0.1, len(audio_array) / 16000.0), 3)

    # Language Identification
    detected_lang_info = language_detector.detect_from_text(res.transcript, language_hint=res.language)

    # Step 3: Print Formatted Output Results
    print("\n" + "=" * 80)
    print("                      PRISM INFERENCE EXTRACTION RESULTS")
    print("=" * 80)
    print(f"  AUDIO DURATION       : {len(audio_array) / 16000.0:.2f} seconds")
    print(f"  DETECTED LANGUAGE    : {detected_lang_info.language_name} ({detected_lang_info.language_code.upper()})")
    print(f"  LANGUAGE CONFIDENCE  : {detected_lang_info.confidence * 100:.1f}%")
    print(f"  CODE-SWITCHED        : {'YES' if res.is_code_switched else 'NO'}")
    print(f"  LANGUAGES INVOLVED   : {', '.join(res.languages).upper()}")
    print("-" * 80)
    print(f"  ORIGINAL TRANSCRIPT  :\n  -> \"{res.transcript if res.transcript else 'No speech detected.'}\"")
    print("-" * 80)
    print(f"  DEFAULT TRANSLATION (ENGLISH) :\n  -> \"{res.english_translation if res.english_translation else 'No translation generated.'}\"")
    print("-" * 80)
    print(f"  TOTAL LATENCY        : {total_latency:.3f} seconds")
    print(f"  REAL-TIME FACTOR     : {rtf:.3f} {'(Faster than Live Speech)' if rtf < 1.0 else '(Near Real-Time)'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
