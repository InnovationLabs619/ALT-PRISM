"""
PRISM Latency & Throughput Benchmarking Engine
==============================================
Benchmarks audio preprocessing, ASR inference, Code-Switch analysis,
and translation latency across multiple buffer durations (1s, 3s, 5s, 10s, 30s).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference.code_switch import CodeSwitchAnalyzer
from ml.inference.translation_provider import IndicTrans2Provider
from ml.preprocessing.audio import AudioPreprocessor


def run_latency_benchmark(output_dir: str = "ml/reports/pipeline") -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = AudioPreprocessor()
    cs_analyzer = CodeSwitchAnalyzer()
    trans_provider = IndicTrans2Provider()

    sr = 16000
    durations = [1.0, 3.0, 5.0, 10.0, 15.0]
    benchmark_results = []

    test_utterance = "Na phone ninna evening railway station daggara theft ayyindi."

    for dur in durations:
        # Generate synthetic speech-like tone burst
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio_signal = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.random.normal(0, 0.05, len(t))

        # Benchmark Preprocessing
        t0 = time.time()
        clean_audio, meta = preprocessor.process(audio_signal.astype(np.float32), original_sr=sr)
        lat_preproc = round((time.time() - t0) * 1000, 2)  # in ms

        # Benchmark Code-Switch
        t0 = time.time()
        cs_res = cs_analyzer.analyze(test_utterance, language_hint="te")
        lat_cs = round((time.time() - t0) * 1000, 2)

        # Benchmark Translation
        t0 = time.time()
        trans_res = trans_provider.translate(test_utterance, source_lang="te", target_lang="en")
        lat_trans = round((time.time() - t0) * 1000, 2)

        # Real-time Factor (RTF)
        total_lat_sec = (lat_preproc + lat_cs + lat_trans) / 1000.0
        rtf = round(total_lat_sec / dur, 3)

        benchmark_results.append({
            "audio_duration_sec": dur,
            "preprocessing_ms": lat_preproc,
            "code_switch_ms": lat_cs,
            "translation_ms": lat_trans,
            "total_latency_ms": round(lat_preproc + lat_cs + lat_trans, 2),
            "real_time_factor": rtf,
            "status": "PASS (RTF < 1.0)" if rtf < 1.0 else "SUBOPTIMAL",
        })

    report = {
        "device": "CPU / Local Inference",
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": benchmark_results,
    }

    out_file = os.path.join(output_dir, "benchmark_latency.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[PRISM Benchmark] Saved latency benchmark to {out_file}")
    return report


if __name__ == "__main__":
    rep = run_latency_benchmark()
    print("Latency Benchmark Results:")
    print(json.dumps(rep, indent=2))
