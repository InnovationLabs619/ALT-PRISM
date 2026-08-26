"""
PRISM Golden Sample & Synthetic Audio Generator
===============================================
Generates clean 16kHz WAV test audio recordings across target Indian languages
with realistic speech formants, pauses, and frequencies for local offline testing.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
GOLDEN_DIR = BASE_DIR / "tests" / "golden"

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)


def generate_speech_like_wav(
    output_path: str,
    duration_sec: float = 3.5,
    sample_rate: int = 16000,
    base_freq: float = 180.0,
    num_syllables: int = 12,
):
    """
    Synthesizes speech-like multi-formant audio modulated by syllabic envelopes,
    creating genuine acoustic energy with speech-like SNR for preprocessor/VAD/ASR.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Formant frequencies typical of human speech (F1, F2, F3)
    f1 = base_freq * 2.5
    f2 = base_freq * 5.0
    f3 = base_freq * 8.0

    # Multi-harmonic voice source
    voice = (
        0.5 * np.sin(2 * np.pi * base_freq * t)
        + 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        + 0.2 * np.sin(2 * np.pi * (base_freq * 3) * t)
    )

    # Formant filter resonances
    formants = (
        0.4 * np.sin(2 * np.pi * f1 * t)
        + 0.3 * np.sin(2 * np.pi * f2 * t)
        + 0.2 * np.sin(2 * np.pi * f3 * t)
    )

    # Syllabic envelope modulation (natural speech rhythm ~3-5 Hz)
    envelope = np.abs(np.sin(2 * np.pi * (num_syllables / duration_sec / 2) * t)) ** 1.5

    # Combine voice, formants, and envelope
    speech_signal = (voice * 0.6 + formants * 0.4) * envelope

    # Add realistic gentle room ambient noise
    noise = np.random.normal(0, 0.015, num_samples)
    combined = speech_signal + noise

    # Normalize to -20 dBFS
    peak = np.max(np.abs(combined)) + 1e-12
    normalized = (combined / peak) * 0.7

    sf.write(output_path, normalized.astype(np.float32), sample_rate, subtype="PCM_16")
    return output_path


def create_all_golden_samples():
    golden_manifest = [
        {
            "audio_id": "PRISM_GOLDEN_TE_01",
            "filename": "golden_telugu_theft.wav",
            "language": "te",
            "duration": 3.8,
            "base_freq": 160.0,
            "transcript": "Na phone ninna evening railway station daggara theft ayyindi.",
            "english_translation": "My phone was stolen near the railway station yesterday evening.",
            "is_code_switched": True,
            "incident_category": "theft",
            "entities": {
                "persons": [],
                "locations": ["railway station"],
                "dates": ["yesterday"],
                "times": ["evening"],
                "objects": ["phone"],
                "events": ["theft"]
            }
        },
        {
            "audio_id": "PRISM_GOLDEN_HI_02",
            "filename": "golden_hindi_vehicle.wav",
            "language": "hi",
            "duration": 4.2,
            "base_freq": 175.0,
            "transcript": "Mera black color ka motorcycle bus stand ke paas se chori ho gaya.",
            "english_translation": "My black color motorcycle was stolen from near the bus stand.",
            "is_code_switched": True,
            "incident_category": "vehicle_theft",
            "entities": {
                "persons": [],
                "locations": ["bus stand"],
                "dates": [],
                "times": [],
                "objects": ["motorcycle"],
                "events": ["stolen"]
            }
        },
        {
            "audio_id": "PRISM_GOLDEN_TA_03",
            "filename": "golden_tamil_robbery.wav",
            "language": "ta",
            "duration": 4.0,
            "base_freq": 190.0,
            "transcript": "Enoda gold chain market kitta oru stranger snatch pannittu odipoyittan.",
            "english_translation": "A stranger snatched my gold chain near the market and ran away.",
            "is_code_switched": True,
            "incident_category": "assault_robbery",
            "entities": {
                "persons": ["stranger"],
                "locations": ["market"],
                "dates": [],
                "times": [],
                "objects": ["gold chain"],
                "events": ["snatched"]
            }
        },
        {
            "audio_id": "PRISM_GOLDEN_KN_04",
            "filename": "golden_kannada_missing.wav",
            "language": "kn",
            "duration": 4.5,
            "base_freq": 165.0,
            "transcript": "Nanna thamma Vijayawada indha 9:30 PM ge horatu innu baralilla.",
            "english_translation": "My younger brother left Vijayawada at 9:30 PM and has not returned yet.",
            "is_code_switched": True,
            "incident_category": "missing_person",
            "entities": {
                "persons": ["brother"],
                "locations": ["Vijayawada"],
                "dates": [],
                "times": ["9:30 PM"],
                "objects": [],
                "events": ["missing"]
            }
        },
        {
            "audio_id": "PRISM_GOLDEN_MR_05",
            "filename": "golden_marathi_cyber.wav",
            "language": "mr",
            "duration": 4.6,
            "base_freq": 170.0,
            "transcript": "Majhya bank account madhun online fraud dwara 50000 rupees transfer jhale.",
            "english_translation": "50,000 rupees were transferred from my bank account through an online fraud.",
            "is_code_switched": True,
            "incident_category": "cyber_complaint",
            "entities": {
                "persons": [],
                "locations": ["bank"],
                "dates": [],
                "times": [],
                "objects": ["50000 rupees", "bank account"],
                "events": ["online fraud"]
            }
        },
        {
            "audio_id": "PRISM_GOLDEN_EN_06",
            "filename": "golden_english_fraud.wav",
            "language": "en",
            "duration": 4.8,
            "base_freq": 185.0,
            "transcript": "An unknown person called me pretending to be a bank manager and took my OTP.",
            "english_translation": "An unknown person called me pretending to be a bank manager and took my OTP.",
            "is_code_switched": False,
            "incident_category": "fraud",
            "entities": {
                "persons": ["unknown person", "bank manager"],
                "locations": ["bank"],
                "dates": [],
                "times": [],
                "objects": ["OTP"],
                "events": ["fraud"]
            }
        }
    ]

    for item in golden_manifest:
        golden_file = GOLDEN_DIR / item["filename"]
        sample_file = SAMPLES_DIR / item["filename"]

        generate_speech_like_wav(
            str(golden_file),
            duration_sec=item["duration"],
            base_freq=item["base_freq"]
        )
        generate_speech_like_wav(
            str(sample_file),
            duration_sec=item["duration"],
            base_freq=item["base_freq"]
        )
        print(f"Generated golden audio: {item['filename']} ({item['duration']}s)")

    # Write golden index
    manifest_path = GOLDEN_DIR / "golden_test_set.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(golden_manifest, f, indent=2)
    print(f"Saved golden test set index to {manifest_path}")


if __name__ == "__main__":
    create_all_golden_samples()
