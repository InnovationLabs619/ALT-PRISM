"""
PRISM IndicVoices Dataset Ingestion Pipeline
============================================
Fetches, normalizes, and manifests real Indic speech samples from the AI4Bharat IndicVoices
dataset (via Hugging Face Datasets-Server API) for testing, evaluation, and fine-tuning.

Usage:
    python scripts/fetch_indic_voices.py --token <HF_TOKEN> --lang telugu --limit 10
    python scripts/fetch_indic_voices.py --token <HF_TOKEN> --lang assamese --limit 5
    python scripts/fetch_indic_voices.py --token <HF_TOKEN> --all-supported --limit 5
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing.audio import AudioPreprocessor

INDIC_VOICES_DATASET = "ai4bharat/IndicVoices"
HF_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"

# Supported language configs in IndicVoices
SUPPORTED_CONFIGS = [
    "assamese",
    "bengali",
    "gujarati",
    "hindi",
    "kannada",
    "malayalam",
    "marathi",
    "odia",
    "punjabi",
    "tamil",
    "telugu",
    "urdu",
]

# Mapping to ISO 639-1 / 639-3 codes
LANG_CODE_MAP = {
    "assamese": "as",
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}


def fetch_indic_voices_rows(
    config: str,
    token: Optional[str] = None,
    split: str = "valid",
    offset: int = 0,
    length: int = 10,
) -> Dict[str, Any]:
    """Queries Hugging Face Datasets-Server API for rows."""
    headers = {}
    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    params = {
        "dataset": INDIC_VOICES_DATASET,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }

    print(f"[IndicVoices] Querying {INDIC_VOICES_DATASET} (config='{config}', split='{split}', limit={length})...")
    response = requests.get(HF_DATASETS_SERVER_URL, headers=headers, params=params, timeout=30)

    if response.status_code == 401 or response.status_code == 403:
        raise PermissionError(
            f"Authentication failed (HTTP {response.status_code}). Please provide a valid HF_TOKEN."
        )
    elif response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch rows (HTTP {response.status_code}): {response.text}"
        )

    return response.json()


def process_and_save_samples(
    rows_data: Dict[str, Any],
    config: str,
    output_dir: Path,
    manifest_file: Path,
    preprocessor: AudioPreprocessor,
) -> List[Dict[str, Any]]:
    """Downloads audio assets, normalizes to 16kHz mono WAV, and writes JSONL manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    rows = rows_data.get("rows", [])
    print(f"[IndicVoices] Processing {len(rows)} samples for '{config}'...")

    processed_records = []

    for idx, item in enumerate(rows):
        row = item.get("row", {})
        row_idx = item.get("row_idx", idx)

        # Extract transcript (fields vary across dataset configs)
        transcript = (
            row.get("verbatim_transcription")
            or row.get("normalized_transcription")
            or row.get("transcription")
            or row.get("text")
            or row.get("sentence")
            or ""
        )

        audio_field = row.get("audio")
        audio_bytes = None
        duration = 0.0

        if isinstance(audio_field, list) and len(audio_field) > 0:
            audio_field = audio_field[0]

        if isinstance(audio_field, dict):
            src_url = audio_field.get("src")
            if src_url:
                try:
                    res = requests.get(src_url, timeout=20)
                    if res.status_code == 200:
                        audio_bytes = res.content
                except Exception as e:
                    print(f"  [Warning] Failed to download audio from {src_url}: {e}")

            # Alternatively, check for bytes payload
            if not audio_bytes and "bytes" in audio_field and audio_field["bytes"]:
                import base64
                raw_b = audio_field["bytes"]
                audio_bytes = base64.b64decode(raw_b) if isinstance(raw_b, str) else raw_b

        audio_filename = f"indicvoices_{config}_{row_idx:04d}.wav"
        audio_save_path = output_dir / audio_filename

        if audio_bytes:
            try:
                # Preprocess & normalize audio to 16kHz mono
                clean_audio, meta = preprocessor.process(audio_bytes)
                sf.write(str(audio_save_path), clean_audio, meta.sample_rate, subtype="PCM_16")
                duration = meta.duration_sec
            except Exception as e:
                print(f"  [Warning] Preprocessing error on row {row_idx}: {e}")
                # Save raw bytes directly if preprocessor failed
                with open(audio_save_path, "wb") as f:
                    f.write(audio_bytes)

        record = {
            "id": f"indicvoices_{config}_{row_idx:04d}",
            "config": config,
            "language": LANG_CODE_MAP.get(config, config[:2]),
            "filename": audio_filename,
            "filepath": str(audio_save_path.relative_to(BASE_DIR)),
            "duration": round(duration, 2),
            "transcript": transcript,
            "raw_metadata": {k: v for k, v in row.items() if k != "audio"},
        }
        processed_records.append(record)
        print(f"  ✓ [{config}] Row {row_idx}: '{transcript[:45]}...' ({duration:.1f}s)")

    # Append or write JSONL manifest
    with open(manifest_file, "a" if manifest_file.exists() else "w", encoding="utf-8") as f:
        for r in processed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[IndicVoices] Saved {len(processed_records)} records to {manifest_file}")
    return processed_records


def main():
    parser = argparse.ArgumentParser(description="Fetch and ingest AI4Bharat IndicVoices dataset samples.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Bearer HF_TOKEN)")
    parser.add_argument("--lang", type=str, default="telugu", help=f"Language configuration name (Options: {', '.join(SUPPORTED_CONFIGS)})")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split (train, valid, test)")
    parser.add_argument("--offset", type=int, default=0, help="Offset row index")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to fetch")
    parser.add_argument("--all-supported", action="store_true", help="Fetch samples for all supported Indic languages")
    parser.add_argument("--inspect-only", action="store_true", help="Print dataset metadata rows without saving files")

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("[Notice] No HF_TOKEN provided. You can pass it with --token <YOUR_HF_TOKEN> or export HF_TOKEN=<token>.")

    preprocessor = AudioPreprocessor()
    target_configs = SUPPORTED_CONFIGS if args.all_supported else [args.lang.lower()]

    for cfg in target_configs:
        if cfg not in SUPPORTED_CONFIGS:
            print(f"[Warning] '{cfg}' not recognized. Supported: {SUPPORTED_CONFIGS}")
            continue

        try:
            data = fetch_indic_voices_rows(
                config=cfg,
                token=token,
                split=args.split,
                offset=args.offset,
                length=args.limit,
            )

            if args.inspect_only:
                print(f"\n--- Preview Rows for '{cfg}' ---")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                continue

            output_dir = BASE_DIR / "data" / "samples" / "indic_voices" / cfg
            manifest_file = BASE_DIR / "data" / "manifests" / f"indic_voices_{cfg}.jsonl"

            process_and_save_samples(
                rows_data=data,
                config=cfg,
                output_dir=output_dir,
                manifest_file=manifest_file,
                preprocessor=preprocessor,
            )

        except Exception as e:
            print(f"[Error] Failed processing config '{cfg}': {e}")


if __name__ == "__main__":
    main()
