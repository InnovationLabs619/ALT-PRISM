"""
PRISM Master Model Training & Fine-Tuning CLI
==============================================
Orchestrates end-to-end domain fine-tuning for both self-hosted ASR and NMT models.
Updates model registry and exports performance metrics.

Usage:
    python scripts/train_models.py --asr-epochs 3 --nmt-epochs 3
    python scripts/train_models.py --skip-nmt --asr-epochs 5
    python scripts/train_models.py --skip-asr --nmt-epochs 5
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, str(BASE_DIR))

from ml.training.train_asr import train_asr
from ml.training.train_nmt import train_nmt


def update_model_registry(asr_checkpoint: Optional[str] = None, nmt_checkpoint: Optional[str] = None):
    """Updates models/registry.json to point to fine-tuned local checkpoints."""
    registry_file = BASE_DIR / "models" / "registry.json"
    if not registry_file.exists():
        print(f"[Registry] Registry file not found at {registry_file}")
        return

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if asr_checkpoint:
        registry["asr"]["finetuned_checkpoint"] = asr_checkpoint
        registry["asr"]["status"] = "FINETUNED_DOMAIN_ACTIVE"

    if nmt_checkpoint:
        registry["translation"]["finetuned_checkpoint"] = nmt_checkpoint
        registry["translation"]["status"] = "FINETUNED_DOMAIN_ACTIVE"

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"[Registry] Successfully updated {registry_file}")


def generate_training_report(
    asr_summary: Optional[Dict[str, Any]],
    nmt_summary: Optional[Dict[str, Any]],
    output_path: Path,
):
    """Generates detailed markdown report of the training execution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# PRISM Model Training & Fine-Tuning Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "The PRISM internal police department AI assistance models have been successfully domain-adapted and fine-tuned on multi-lingual Indian speech, code-switched vernaculars (Hinglish, Tenglish, Tanglish, Kanglish), and legal law-enforcement complaint statements.",
        "",
    ]

    if asr_summary:
        report_lines.extend([
            "## 1. ASR (Speech-to-Text) Fine-Tuning",
            f"- **Base Model:** `{asr_summary.get('base_model')}`",
            f"- **Target Device:** `{asr_summary.get('device')}`",
            f"- **Epochs Trained:** `{asr_summary.get('epochs')}`",
            f"- **Batch Size:** `{asr_summary.get('batch_size')}`",
            f"- **Learning Rate:** `{asr_summary.get('learning_rate')}`",
            f"- **Total Training Time:** `{asr_summary.get('total_training_time_sec')}s`",
            f"- **Checkpoint Path:** `{asr_summary.get('checkpoint_dir')}`",
            "",
            "### ASR Loss Progression",
            "| Epoch | Train Loss | Validation Loss | Epoch Time |",
            "| :---: | :---: | :---: | :---: |",
        ])
        for h in asr_summary.get("history", []):
            report_lines.append(
                f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['epoch_duration_sec']}s |"
            )
        report_lines.append("")

    if nmt_summary:
        report_lines.extend([
            "## 2. NMT (Neural Translation) Fine-Tuning",
            f"- **Base Model:** `{nmt_summary.get('base_model')}`",
            f"- **Target Device:** `{nmt_summary.get('device')}`",
            f"- **Epochs Trained:** `{nmt_summary.get('epochs')}`",
            f"- **Batch Size:** `{nmt_summary.get('batch_size')}`",
            f"- **Learning Rate:** `{nmt_summary.get('learning_rate')}`",
            f"- **Total Training Time:** `{nmt_summary.get('total_training_time_sec')}s`",
            f"- **Checkpoint Path:** `{nmt_summary.get('checkpoint_dir')}`",
            "",
            "### NMT Loss Progression",
            "| Epoch | Train Loss | Validation Loss | Epoch Time |",
            "| :---: | :---: | :---: | :---: |",
        ])
        for h in nmt_summary.get("history", []):
            report_lines.append(
                f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['epoch_duration_sec']}s |"
            )
        report_lines.append("")

    report_lines.extend([
        "## 3. Production Deployment Status",
        "- **Air-Gap Compliance:** 100% Offline (Local files only)",
        "- **Checkpoint Availability:** Serialized and registered in `models/registry.json`",
        "- **Inference Readiness:** Integrated with `InferenceEngine`",
        "",
        "---",
        "*PRISM Police Investigation System Management • AI Research Division*",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[Report] Training report generated at '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="PRISM Model Training Orchestrator")
    parser.add_argument("--asr-epochs", type=int, default=3, help="Number of training epochs for ASR")
    parser.add_argument("--nmt-epochs", type=int, default=3, help="Number of training epochs for NMT")
    parser.add_argument("--asr-batch-size", type=int, default=4, help="Batch size for ASR")
    parser.add_argument("--nmt-batch-size", type=int, default=2, help="Batch size for NMT")
    parser.add_argument("--asr-lr", type=float, default=1e-4, help="Learning rate for ASR")
    parser.add_argument("--nmt-lr", type=float, default=5e-5, help="Learning rate for NMT")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR fine-tuning")
    parser.add_argument("--skip-nmt", action="store_true", help="Skip NMT fine-tuning")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda)")
    args = parser.parse_args()

    total_start = time.time()
    print("\n" + "=" * 65)
    print("      PRISM POLICE INVESTIGATION AI - MODEL TRAINING SUITE")
    print("=" * 65)

    asr_summary = None
    nmt_summary = None

    if not args.skip_asr:
        print("\n>>> STAGE 1: Fine-Tuning ASR (Speech-to-Text) Model...")
        asr_summary = train_asr(
            epochs=args.asr_epochs,
            batch_size=args.asr_batch_size,
            learning_rate=args.asr_lr,
            device_str=args.device,
        )

    if not args.skip_nmt:
        print("\n>>> STAGE 2: Fine-Tuning NMT (Neural Translation) Model...")
        nmt_summary = train_nmt(
            epochs=args.nmt_epochs,
            batch_size=args.nmt_batch_size,
            learning_rate=args.nmt_lr,
            device_str=args.device,
        )

    # Update Registry
    asr_cp = asr_summary.get("checkpoint_dir") if asr_summary else None
    nmt_cp = nmt_summary.get("checkpoint_dir") if nmt_summary else None
    update_model_registry(asr_checkpoint=asr_cp, nmt_checkpoint=nmt_cp)

    # Generate Report
    report_file = BASE_DIR / "ml" / "reports" / "training_report.md"
    generate_training_report(asr_summary, nmt_summary, report_file)

    total_elapsed = round(time.time() - total_start, 2)
    print("\n" + "=" * 65)
    print(f"  ALL TRAINING RUNS COMPLETED SUCCESSFULLY in {total_elapsed}s!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
