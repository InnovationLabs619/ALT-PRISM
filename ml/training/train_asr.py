"""
PRISM ASR Fine-Tuning Module
=============================
Fine-tunes self-hosted Whisper ASR model on Indian multi-lingual and police domain speech.
Saves fine-tuned model artifacts to 'models/prism_asr_finetuned/'.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

CACHE_DIR = BASE_DIR / ".cache"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, get_linear_schedule_with_warmup

from ml.training.dataset import PoliceDomainCorpus, PRISMASRDataset



def train_asr(
    model_name: str = "openai/whisper-tiny",
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device_str: str = "auto",
    freeze_encoder: bool = True,
) -> Dict[str, Any]:
    """
    Executes domain fine-tuning for Whisper ASR model.
    """
    output_path = output_dir or (BASE_DIR / "models" / "prism_asr_finetuned")
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  PRISM ASR Fine-Tuning Engine: {model_name}")
    print(f"  Target Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"{'='*60}\n")

    # 1. Load Processor and Model
    print(f"[ASR Training] Loading base model '{model_name}'...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR / "huggingface"),
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR / "huggingface"),
        low_cpu_mem_usage=True,
    ).to(device)

    # 2. Freeze encoder if specified to focus on decoder domain adaptation
    if freeze_encoder:
        print("[ASR Training] Freezing acoustic encoder layers (fine-tuning decoder & projection heads)...")
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    # 3. Prepare Datasets & DataLoaders
    train_samples, val_samples = PoliceDomainCorpus.get_split(train_ratio=0.8)
    print(f"[ASR Training] Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_dataset = PRISMASRDataset(train_samples, processor, augment=True)
    val_dataset = PRISMASRDataset(val_samples, processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5. Training Loop
    history: List[Dict[str, float]] = []
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                batch_count += 1

        avg_train_loss = train_loss / max(1, batch_count)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_features=input_features, labels=labels)
                if outputs.loss is not None:
                    val_loss += outputs.loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        epoch_duration = round(time.time() - epoch_start, 2)
        model.train()

        print(
            f"  Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "epoch_duration_sec": epoch_duration,
        })

    total_training_time = round(time.time() - start_time, 2)
    print(f"\n[ASR Training] Fine-tuning complete in {total_training_time}s.")

    # 6. Save Fine-Tuned Artifacts
    print(f"[ASR Training] Saving fine-tuned checkpoint to '{output_path}'...")
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    # Save training metadata summary
    summary = {
        "model_type": "ASR",
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "total_training_time_sec": total_training_time,
        "history": history,
        "checkpoint_dir": str(output_path.relative_to(BASE_DIR)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ASR Training] Checkpoint and metadata saved successfully.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="PRISM ASR Fine-Tuning CLI")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_asr(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
