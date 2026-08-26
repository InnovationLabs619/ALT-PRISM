"""
PRISM NMT (Neural Machine Translation) Fine-Tuning Module
=========================================================
Fine-tunes self-hosted NLLB-200 translation model on Indian multi-lingual and code-switched
police domain statements to clean official English FIR statements.
Saves fine-tuned model artifacts to 'models/prism_nmt_finetuned/'.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / ".cache"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from ml.training.dataset import PoliceDomainCorpus, PRISMNMTDataset


def train_nmt(
    model_name: str = "facebook/nllb-200-distilled-600M",
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    device_str: str = "auto",
    freeze_encoder_layers: int = 6,
) -> Dict[str, Any]:
    """
    Executes domain fine-tuning for NLLB-200 translation model.
    """
    output_path = output_dir or (BASE_DIR / "models" / "prism_nmt_finetuned")
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  PRISM NMT Fine-Tuning Engine: {model_name}")
    print(f"  Target Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"{'='*60}\n")

    # 1. Load Tokenizer and Model
    print(f"[NMT Training] Loading base model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR / "huggingface"),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR / "huggingface"),
        low_cpu_mem_usage=True,
    ).to(device)

    # 2. Freeze full encoder and embeddings for fast CPU domain adaptation
    if hasattr(model, "model"):
        if hasattr(model.model, "encoder"):
            print("[NMT Training] Freezing encoder layers (fine-tuning decoder & LM heads)...", flush=True)
            for param in model.model.encoder.parameters():
                param.requires_grad = False
        if hasattr(model.model, "shared"):
            for param in model.model.shared.parameters():
                param.requires_grad = False

    # 3. Prepare Datasets & DataLoaders
    train_samples, val_samples = PoliceDomainCorpus.get_split(train_ratio=0.8)
    print(f"[NMT Training] Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}", flush=True)

    train_dataset = PRISMNMTDataset(train_samples, tokenizer, max_length=64)
    val_dataset = PRISMNMTDataset(val_samples, tokenizer, max_length=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Optimizer and Scheduler
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if outputs.loss is not None:
                    val_loss += outputs.loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        epoch_duration = round(time.time() - epoch_start, 2)
        model.train()

        print(
            f"  Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration}s",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "epoch_duration_sec": epoch_duration,
        })

    total_training_time = round(time.time() - start_time, 2)
    print(f"\n[NMT Training] Fine-tuning complete in {total_training_time}s.", flush=True)

    # 6. Save Fine-Tuned Artifacts
    print(f"[NMT Training] Saving fine-tuned checkpoint to '{output_path}'...", flush=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save training metadata summary
    summary = {
        "model_type": "NMT",
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

    print(f"[NMT Training] Checkpoint and metadata saved successfully.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="PRISM NMT Fine-Tuning CLI")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_nmt(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
