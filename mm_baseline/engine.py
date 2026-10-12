import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import FaultDataset, load_manifest
from .metadata import fit_numeric
from .metrics import classification_metrics
from .model import FaultClassifier
from .preprocess import read_csv


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def batch_forward(model, batch, tokenizer, device):
    tokens = None
    if tokenizer is not None:
        tokens = tokenizer(list(batch["text"]), padding=True, truncation=True,
                           max_length=model.cfg.max_tokens, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
    return model(batch["visual"].to(device), tokens, batch["numeric"].to(device))


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, num_classes):
    model.eval()
    targets, predictions, probabilities, ids = [], [], [], []
    total_loss = 0.0
    for batch in loader:
        logits = batch_forward(model, batch, tokenizer, device)
        target = batch["label"].to(device)
        total_loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
        probabilities.extend(logits.softmax(-1).cpu().tolist())
        ids.extend(batch["sample_id"])
    metrics = classification_metrics(targets, predictions, num_classes)
    metrics["loss"] = total_loss/len(targets)
    return metrics, list(zip(ids, targets, predictions, probabilities))


def write_predictions(path, records, labels):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "true_label", "predicted_label"] + [f"p_{label}" for label in labels])
        for sample_id, target, predicted, probabilities in records:
            writer.writerow([sample_id, labels[target] if target >= 0 else "", labels[predicted]]+probabilities)


def train(manifest, config_path, output, device_name="auto"):
    cfg = Config.read(config_path)
    seed_all(cfg.seed)
    device = choose_device(device_name)
    rows, labels, preprocessing, root = load_manifest(manifest)
    training_rows = [r for r in rows if r["split"] == "train"]
    numeric_stats = fit_numeric(training_rows) if cfg.fusion == "numeric" else None
    if numeric_stats is not None and any(r["load_unit"] != numeric_stats["load_unit"] for r in rows):
        raise ValueError("All splits must use the same load unit for numeric fusion")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory must be empty; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = FaultDataset([r for r in rows if r["split"] == split], root, cfg, labels,
                               preprocessing["normalization"], numeric_stats)
        loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split == "train",
            num_workers=cfg.workers, generator=torch.Generator().manual_seed(cfg.seed))
    model = FaultClassifier(cfg, len(labels), len(numeric_stats["models"])+3 if numeric_stats else 0).to(device)
    tokenizer = None
    if model.text_encoder is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model, local_files_only=cfg.local_files_only,
                                                   trust_remote_code=False)
        tokenizer.save_pretrained(output / "tokenizer")
    backbones = {id(parameter) for name, parameter in model.named_parameters()
                 if name.startswith(("signal_encoder.", "text_encoder."))}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in backbones]
    encoders = [p for p in model.parameters() if p.requires_grad and id(p) in backbones]
    groups = [{"params": heads, "lr": cfg.learning_rate}]
    if encoders:
        groups.append({"params": encoders, "lr": cfg.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    run_info = {"config": cfg.to_dict(), "classes": labels, "preprocessing": preprocessing,
                "numeric_stats": numeric_stats, "device": str(device),
                "torch_version": str(torch.__version__),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "split_counts": {s: len(loader.dataset) for s, loader in loaders.items()}}
    (output / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    best, history = -1.0, []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = batch_forward(model, batch, tokenizer, device)
            loss = nn.functional.cross_entropy(logits, batch["label"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()*len(batch["label"])
            seen += len(batch["label"])
        validation, _ = evaluate(model, loaders["val"], tokenizer, device, len(labels))
        record = {"epoch": epoch+1, "train_loss": running_loss/seen, "validation": validation}
        history.append(record)
        print(f"epoch={epoch+1} train_loss={record['train_loss']:.4f} "
              f"val_acc={validation['accuracy']:.4f} val_macro_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            torch.save({"format_version": 1, "state_dict": model.state_dict(), "config": cfg.to_dict(),
                "classes": labels, "preprocessing": preprocessing, "numeric_stats": numeric_stats,
                "text_config": model.text_config(), "epoch": epoch+1}, output / "best.pt")
        scheduler.step()
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, records = evaluate(model, loaders["test"], tokenizer, device, len(labels))
    test_metrics["classes"] = labels
    test_metrics["selected_epoch"] = checkpoint["epoch"]
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    write_predictions(output / "test_predictions.csv", records, labels)
    print(json.dumps({"output": str(output), "test": test_metrics}, ensure_ascii=False), flush=True)
    return output / "best.pt"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    cfg = Config(**checkpoint["config"]).validate()
    stats = checkpoint["numeric_stats"]
    model = FaultClassifier(cfg, len(checkpoint["classes"]), len(stats["models"])+3 if stats else 0,
                            checkpoint["text_config"], initialize_pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = None
    if model.text_encoder is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(Path(path).resolve().parent / "tokenizer", local_files_only=True,
                                                   trust_remote_code=False)
    return model, tokenizer, checkpoint


@torch.no_grad()
def predict(checkpoint_path, manifest, output, split="test", device_name="auto"):
    device = choose_device(device_name)
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device)
    manifest = Path(manifest).resolve()
    rows = read_csv(manifest)
    rows = [r for r in rows if split == "all" or r.get("split") == split]
    if not rows:
        raise ValueError("No prediction samples match --split")
    metadata_path = manifest.parent / "preprocessing.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["preprocessing"] != checkpoint["preprocessing"]["preprocessing"]:
            raise ValueError("Prediction STFT settings differ from checkpoint")
        if model.cfg.input_kind == "png" and metadata["normalization"] != checkpoint["preprocessing"]["normalization"]:
            raise ValueError("PNG gray mapping differs from training; prepare with --reference-preprocessing")
    # Always use training-time statistics stored in the checkpoint.
    dataset = FaultDataset(rows, manifest.parent, model.cfg, checkpoint["classes"],

        checkpoint["preprocessing"]["normalization"], checkpoint["numeric_stats"], inference=True)
    loaders = DataLoader(dataset, batch_size=model.cfg.batch_size, num_workers=model.cfg.workers)
    records = []
    for batch in loaders:
        logits = batch_forward(model, batch, tokenizer, device)
        for sid, prediction, probabilities in zip(batch["sample_id"], logits.argmax(-1).cpu().tolist(),
                                                   logits.softmax(-1).cpu().tolist()):
            records.append((sid, -1, prediction, probabilities))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, records, checkpoint["classes"])
    print(f"Saved {len(records)} predictions to {output}", flush=True)
    return records
