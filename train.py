"""Fine-tune ViT-B/16 or ResNet-50 on Oxford-IIIT Pets and evaluate on the test set.

Usage:
    python train.py --model vit
    python train.py --model resnet
    python train.py --model vit --quick     # tiny smoke test
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import get_loaders
from models import build_model, count_params

DEFAULT_LR = {"vit": 5e-5, "resnet": 2e-4}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / len(labels), float((preds == labels).mean()), preds, labels


def plot_confusion(cm, classes, path, title):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm, classes, k=5):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    out = []
    for idx in flat:
        i, j = np.unravel_index(idx, cm.shape)
        if cm[i, j] > 0:
            out.append({"true": classes[i], "predicted": classes[j], "count": int(cm[i, j])})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vit", "resnet"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--quick", action="store_true", help="tiny run to verify setup")
    args = p.parse_args()

    if args.quick:
        args.epochs, args.batch_size = 1, 8
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | LR: {lr} | Epochs: {args.epochs}")
    if device.type == "cpu" and not args.quick:
        print("WARNING: training on CPU will be very slow. Use a GPU (e.g. free Colab/Kaggle).")

    out_dir, ckpt_dir = Path(args.out_dir), Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_loaders(
        args.data_dir, args.batch_size, seed=args.seed, num_workers=args.workers,
        limit=64 if args.quick else None, pin_memory=device.type == "cuda")
    print(f"Classes: {len(classes)} | train: {len(train_loader.dataset)} | "
          f"val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=len(classes), pretrained=True).to(device)
    n_params = count_params(model)
    print(f"Parameters: {n_params / 1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_path = -1.0, ckpt_dir / f"{args.model}.pt"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(run_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{args.epochs} | train loss {run_loss / seen:.4f} acc {correct / seen:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s elapsed")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "model_name": args.model}, best_path)

    train_time = time.time() - start

    # ---- Final evaluation on the held-out test set using the best checkpoint ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    print(f"\nTEST | loss {test_loss:.4f} | accuracy {test_acc:.4f} | macro-F1 {macro_f1:.4f}")

    plot_confusion(cm, classes, out_dir / f"{args.model}_confusion_matrix.png",
                   f"{args.model.upper()} - normalized confusion matrix (test set)")
    report = classification_report(labels, preds, labels=list(range(len(classes))),
                                   target_names=classes, zero_division=0)
    (out_dir / f"{args.model}_classification_report.txt").write_text(report)

    metrics = {
        "model": args.model, "test_accuracy": test_acc, "test_macro_f1": macro_f1,
        "best_val_accuracy": best_val, "parameters_millions": round(n_params / 1e6, 1),
        "epochs": args.epochs, "learning_rate": lr, "batch_size": args.batch_size,
        "train_time_seconds": round(train_time, 1), "num_classes": len(classes),
        "num_train_images": len(train_loader.dataset),
        "num_test_images": len(test_loader.dataset),
        "top_confusions": top_confusions(cm, classes),
        "history": history, "quick_run": args.quick,
    }
    (out_dir / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved results to {out_dir}/ and checkpoint to {best_path}")


if __name__ == "__main__":
    main()
