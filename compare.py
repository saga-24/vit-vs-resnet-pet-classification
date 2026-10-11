"""Compare ViT vs ResNet results. Run after training both models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results")


def load(name):
    path = OUT / f"{name}_metrics.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python train.py --model {name}")
    return json.loads(path.read_text())


def main():
    vit, res = load("vit"), load("resnet")
    rows = [
        ("Test accuracy", f"{vit['test_accuracy']*100:.2f}%", f"{res['test_accuracy']*100:.2f}%"),
        ("Test macro-F1", f"{vit['test_macro_f1']:.4f}", f"{res['test_macro_f1']:.4f}"),
        ("Parameters", f"{vit['parameters_millions']}M", f"{res['parameters_millions']}M"),
        ("Training time", f"{vit['train_time_seconds']:.0f}s", f"{res['train_time_seconds']:.0f}s"),
    ]
    lines = ["| Metric | ViT-B/16 | ResNet-50 |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    table = "\n".join(lines)
    print(table)
    diff = (vit["test_accuracy"] - res["test_accuracy"]) * 100
    print(f"\nViT vs ResNet accuracy difference: {diff:+.2f} percentage points")
    (OUT / "comparison.md").write_text(table + f"\n\nAccuracy difference (ViT - ResNet): {diff:+.2f} pp\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m, label in ((vit, "ViT-B/16"), (res, "ResNet-50")):
        ep = range(1, len(m["history"]["val_acc"]) + 1)
        axes[0].plot(ep, m["history"]["val_acc"], marker="o", label=label)
        axes[1].plot(ep, m["history"]["train_loss"], marker="o", label=label)
    axes[0].set_title("Validation accuracy")
    axes[1].set_title("Training loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    print(f"Saved {OUT / 'comparison.md'} and {OUT / 'training_curves.png'}")


if __name__ == "__main__":
    main()
