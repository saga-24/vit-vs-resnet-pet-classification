"""Visualize what the fine-tuned ViT looks at using attention rollout.

Usage: python attention.py [--num-images 6]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MEAN, STD, get_test_dataset
from models import build_model


class AttnRecorder(nn.Module):
    """Wraps nn.MultiheadAttention so the attention weights can be read after a forward pass."""

    def __init__(self, mha):
        super().__init__()
        self.mha = mha
        self.attn = None

    def forward(self, query, key, value, **kwargs):
        out, weights = self.mha(query, key, value, need_weights=True, average_attn_weights=True)
        self.attn = weights.detach()
        return out, None


def attention_rollout(attn_list):
    """Abdou et al. / Dosovitskiy-style rollout across layers. Each item: (B, T, T)."""
    result = None
    for a in attn_list:
        a = a + torch.eye(a.size(-1), device=a.device)  # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--ckpt", default="checkpoints/vit.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/attention_maps.png")
    args = p.parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint {args.ckpt} not found. Run: python train.py --model vit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model("vit", len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    recorders = []
    for block in model.encoder.layers:
        rec = AttnRecorder(block.self_attention)
        block.self_attention = rec
        recorders.append(rec)

    dataset = get_test_dataset(args.data_dir)
    step = max(1, len(dataset) // args.num_images)
    indices = [i * step for i in range(args.num_images)]

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, args.num_images, figsize=(3 * args.num_images, 6.4))
    axes = axes.reshape(2, args.num_images)
    for col, idx in enumerate(indices):
        x, y = dataset[idx]
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device))
        pred = logits.argmax(1).item()

        rolled = attention_rollout([r.attn for r in recorders])  # (1, 197, 197)
        cls_to_patches = rolled[0, 0, 1:]
        grid = int(cls_to_patches.numel() ** 0.5)  # 14 for ViT-B/16 at 224px
        heat = cls_to_patches.reshape(1, 1, grid, grid)
        heat = F.interpolate(heat, size=224, mode="bilinear", align_corners=False)[0, 0].cpu()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"True: {classes[y]}\nPred: {classes[pred]}", fontsize=8,
                               color="green" if pred == y else "red")
        axes[1, col].imshow(img)
        axes[1, col].imshow(heat.numpy(), cmap="jet", alpha=0.45)
        axes[1, col].set_title("Attention rollout", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
