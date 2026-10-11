"""Model builders: ViT-B/16 (main model) and ResNet-50 (CNN baseline)."""
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "vit":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == "resnet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Use 'vit' or 'resnet'.")
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
