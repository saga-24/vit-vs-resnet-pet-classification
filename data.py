"""Data loading for the Oxford-IIIT Pet dataset (37 breeds, auto-downloaded)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_test_dataset(data_dir):
    return datasets.OxfordIIITPet(
        root=data_dir, split="test", target_types="category",
        transform=get_transforms(train=False), download=True)


def get_loaders(data_dir="data", batch_size=32, val_frac=0.1, seed=42,
                num_workers=2, limit=None, pin_memory=False):
    """Returns train/val/test loaders and the list of class names."""
    train_full = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        transform=get_transforms(train=True), download=True)
    val_full = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        transform=get_transforms(train=False), download=True)
    test_set = get_test_dataset(data_dir)

    n = len(train_full)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    test_idx = list(range(len(test_set)))

    if limit:  # tiny subset for a quick smoke test
        train_idx, val_idx, test_idx = train_idx[:limit], val_idx[:limit], test_idx[:limit]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)
    test_sub = Subset(test_set, test_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_sub, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_full.classes
