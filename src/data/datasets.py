"""
Dataset pipelines for SOTLI (Structured Off-Task Learning Intervals).
Standardized on 32x32 RGB inputs across all tasks.
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np


# Standard ImageNet normalization stats (for pretrained 224x224 vision regime)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Standard CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Standard CIFAR-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Standard STL-10 normalization stats
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

# Standard SVHN normalization stats
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


class FixedGaussianNoiseDataset(Dataset):
    """
    Condition B7: Synthetic Random-Input Control.
    Generates a fixed set of synthetic Gaussian noise images with fixed random labels.
    Standardized to 32x32x3 RGB images with 10 classes.
    """
    def __init__(self, num_samples=10000, num_classes=10, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Deterministically generate fixed synthetic tensors
        rng = np.random.RandomState(seed)
        # Sample normal distribution, clip to standard image range [0, 1]
        data = rng.randn(num_samples, 3, 32, 32).astype(np.float32)
        data = np.clip((data * 0.2) + 0.5, 0.0, 1.0)
        self.data = torch.from_numpy(data)
        
        # Fixed random uniform labels
        labels = rng.randint(0, num_classes, size=(num_samples,))
        self.labels = torch.from_numpy(labels).long()
        
        # Normalize to standard scale
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.normalize(img)
        label = self.labels[idx]
        return img, label


def get_cifar100_loaders(data_dir="./data_cache", batch_size=128, num_workers=2, val_size=5000, seed=42, return_val_indices=False):
    """
    Returns train, validation, and test loaders for Target Task A (CIFAR-100).
    50,000 train set is split into (50,000 - val_size) train and val_size validation.

    Returns:
        If return_val_indices is False: (train_loader, val_loader, test_loader)
        If return_val_indices is True:  (train_loader, val_loader, test_loader, val_indices)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_set_base = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # Stratified train/val split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(full_train))
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_set_base, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if return_val_indices:
        return train_loader, val_loader, test_loader, val_indices
    return train_loader, val_loader, test_loader


def get_auxiliary_loader(condition_id, data_dir="./data_cache", batch_size=128, num_workers=2, seed=42,
                         exclude_indices=None):
    """
    Returns an infinite or repeating DataLoader for auxiliary interval tasks B.
    Supported condition_id:
      - 'B0': Continuous CIFAR-100 (Target A)
      - 'B1': Head-Switch CIFAR-100 (Target A with temporary 100-class head)
      - 'B3': CIFAR-10 (Related natural images, 10 classes)
      - 'B4': STL-10 (Distinct natural images, resized 32x32, 10 classes)
      - 'B5': SVHN (Distant street digits, cropped 32x32, 10 classes)
      - 'B7': Fixed Gaussian Noise (10,000 samples, 10 classes)

    Args:
        exclude_indices: Optional list/array of indices to exclude from the dataset.
            Used by B0/B1 to prevent training on validation examples (data leakage fix).
            If None and condition_id in ['B0', 'B1'], automatically computes the 5,000
            validation indices using the provided seed.
    """
    if condition_id in ['B0', 'B1']:
        # Uses Target Task A dataset
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        # Exclude validation indices to prevent data leakage (P0-A fix)
        if exclude_indices is None:
            rng = np.random.RandomState(seed)
            indices = np.arange(len(dataset))
            rng.shuffle(indices)
            exclude_indices = indices[:5000]

        all_indices = set(range(len(dataset)))
        keep_indices = sorted(all_indices - set(exclude_indices))
        dataset = Subset(dataset, keep_indices)
        num_classes = 100

    elif condition_id == 'B3':
        # CIFAR-10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        num_classes = 10

    elif condition_id == 'B4':
        # STL-10 labeled train set, resized to 32x32
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ])
        dataset = torchvision.datasets.STL10(root=data_dir, split='train', download=True, transform=train_transform)
        num_classes = 10

    elif condition_id == 'B5':
        # SVHN train set (digits, no horizontal flip)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        dataset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
        num_classes = 10

    elif condition_id == 'B7':
        # Fixed Gaussian noise control
        dataset = FixedGaussianNoiseDataset(num_samples=10000, num_classes=10, seed=seed)
        num_classes = 10

    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader, num_classes


def get_fixed_probe_set(data_dir="./data_cache", num_samples=2000, seed=42):
    """
    Returns a fixed, deterministic set of Task A images for tracking representation drift (CKA)
    and Effective Representation Rank across checkpoints theta_1, theta_2, theta_3.
    """
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    full_val = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)
    
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(full_val), size=num_samples, replace=False)
    probe_subset = Subset(full_val, indices)
    probe_loader = DataLoader(probe_subset, batch_size=128, shuffle=False, num_workers=2)
    return probe_loader


# =====================================================================
# Applied Fine-Tuning Regime: 224x224 Datasets (ISIC 2019 & Imagenette)
# =====================================================================

class ISIC2019Dataset(Dataset):
    """
    Target Task A: ISIC 2019 Skin Lesion Classification (8 classes).
    Loads images based on prespecified, deterministic stratified splits.
    """
    def __init__(self, splits_json, images_dir, split="train", transform=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.split = split

        with open(splits_json, "r") as f:
            data = json.load(f)

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.samples = data[split]
        self.num_classes = data.get("num_classes", 8)
        self.classes = data.get("classes", ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_id = item["image_id"]
        # Try both direct name and .jpg suffix
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / image_id
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(item["label"], dtype=torch.long)
        return img, label


class ImagenetteDataset(Dataset):
    """
    Auxiliary Task B: Imagenette 10-class natural images subset.
    Supports deterministic label-shuffled control (S1-shuf).
    """
    def __init__(self, splits_json, root_dir, split="train", shuffle_labels=False, seed=42, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.shuffle_labels = shuffle_labels

        with open(splits_json, "r") as f:
            data = json.load(f)

        if split not in ["train", "val"]:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

        self.samples = data[split]
        self.num_classes = data.get("num_classes", 10)

        # Store labels; optionally apply fixed label permutation
        raw_labels = np.array([s["label"] for s in self.samples], dtype=np.int64)
        if shuffle_labels:
            rng = np.random.RandomState(seed)
            self.labels = rng.permutation(raw_labels)
        else:
            self.labels = raw_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.root_dir / item["rel_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Imagenette image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class ISICProbeDataset(Dataset):
    """
    Deterministic 2,000-image diagnostic probe set for Effective Rank and CKA drift.
    """
    def __init__(self, probe_json, images_dir, transform=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transform = transform

        with open(probe_json, "r") as f:
            data = json.load(f)

        self.samples = data["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_id = item["image_id"]
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / image_id

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(item["label"], dtype=torch.long)
        return img, label


def get_applied_isic2019_loaders(splits_json, images_dir, batch_size=64, num_workers=4):
    """
    Builds train, validation, and held-out test loaders for ISIC 2019 at 224x224.
    Train augmentation: RandomResizedCrop(224, scale=(0.7, 1.0)), RandomHorizontalFlip.
    Evaluation: Resize((224, 224)), ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = ISIC2019Dataset(splits_json, images_dir, split="train", transform=train_transform)
    val_dataset = ISIC2019Dataset(splits_json, images_dir, split="val", transform=eval_transform)
    test_dataset = ISIC2019Dataset(splits_json, images_dir, split="test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_applied_imagenette_loader(splits_json, root_dir, batch_size=64, num_workers=4, shuffle_labels=False, seed=42):
    """
    Builds DataLoader for Auxiliary Task B (Imagenette) at 224x224.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = ImagenetteDataset(splits_json, root_dir, split="train",
                                shuffle_labels=shuffle_labels, seed=seed, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, 10


def get_applied_isic_probe_loader(probe_json, images_dir, batch_size=64, num_workers=4):
    """
    Builds DataLoader for the deterministic 2,000-image diagnostic probe set.
    """
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = ISICProbeDataset(probe_json, images_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader

