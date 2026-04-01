"""
dataset.py
Data loading utilities: safe ImageFolder wrappers, transforms, and dataloader creation.

Key features:
- Validates/skips corrupted images
- Strong train-time augmentation (dynamic)
- Consistent validation/test preprocessing
"""
import os
import json
import logging
import random
from collections import Counter

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, CLASS_MAPPING_FILE

logging.basicConfig(level=logging.INFO)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_valid_image(path: str) -> bool:
    """Try to open an image to detect corruption. Returns True if valid."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        logging.warning(f"Corrupted or unreadable image skipped: {path}")
        return False


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            return torch.clamp(tensor, 0.0, 1.0)
        return tensor


class RandomShadow(object):
    """Simple rectangular shadow overlay to simulate sunlight/shade."""

    def __init__(self, p=0.5, darkness=(0.4, 0.8)):
        self.p = p
        self.darkness = darkness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        x1 = random.randint(0, w // 2)
        x2 = random.randint(w // 2, w)
        y1 = random.randint(0, h // 2)
        y2 = random.randint(h // 2, h)
        alpha = random.uniform(*self.darkness)
        arr[y1:y2, x1:x2, :] = arr[y1:y2, x1:x2, :] * alpha
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def make_transforms(img_size=IMG_SIZE):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        RandomShadow(p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02, p=0.4),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, val_test_transforms


def _filter_samples(dataset: datasets.ImageFolder) -> None:
    """Remove corrupted images from an ImageFolder dataset in-place."""
    good = []
    for path, idx in dataset.samples:
        if is_valid_image(path):
            good.append((path, idx))
    dataset.samples = good
    dataset.targets = [s[1] for s in good]
    dataset.imgs = dataset.samples


def create_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Create PyTorch DataLoaders for train/val/test.

    Returns: train_loader, val_loader, test_loader, class_names, class_weights
    """
    train_t, val_t = make_transforms()

    train_ds = datasets.ImageFolder(train_dir, transform=train_t)
    val_ds = datasets.ImageFolder(val_dir, transform=val_t)
    test_ds = datasets.ImageFolder(test_dir, transform=val_t)

    _filter_samples(train_ds)
    _filter_samples(val_ds)
    _filter_samples(test_ds)

    class_names = train_ds.classes

    # compute class weights for imbalanced datasets
    counts = Counter(train_ds.targets)
    total = sum(counts.values())
    num_classes = len(class_names)
    class_weights = [0.0] * num_classes
    for i in range(num_classes):
        class_weights[i] = total / (num_classes * (counts.get(i, 1)))

    # save mapping
    mapping = {cls: idx for idx, cls in enumerate(class_names)}
    try:
        with open(CLASS_MAPPING_FILE, "w") as f:
            json.dump(mapping, f)
    except Exception:
        logging.warning("Could not write class mapping to disk.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, test_loader, class_names, torch.tensor(class_weights, dtype=torch.float)
