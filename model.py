"""
model.py
Builds MobileNetV3-based classifier and utilities to freeze/unfreeze layers.
"""
import torch
import torch.nn as nn
from torchvision import models


def get_mobilenet_v3(num_classes: int = 7, dropout: float = 0.5, pretrained: bool = True, freeze_backbone: bool = True):
    """
    Returns a MobileNetV3-Large model with a custom classifier head.

    Note: we DO NOT include Softmax in the model head because
    CrossEntropyLoss expects raw logits. Apply Softmax at inference time.
    """
    model = models.mobilenet_v3_large(pretrained=pretrained)

    # MobileNetV3 classifier first Linear layer input dim
    try:
        in_features = model.classifier[0].in_features
    except Exception:
        # fallback
        in_features = model.classifier[-1].in_features

    # replace classifier with: Linear -> ReLU -> Dropout -> Linear (logits)
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes),
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


def unfreeze_backbone(model: torch.nn.Module, fraction: float = 0.3):
    """
    Unfreeze the last `fraction` of the backbone modules to allow fine-tuning.
    fraction: float between 0 and 1 (e.g., 0.3 unfreezes last 30% of layers)
    """
    assert 0.0 <= fraction <= 1.0
    modules = list(model.features)
    n = len(modules)
    start_idx = max(0, n - int(n * fraction))
    for i, m in enumerate(modules):
        requires = i >= start_idx
        for p in m.parameters():
            p.requires_grad = requires
