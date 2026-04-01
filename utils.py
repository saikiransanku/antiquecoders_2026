"""
utils.py
Helper utilities: metrics, plotting, saving/loading models.
"""
import os
import json
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import PLOTS_DIR, HISTORY_PATH, BEST_MODEL_PATH


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(state, path=BEST_MODEL_PATH):
    torch.save(state, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def topk_accuracy(output, target, k=1):
    with torch.no_grad():
        maxk = max((1, k))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().item())
        return res / output.size(0)


def compute_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return report, cm, acc


def plot_history(history: dict, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("train_acc", []), label="train_acc")
    plt.plot(epochs, history.get("val_acc", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = os.path.join(out_dir, f"training_plots_{int(time.time())}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path
