import os
import time
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import config
from dataset import create_dataloaders
# Import the local `model.py` safely. If a different installed package named
# `model` shadows the local file, fall back to loading by file path so the
# expected `get_mobilenet_v3` and `unfreeze_backbone` are available.
try:
    import model as model_module
    if not hasattr(model_module, "get_mobilenet_v3"):
        raise ImportError("Imported 'model' module does not expose 'get_mobilenet_v3'")
except Exception:
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("local_model", os.path.join(os.path.dirname(__file__), "model.py"))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

import utils


def make_plateau_scheduler(optimizer, mode='min', patience=3, factor=0.5):
    """Instantiate ReduceLROnPlateau compatibly across PyTorch versions.

    Some older/newer PyTorch versions differ in whether they accept a
    `verbose` keyword. Try with `verbose=True` first, fall back otherwise.
    """
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor, verbose=True)
    except TypeError:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, clip_grad=1.0, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="val", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def main():
    set_seed(config.SEED)

    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders(batch_size=config.BATCH_SIZE)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Phase 1: build model with backbone frozen
    model = model_module.get_mobilenet_v3(num_classes=num_classes, dropout=config.DROPOUT, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # criterion with optional class weights and label smoothing
    weight = class_weights.to(device) if class_weights is not None else None
    try:
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.LABEL_SMOOTHING)
    except TypeError:
        # older torch doesn't have label_smoothing arg
        criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR_PHASE1, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_plateau_scheduler(optimizer, mode='min', patience=3, factor=0.5)
    scaler = GradScaler()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float('inf')
    early_stop_counter = 0

    # Phase 1 training (classifier only)
    for epoch in range(1, config.EPOCHS_PHASE1 + 1):
        print(f"Epoch Phase1 {epoch}/{config.EPOCHS_PHASE1}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, clip_grad=1.0, mixup_alpha=config.MIXUP_ALPHA)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            utils.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_names': class_names,
            }, path=config.BEST_MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered during phase 1")
                break

    # Phase 2: unfreeze part of the backbone and fine-tune
    print("Unfreezing last 30% of backbone for fine-tuning")
    model = model_module.get_mobilenet_v3(num_classes=num_classes, dropout=config.DROPOUT, pretrained=True, freeze_backbone=True)
    # load the classifier weights we trained in phase 1 if available
    try:
        ckpt = utils.load_checkpoint(config.BEST_MODEL_PATH)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    except Exception:
        pass

    model = model.to(device)
    model_module.unfreeze_backbone(model, fraction=0.3)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR_PHASE2, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_plateau_scheduler(optimizer, mode='min', patience=4, factor=0.5)

    best_val_loss_phase2 = best_val_loss
    early_stop_counter = 0

    for epoch in range(1, config.EPOCHS_PHASE2 + 1):
        print(f"Epoch Phase2 {epoch}/{config.EPOCHS_PHASE2}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, clip_grad=1.0, mixup_alpha=config.MIXUP_ALPHA)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            early_stop_counter = 0
            utils.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_names': class_names,
            }, path=config.BEST_MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered during phase 2")
                break

    # Save history and plots
    try:
        with open(config.HISTORY_PATH, 'w') as f:
            json.dump(history, f)
    except Exception:
        pass

    plot_path = utils.plot_history(history)
    print(f"Saved training plots to {plot_path}")

    # Final evaluation on test set
    print("Loading best model for final evaluation...")
    ckpt = utils.load_checkpoint(config.BEST_MODEL_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # run a full evaluation: collect preds + metrics
    all_preds = []
    all_targets = []
    all_probs = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="test"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs).cpu().numpy()
            preds = probs.argmax(axis=1).tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())
            all_probs.extend(probs.tolist())

    report, cm, acc = utils.compute_metrics(all_targets, all_preds, class_names)
    print("Test accuracy:", acc)
    print("Classification report summary (per class):")
    for cls in class_names:
        print(cls, report.get(cls, {}))

    # Save final torchscript model for deployment
    try:
        example_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
        scripted = torch.jit.trace(model.to(device).eval(), example_input)
        scripted.save(config.TORCHSCRIPT_PATH)
        print(f"Saved TorchScript model to {config.TORCHSCRIPT_PATH}")
    except Exception as e:
        print("Could not export TorchScript model:", e)

    print("Training complete.")


if __name__ == '__main__':
    main()
