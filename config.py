"""
Configuration and hyperparameters for the wheat disease pipeline.
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = min(8, (os.cpu_count() or 4))
PIN_MEMORY = True

# Training phases
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 20
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-6
DROPOUT = 0.5

EARLY_STOPPING_PATIENCE = 7
MIXUP_ALPHA = 0.0
LABEL_SMOOTHING = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CONFIDENCE_THRESHOLD = 0.6

CLASS_MAPPING_FILE = os.path.join(MODEL_DIR, "class_to_idx.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
TORCHSCRIPT_PATH = os.path.join(MODEL_DIR, "model_script.pt")
HISTORY_PATH = os.path.join(LOG_DIR, "training_history.json")
PLOTS_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
