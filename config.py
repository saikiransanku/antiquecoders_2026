"""
Configuration and hyperparameters for the wheat disease pipeline.
"""
import os
import torch
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_processed_splits(base_dir: str):
	"""
	Look for a `training/data/processed` folder and try to locate sensible
	`train`, `val`, `test` (or `bal`) split directories inside it. This
	supports nested layouts such as `train/train` or `test/test`.
	Returns a dict mapping split name -> path if found, otherwise None.
	"""
	processed_root = os.path.join(base_dir, "training", "data", "processed")
	if not os.path.isdir(processed_root):
		return None

	splits = {}
	target_names = {"train", "val", "test", "bal"}

	def _is_valid_split(path: str) -> bool:
		# Valid if contains at least one class subdirectory that contains files
		try:
			for entry in os.listdir(path):
				epath = os.path.join(path, entry)
				if os.path.isdir(epath):
					# check for files inside class folder
					for f in os.listdir(epath):
						if os.path.isfile(os.path.join(epath, f)):
							return True
		except Exception:
			return False
		return False

	for root, dirs, files in os.walk(processed_root):
		name = os.path.basename(root).lower()
		if name in target_names and _is_valid_split(root):
			splits[name] = root
		# stop early if we found the main splits
		if "train" in splits and "val" in splits and "test" in splits:
			break

	return splits if splits else None


# Prefer any processed dataset under training/data/processed if available
_splits = _find_processed_splits(BASE_DIR)
if _splits:
	logging.info(f"Using processed dataset under training/data/processed; found splits: {_splits}")
	DATA_DIR = os.path.join(BASE_DIR, "training", "data", "processed")
	# priority: explicit 'train' -> 'bal' fallback
	TRAIN_DIR = _splits.get("train") or _splits.get("bal") or os.path.join(DATA_DIR, "train")
	VAL_DIR = _splits.get("val") or os.path.join(DATA_DIR, "val")
	TEST_DIR = _splits.get("test") or os.path.join(DATA_DIR, "test")
else:
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
