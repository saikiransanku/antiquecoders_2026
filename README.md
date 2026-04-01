# Wheat Disease Detection (PyTorch)

This repository implements a robust PyTorch image-classification pipeline for wheat plant disease detection using MobileNetV3-Large (ImageNet pretrained) and strong train-time augmentation. The code is organized, modular and built with deployment in mind (TorchScript export).

Quick start

1. Install dependencies (recommended: virtualenv)

```bash
pip install -r requirements.txt
```

2. Prepare dataset with structure:

```
dataset/
  train/
    class_a/
    class_b/
  val/
  test/
```

3. Train

```bash
python train.py
```

4. Inference

```bash
python inference.py --image path/to/image.jpg
```

Files
- `dataset.py`: data loaders and transforms
- `model.py`: MobileNetV3 model builder
- `train.py`: training loop and orchestration
- `utils.py`: metrics, plotting and helpers
- `inference.py`: single-image inference script
- `config.py`: hyperparameters and paths

Notes
- The model saves `models/best_model.pth` and attempts to export a TorchScript file for deployment.
- Validation/test transforms do not apply augmentations.
# antiquecoders_2026
24hours Hackthon 
