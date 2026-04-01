"""
inference.py
Simple inference utility for single-image predictions.

Usage:
    python inference.py --image path/to/image.jpg
"""
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
import model as model_module


def load_class_mapping(path=config.CLASS_MAPPING_FILE):
    try:
        with open(path, 'r') as f:
            m = json.load(f)
        # JSON is class->idx, invert it
        inv = {int(v): k for k, v in m.items()}
        class_names = [inv[i] for i in range(len(inv))]
        return class_names
    except Exception:
        return None


def make_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def predict(image_path, model_path=config.BEST_MODEL_PATH, topk=2, threshold=config.CONFIDENCE_THRESHOLD, device=config.DEVICE):
    class_names = load_class_mapping()
    if class_names is None:
        raise RuntimeError("Class mapping file not found. Train first.")

    transform = make_transform()
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0)

    model = model_module.get_mobilenet_v3(num_classes=len(class_names), dropout=config.DROPOUT, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        inp = inp.to(device)
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = probs.argsort()[::-1][:topk]
    results = [(class_names[int(i)], float(probs[int(i)])) for i in top_idx]
    if results[0][1] < threshold:
        uncertain = True
    else:
        uncertain = False

    return {
        'predictions': results,
        'uncertain': uncertain,
        'top1': results[0]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=config.CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    out = predict(args.image, model_path=args.model, topk=args.topk, threshold=args.threshold)
    print("Top predictions:")
    for cls, p in out['predictions']:
        print(f"{cls}: {p:.4f}")
    if out['uncertain']:
        print("Prediction uncertain (below threshold)")
