import argparse
import json
import os

from PIL import Image
import numpy as np
import torch

import unicom  

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def build_unicom_npy(json_path: str, out_npy: str, img_root: str = ""):
    with open(json_path, "r") as f:
        items = json.load(f)

    print("Loading UNICOM ViT-B/32...")
    model, preprocess = unicom.load("ViT-B/32")
    model = model.to(device)
    model.eval()

    feat_dict = {}
    total = len(items)
    print(f"Processing {total} items from {json_path}")

    for i, obj in enumerate(items):
        img_path = obj["image_path"]
        if img_root and not os.path.isabs(img_path):
            img_path = os.path.join(img_root, img_path)
        img_id = obj.get("image_id", os.path.basename(img_path))

        image = Image.open(img_path).convert("RGB")
        x = preprocess(image).unsqueeze(0).to(device)

        feat = model(x)

        # Handle [1, D] vs [1, N, D]
        if feat.ndim == 2:
            feat = feat[0]
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)[0]

        feat_dict[img_id] = feat.cpu().numpy().astype("float32")

        if (i + 1) % 500 == 0:
            print(f"{i+1}/{total} done")

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, feat_dict)
    print(f"Saved {len(feat_dict)} entries → {out_npy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to train/val/test json for Flickr30k")
    parser.add_argument("--out", required=True, help="Output .npy path (e.g. train_unicom.npy)")
    parser.add_argument("--img_root", default="", help="Optional root if json has relative image paths")
    args = parser.parse_args()

    build_unicom_npy(args.json, args.out, args.img_root)


if __name__ == "__main__":
    main()
