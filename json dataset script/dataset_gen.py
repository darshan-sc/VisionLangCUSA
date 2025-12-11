import json
import os

NEW_PREFIX = "flickr30k_images"


INPUT_FILES = {
    "train": "new_train.json",
    "val": "new_val.json",
    "test": "new_test.json",
}

#save loc
OUT_DIR = "./fixed_splits"
os.makedirs(OUT_DIR, exist_ok=True)


for split, in_file in INPUT_FILES.items():
    print(f"Processing {split} split...")

    with open(in_file, "r") as f:
        data = json.load(f)

    for item in data:
        old_path = item["image_path"]

        filename = os.path.basename(old_path)

        item["image_path"] = f"{NEW_PREFIX}/{filename}"

    out_path = os.path.join(OUT_DIR, f"new_{split}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{split}: {len(data)} entries → {out_path}")

print("All image paths converted to relative format.")
