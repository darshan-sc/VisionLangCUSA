import json
import os

# ----------------------------------------------------
# Target prefix you want in the final JSON
# ----------------------------------------------------
NEW_PREFIX = "flickr30k_images"

# Your existing correct split files
INPUT_FILES = {
    "train": "new_train.json",
    "val": "new_val.json",
    "test": "new_test.json",
}

# Where to save updated JSONs
OUT_DIR = "./fixed_splits"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Path Rewriting Logic
# ----------------------------------------------------
for split, in_file in INPUT_FILES.items():
    print(f"Processing {split} split...")

    with open(in_file, "r") as f:
        data = json.load(f)

    for item in data:
        old_path = item["image_path"]

        # Extract just the filename (e.g., 41105465.jpg)
        filename = os.path.basename(old_path)

        # Build the new relative path
        item["image_path"] = f"{NEW_PREFIX}/{filename}"

    out_path = os.path.join(OUT_DIR, f"new_{split}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{split}: {len(data)} entries → {out_path}")

print("✅ All image paths converted to relative format.")
