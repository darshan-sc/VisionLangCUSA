import json
import os
import random

CAPTIONS_FILE = "/datasets/flickr30k_raw/captions.txt"

SAVE_DIR = "/datasets/flickr30k_cusa"
IMAGES_SUBDIR = "flickr30k_images"  

os.makedirs(SAVE_DIR, exist_ok=True)

data = []
current_image = None

with open(CAPTIONS_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue


        if line.lower().endswith(".jpg"):
            current_image = line
            continue

    
        if current_image is None:
            continue

        caption = line
        image_name = current_image
        image_id = os.path.splitext(image_name)[0]

        data.append({
            "image_path": f"{IMAGES_SUBDIR}/{image_name}",
            "image_id": image_id,
            "caption": caption,
        })

print("Total image-caption pairs:", len(data))

random.seed(23)
random.shuffle(data)
N = len(data)
train = data[:int(0.8 * N)]
val   = data[int(0.8 * N):int(0.9 * N)]
test  = data[int(0.9 * N):]

for split_name, subset in [("train", train), ("val", val), ("test", test)]:
    out_path = os.path.join(SAVE_DIR, f"new_{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(subset, f)
    print(f"{split_name}: {len(subset)} entries → {out_path}")
