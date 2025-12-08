
# VisionLangCUSA

Original Paper - https://ojs.aaai.org/index.php/AAAI/article/view/29789

## 1. Install Environment

```bash
conda create -n cusa python=3.10 -y
conda activate cusa

pip install -r requirements.txt
```


---

## 2. Prepare Datasets

### Flickr30k

You can obtain the dataset JSON files from:

[https://drive.google.com/drive/folders/1z5xDebGm3XYrfjlHV6r_Kl3r5SF23db3](https://drive.google.com/drive/folders/1z5xDebGm3XYrfjlHV6r_Kl3r5SF23db3)

Alternatively, you can generate the JSONs from scratch using:

```
json dataset script/dataset_gen.py
```

Images and captions can be downloaded from the official Flickr30k source.

After preparing the dataset, you should have:

```
datasets/
└── flickr30k/
    ├── images/
    ├── flickr30k_train.json
    ├── flickr30k_val.json
    └── flickr30k_test.json
```

Update the dataset path in:

```
configs/vitb32/flickr/cusa.yaml
```

---

## 3. Generating UniCOM Feature Files (`unicom_*.npy`)

CUSA requires precomputed UniCOM visual features for train/val/test splits.
These must be generated **before** training or evaluation.

### 3.1 Clone the UniCOM repository

```bash
git clone https://github.com/deepglint/unicom
```

### 3.2 Add the feature extraction script

Place your `build_unicom_split.py` file inside:

```
unicom/unicom/build_unicom_split.py
```

You do **not** need to install the UniCOM repo. Running the script directly works.

### 3.3 Generate UniCOM feature files

Assuming the dataset structure:

```
datasets/flickr30k/
├── flickr30k_train.json
├── flickr30k_val.json
├── flickr30k_test.json
└── images/
```

Navigate into the UniCOM folder:

```bash
cd unicom
```

#### Train split

```bash
python unicom/build_unicom_split.py \
  --json ../datasets/flickr30k/flickr30k_train.json \
  --img_root ../datasets/flickr30k/images \
  --out ../datasets/flickr30k/flickr30k_train_unicom.npy
```

#### Validation split

```bash
python unicom/build_unicom_split.py \
  --json ../datasets/flickr30k/flickr30k_val.json \
  --img_root ../datasets/flickr30k/images \
  --out ../datasets/flickr30k/flickr30k_val_unicom.npy
```

#### Test split

```bash
python unicom/build_unicom_split.py \
  --json ../datasets/flickr30k/flickr30k_test.json \
  --img_root ../datasets/flickr30k/images \
  --out ../datasets/flickr30k/flickr30k_test_unicom.npy
```

### 3.4 Verify output

You should now have:

```
datasets/flickr30k/
├── flickr30k_train.json
├── flickr30k_val.json
├── flickr30k_test.json
├── images/
├── flickr30k_train_unicom.npy
├── flickr30k_val_unicom.npy
└── flickr30k_test_unicom.npy
```

---

## 4. Train CUSA

Example: Flickr30k, ViT-B/32 (multi-GPU):

```bash
torchrun --nproc_per_node=4 \
  main.py \
  --config ./configs/vitb32/flickr/cusa.yaml
```

Single GPU:

```bash
torchrun --nproc_per_node=1 \
  --master_port=25110 \
  main.py \
  --config ./configs/vitb32/flickr/cusa.yaml
```

Logs and checkpoints will be saved under:

```
checkpoints/flickr30k/vitb32/cusa/
```

---

## 5. Evaluate a Trained Model

Example command:

```bash
torchrun --nproc_per_node=1 \
  --master_port=25110 \
  retrieval.py \
  --config ./configs/vitb32/flickr/cusa.yaml \
  --eval \
  --resume \
  --checkpoint /path/to/checkpoint_best.pth
```

Evaluation prints:

- R@1, R@5, R@10 (text → image)
- R@1, R@5, R@10 (image → text)
- `r_mean` and `r_sum`

---

## 6. Text Encoder Improvement (Optional)

We replace the default MPNet encoder with **`thenlper/gte-large`** in two files:

### `retrieval.py`

```python
txt_enc_assisant = SentenceTransformer('thenlper/gte-large').to(device=device)
```

### `evaluation_sts.py`

```python
def get_mpnet_model(device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('thenlper/gte-large').to(device=device)
```

This improves soft-label quality and boosts retrieval performance.

---

File Description:

## Repository Structure

- `retrieval.py`  
  Main training + cross-modal retrieval script for CUSA. Loads configs, builds the UNIRE/CUSA model, sets up dataloaders, runs training, and runs image–text retrieval evaluation.

- `evaluation.py`  
  Computes image–text retrieval metrics (R@1/5/10, r_mean, etc.) for a trained model on Flickr30k/COCO-style datasets.

- `evaluation_eccv.py`  
  Script to evaluate the model on the ECCV Caption benchmark using COCO annotations and the ECCV caption metrics.

- `evaluation_img.py`  
  Image-only retrieval / metric-learning evaluation script (Rank-1 accuracy) on datasets like CUB, SOP, InShop, and iNaturalist.

- `evaluation_sts.py`  
  SentEval-based script for evaluating sentence representations (STS and transfer tasks). Uses the same text encoder as CUSA for downstream sentence embedding benchmarks.

- `utils.py`  
  Utility functions for distributed training (init, rank/world size helpers), metric logging, and smoothed statistics used during training and evaluation.

- `requirements.txt`  
  Python dependencies (PyTorch, torchvision, transformers, sentence-transformers, SentEval, etc.) needed to run training and evaluation.

- `configs/`  
  YAML configuration files defining experiment settings: dataset paths, batch sizes, optimizer and scheduler settings, and model/backbone choices (e.g., `vitb32/flickr/cusa.yaml`).

- `dataset/`  
  Dataset utilities for cross-modal retrieval (e.g., Flickr30k/COCO). Provides `get_dataset`, `create_sampler`, and `create_loader` used by `retrieval.py`.

- `dataset_evalimg/`  
  Dataset wrappers for image-only retrieval experiments (CUBirds, Cars, SOP, InShop, iNaturalist). Used by `evaluation_img.py` for Rank-1 evaluations.

- `dataset_example/`  
  Example / placeholder dataset directory (not required for the core Flickr30k CUSA reproduction, but can be used as a template for custom datasets).

- `json dataset script/`  
  Contains `dataset_gen.py`, which builds Flickr30k-style JSON files (train/val/test splits) from a raw `captions.txt` file and image folder.

- `clip/`  
  Local CLIP implementation/wrapper used by some evaluation scripts (e.g., ECCV caption and text-feature baselines).

- `optim/`  
  Optimizer utilities (e.g., AdamW / SGD factory functions). `retrieval.py` calls `create_optimizer` from here based on the selected config.

- `scheduler/`  
  Learning-rate scheduler utilities (warmup, cosine, etc.). `retrieval.py` calls `create_scheduler` from here using parameters in the YAML config.

- `unicom script/`  
  Contains `build_unicom_split.py`, a helper script for generating UniCOM image feature `.npy` files (`*_unicom.npy`) from Flickr30k JSONs and raw images.

- `unire/`  
  Core UNIRE/CUSA model implementation (vision–language backbone, projection heads, contrastive and soft-label objectives). This is the main model used during training and retrieval evaluation.


