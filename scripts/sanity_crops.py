"""Standalone sanity-check: load CCT splits with bboxes, verify counts, render crop grid.
Mirrors the data-loading + sample-grid sections of 03_species_classifier_cropped.ipynb.
"""
import json, random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ANNOTATION_DIR = Path("data/eccv_18_annotation_files")
IMAGE_DIR      = Path("data/eccv_18_all_images_sm")
OUT_DIR        = Path("results/species_cropped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMPTY_CAT_ID = 30
DROP_CAT_IDS = {21, 34, 51}    # badger, deer, fox

ALL_CLASS_IDS = [1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 33, 34, 51, 99]
ALL_CLASS_NAMES = ["opossum","raccoon","squirrel","bobcat","skunk","dog","coyote",
                   "rabbit","bird","cat","badger","car","deer","fox","rodent"]
CLASS_IDS   = [c for c in ALL_CLASS_IDS if c not in DROP_CAT_IDS]
CLASS_NAMES = [n for c, n in zip(ALL_CLASS_IDS, ALL_CLASS_NAMES) if c not in DROP_CAT_IDS]
CAT_ID_TO_IDX = {c: i for i, c in enumerate(CLASS_IDS)}


def load_split(json_path):
    with open(json_path) as f:
        data = json.load(f)
    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    samples = []
    skip_empty = skip_drop = skip_unk = skip_nb = 0
    for ann in data["annotations"]:
        c = ann["category_id"]
        if c == EMPTY_CAT_ID: skip_empty += 1; continue
        if c in DROP_CAT_IDS: skip_drop += 1; continue
        if c not in CAT_ID_TO_IDX: skip_unk += 1; continue
        b = ann.get("bbox")
        if not b or len(b) != 4 or b[2] <= 1 or b[3] <= 1: skip_nb += 1; continue
        f_name = img_id_to_file.get(ann["image_id"])
        if f_name is None: continue
        samples.append((f_name, list(b), CAT_ID_TO_IDX[c]))
    counts = Counter(idx for _, _, idx in samples)
    print(f"  {json_path.name}: {len(samples):,} crops "
          f"(empty={skip_empty:,}, dropped={skip_drop:,}, no_bbox={skip_nb:,})")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {i:2d}  {n:<10}: {counts.get(i, 0):>5,}")
    return samples


def render_grid(samples, out_path, n=16, cols=4, pad_frac=0.10):
    rng = random.Random(0)
    picks = rng.sample(samples, min(n, len(samples)))
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax, (fname, bbox, label) in zip(axes, picks):
        path = IMAGE_DIR / fname
        img = Image.open(path).convert("RGB")
        W, H = img.size
        x, y, w, h = bbox
        px, py = w * pad_frac, h * pad_frac
        x1 = max(0, int(round(x - px))); y1 = max(0, int(round(y - py)))
        x2 = min(W, int(round(x + w + px))); y2 = min(H, int(round(y + h + py)))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, W, H
        crop = img.crop((x1, y1, x2, y2))
        ax.imshow(crop)
        ax.set_title(CLASS_NAMES[label], fontsize=10)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out_path}")


print("Loading splits ...")
train     = load_split(ANNOTATION_DIR / "train_annotations.json")
val       = load_split(ANNOTATION_DIR / "cis_val_annotations.json")
cis_test  = load_split(ANNOTATION_DIR / "cis_test_annotations.json")
trans_test= load_split(ANNOTATION_DIR / "trans_test_annotations.json")

print(f"\nTotals — train: {len(train):,} | val: {len(val):,} | "
      f"cis_test: {len(cis_test):,} | trans_test: {len(trans_test):,}")

render_grid(train, OUT_DIR / "sample_grid_cropped.png")
