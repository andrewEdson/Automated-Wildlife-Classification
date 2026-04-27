"""
Run MegaDetector v5a over the CCT20 image set.

Output: data/megadetector/detections.json with schema
    {
      "<image_id>": {
        "file_name": "<path relative to image_dir>",
        "width":  <int>,
        "height": <int>,
        "detections": [
          {"bbox_xyxy": [x1,y1,x2,y2], "conf": 0.97, "class_id": 0, "class_name": "animal"},
          ...
        ]
      },
      ...
    }

Bounding boxes are in *absolute pixel* coordinates of the original image (xyxy).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ANNOTATION_DIR = Path("data/eccv_18_annotation_files")
IMAGE_DIR      = Path("data/eccv_18_all_images_sm")
OUTPUT_PATH    = Path("data/megadetector/detections.json")

ANNOTATION_FILES = [
    "train_annotations.json",
    "cis_val_annotations.json",
    "cis_test_annotations.json",
    "trans_val_annotations.json",
    "trans_test_annotations.json",
]


def build_image_index(annotation_dir: Path) -> dict:
    """Return image_id -> {'file_name': str} merged across all splits."""
    index = {}
    for fname in ANNOTATION_FILES:
        path = annotation_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} (not found)")
            continue
        with open(path) as f:
            data = json.load(f)
        n_new = 0
        for img in data["images"]:
            if img["id"] not in index:
                index[img["id"]] = {"file_name": img["file_name"]}
                n_new += 1
        print(f"  {fname}: {len(data['images']):,} images ({n_new:,} new)")
    return index


def load_megadetector(device: str):
    """Load MDv5a. Loads on CPU (weights have float64 buffers MPS rejects),
    then casts to float32 and moves to the target device."""
    from PytorchWildlife.models import detection as pw_detection

    print(f"Loading MegaDetector v5a (target device: {device})...")
    model = pw_detection.MegaDetectorV5(device="cpu", pretrained=True, version="a")
    if device != "cpu":
        model.model = model.model.float().to(device)
        model.device = device
    return model


def detect_image(model, img_path: Path, conf_thres: float):
    """Run MD on one image. Returns (width, height, list[detection_dict])."""
    pil = Image.open(img_path).convert("RGB")
    w, h = pil.size
    img_np = np.array(pil)

    out = model.single_image_detection(
        img_np, img_path=str(img_path), det_conf_thres=conf_thres
    )

    dets = out["detections"]
    boxes = dets.xyxy            # (N,4) float32, absolute pixels
    confs = dets.confidence      # (N,) float32
    classes = dets.class_id      # (N,) int64

    records = []
    for box, conf, cls in zip(boxes, confs, classes):
        records.append({
            "bbox_xyxy": [float(x) for x in box],
            "conf":      float(conf),
            "class_id":  int(cls),
            "class_name": model.CLASS_NAMES[int(cls)],
        })
    return w, h, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N images (smoke test).")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Detection confidence threshold (keep MD high-recall).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: mps|cuda|cpu. Auto-detected if omitted.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output JSON path.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip image_ids already present in the output file.")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print("Building image index from annotations...")
    index = build_image_index(ANNOTATION_DIR)
    print(f"Total unique images: {len(index):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.resume and args.output.exists():
        with open(args.output) as f:
            existing = json.load(f)
        print(f"Resume: {len(existing):,} image_ids already done — skipping.")

    image_ids = list(index.keys())
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
        print(f"--limit set: processing first {len(image_ids):,} images only.")

    todo = [iid for iid in image_ids if iid not in existing]
    print(f"To process: {len(todo):,} images at conf>={args.conf}")

    model = load_megadetector(device)
    results = dict(existing)

    save_every = 500
    n_missing = 0
    n_zero_det = 0
    t0 = time.time()

    pbar = tqdm(todo, desc="MD inference")
    for i, image_id in enumerate(pbar):
        file_name = index[image_id]["file_name"]
        img_path = IMAGE_DIR / file_name
        if not img_path.exists():
            n_missing += 1
            results[image_id] = {
                "file_name": file_name, "width": None, "height": None,
                "detections": [], "missing": True,
            }
            continue
        try:
            w, h, dets = detect_image(model, img_path, args.conf)
        except Exception as e:
            n_missing += 1
            results[image_id] = {
                "file_name": file_name, "width": None, "height": None,
                "detections": [], "error": str(e),
            }
            continue

        if not dets:
            n_zero_det += 1
        results[image_id] = {
            "file_name":  file_name,
            "width":      w,
            "height":     h,
            "detections": dets,
        }

        if (i + 1) % save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate
            pbar.set_postfix({
                "rate": f"{rate:.2f} im/s",
                "eta":  f"{eta/60:.1f} min",
                "missing": n_missing,
                "zero_det": n_zero_det,
            })

    with open(args.output, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone. {len(todo):,} images in {elapsed/60:.1f} min "
          f"({len(todo)/max(elapsed,1):.2f} im/s)")
    print(f"  missing/error : {n_missing:,}")
    print(f"  zero detect.  : {n_zero_det:,}")
    print(f"  output        : {args.output}")


if __name__ == "__main__":
    main()
