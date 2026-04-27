"""Sanity-check MD crops: render an N x N grid of images with their best box."""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

IMAGE_DIR = Path("data/eccv_18_all_images_sm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path,
                   default=Path("data/megadetector/_smoke.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/megadetector/smoke_grid.png"))
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.detections) as f:
        results = json.load(f)
    items = list(results.items())
    random.seed(0)
    random.shuffle(items)

    cols = 4
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    shown = 0
    for image_id, rec in items:
        if shown >= args.n:
            break
        path = IMAGE_DIR / rec["file_name"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for d in rec["detections"]:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            color = {"animal": "lime", "person": "red", "vehicle": "yellow"}.get(d["class_name"], "white")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 4, y1 + 4), f"{d['class_name']} {d['conf']:.2f}", fill=color)
        ax = axes[shown]
        ax.imshow(img)
        n_dets = len(rec["detections"])
        ax.set_title(f"{n_dets} det", fontsize=9)
        ax.axis("off")
        shown += 1

    for ax in axes[shown:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
