"""
Detection-coverage analysis for the MegaDetector pass.

For each split and each species:
  - fraction of images where MD found at least one box (any class)
  - fraction where the top box is class 'animal'
  - mean confidence of the top box
  - distribution of relative box area (min vs full frame)

Also: confidence threshold sweep — fraction of CCT-labeled-animal images that
retain at least one detection at threshold t. Used to pick the operating point.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ANNOTATION_DIR = Path("data/eccv_18_annotation_files")
EMPTY_CAT_ID = 30

CLASS_IDS = [1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 33, 34, 51, 99]
CLASS_NAMES = ["opossum","raccoon","squirrel","bobcat","skunk","dog","coyote",
               "rabbit","bird","cat","badger","car","deer","fox","rodent"]
CAT_ID_TO_NAME = dict(zip(CLASS_IDS, CLASS_NAMES))

SPLITS = {
    "train":      "train_annotations.json",
    "cis_val":    "cis_val_annotations.json",
    "cis_test":   "cis_test_annotations.json",
    "trans_val":  "trans_val_annotations.json",
    "trans_test": "trans_test_annotations.json",
}


def first_animal_label(anns):
    animal = [a for a in anns if a["category_id"] != EMPTY_CAT_ID]
    if not animal:
        return None
    return CAT_ID_TO_NAME.get(animal[0]["category_id"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path,
                   default=Path("data/megadetector/detections.json"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/megadetector"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.detections}")
    with open(args.detections) as f:
        det = json.load(f)
    print(f"  {len(det):,} image_ids in detections file")

    rows = []           # (split, species, has_any, has_animal, top_conf, rel_area)
    per_split_counts = defaultdict(Counter)

    for split, fname in SPLITS.items():
        path = ANNOTATION_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        anns_by_img = defaultdict(list)
        for a in data["annotations"]:
            anns_by_img[a["image_id"]].append(a)

        for img in data["images"]:
            iid = img["id"]
            label = first_animal_label(anns_by_img.get(iid, []))
            if label is None:
                continue          # empty frame — exclude from coverage stats
            rec = det.get(iid)
            if rec is None:
                per_split_counts[split]["no_md_record"] += 1
                continue
            dets = rec.get("detections", [])
            w, h = rec.get("width"), rec.get("height")
            has_any = len(dets) > 0
            animal_dets = [d for d in dets if d["class_name"] == "animal"]
            top_conf = max((d["conf"] for d in dets), default=0.0)

            rel_area = 0.0
            if has_any and w and h:
                d0 = max(dets, key=lambda d: d["conf"])
                x1, y1, x2, y2 = d0["bbox_xyxy"]
                rel_area = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)

            rows.append({
                "split": split, "species": label,
                "has_any":    has_any,
                "has_animal": len(animal_dets) > 0,
                "top_conf":   top_conf,
                "rel_area":   rel_area,
            })
            per_split_counts[split]["total_animal_imgs"] += 1
            per_split_counts[split]["with_any_det"] += int(has_any)
            per_split_counts[split]["with_animal_det"] += int(len(animal_dets) > 0)

    # ── Per-split coverage table ─────────────────────────────────────────────
    print("\nPer-split coverage (animal-labeled images only):")
    print(f"  {'split':<11} {'n':>8} {'any':>10} {'animal':>10}")
    for split, c in per_split_counts.items():
        n = c["total_animal_imgs"]
        if n == 0:
            continue
        any_pct = c["with_any_det"] / n * 100
        ani_pct = c["with_animal_det"] / n * 100
        print(f"  {split:<11} {n:>8,} {any_pct:>9.1f}% {ani_pct:>9.1f}%")

    # ── Per-species coverage on trans_test ───────────────────────────────────
    print("\nPer-species coverage on trans_test (the hard split):")
    print(f"  {'species':<10} {'n':>6} {'any':>9} {'animal':>9} {'mean_conf':>10} {'mean_area':>10}")
    by_species = defaultdict(list)
    for r in rows:
        if r["split"] == "trans_test":
            by_species[r["species"]].append(r)
    species_rows = []
    for sp in CLASS_NAMES:
        items = by_species.get(sp, [])
        if not items:
            continue
        n = len(items)
        any_pct = sum(r["has_any"]    for r in items) / n * 100
        ani_pct = sum(r["has_animal"] for r in items) / n * 100
        mean_c  = float(np.mean([r["top_conf"] for r in items if r["has_any"]] or [0]))
        mean_a  = float(np.mean([r["rel_area"] for r in items if r["has_any"]] or [0]))
        species_rows.append((sp, n, any_pct, ani_pct, mean_c, mean_a))
        print(f"  {sp:<10} {n:>6,} {any_pct:>8.1f}% {ani_pct:>8.1f}% {mean_c:>10.3f} {mean_a:>10.3f}")

    # ── Confidence threshold sweep ──────────────────────────────────────────
    thresholds = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["train", "cis_test", "trans_test"]:
        confs = [r["top_conf"] for r in rows if r["split"] == split]
        if not confs:
            continue
        confs = np.array(confs)
        recall = [(confs >= t).mean() * 100 for t in thresholds]
        ax.plot(thresholds, recall, marker="o", label=f"{split} (n={len(confs):,})")
    ax.set_xlabel("Detection confidence threshold")
    ax.set_ylabel("% of CCT animal images with ≥1 box at threshold")
    ax.set_title("MegaDetector recall vs. threshold")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    out = args.out_dir / "threshold_sweep.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out}")

    # ── Per-species recall bar (trans_test) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    sps = [r[0] for r in species_rows]
    pcts = [r[2] for r in species_rows]
    ns = [r[1] for r in species_rows]
    bars = ax.bar(sps, pcts, color="steelblue", edgecolor="white")
    for bar, pct, n in zip(bars, pcts, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="80% target")
    ax.set_ylabel("% with ≥1 MD detection")
    ax.set_title("Per-species MD recall — trans_test")
    ax.set_ylim(0, 115)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    out = args.out_dir / "per_species_recall_trans.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")

    # ── Save raw rows ──────────────────────────────────────────────────────
    summary_path = args.out_dir / "coverage_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "per_split": {s: dict(c) for s, c in per_split_counts.items()},
            "per_species_trans_test": [
                {"species": sp, "n": n, "pct_any": a, "pct_animal": an,
                 "mean_top_conf": mc, "mean_rel_area": ma}
                for sp, n, a, an, mc, ma in species_rows
            ],
        }, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
