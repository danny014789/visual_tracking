"""
Split labeled images in dataset/raw/ into dataset/{images,labels}/{train,val}
and write dataset/dataset.yaml.

Expectation: each image img_NNNN.jpg in dataset/raw/ has a matching
img_NNNN.txt label file alongside it (YOLO format, written by labelImg
or any YOLO-format exporter).

Usage:
    python prepare_dataset.py                # default 80/20 split
    python prepare_dataset.py --val 0.15     # custom val ratio
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "dataset"
RAW = DATA / "raw"
CLASS_NAME = "fixture"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=float, default=0.2,
                   help="fraction of images for validation (default 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of moving them (keeps raw/ intact)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    images = sorted(RAW.glob("img_*.jpg"))
    pairs = []
    skipped = 0
    for img in images:
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            skipped += 1

    print(f"Found {len(images)} images, {len(pairs)} labeled, {skipped} unlabeled (skipped)")
    if not pairs:
        raise SystemExit("No labeled image pairs found. Label the images first "
                         "(YOLO format .txt next to each .jpg in dataset/raw/).")

    random.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val)))
    val = pairs[:n_val]
    train = pairs[n_val:]
    print(f"Split: train={len(train)}, val={len(val)}")

    transfer = shutil.copy2 if args.copy else shutil.move

    for split, items in (("train", train), ("val", val)):
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            transfer(str(img), str(img_dir / img.name))
            transfer(str(lbl), str(lbl_dir / lbl.name))

    yaml_path = DATA / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DATA.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
