"""
Fine-tune YOLOv8n on the metal fixture dataset.

Run this AFTER:
  1. python capture.py          (collect images)
  2. label them (YOLO .txt files alongside .jpg in dataset/raw/)
  3. python prepare_dataset.py  (split + write dataset.yaml)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
# Ultralytics chokes on the non-ASCII '桌面' in this project's path, so the
# dataset and training run live under a clean ASCII path on C:\.
CLEAN_ROOT = Path("C:/yolo_data_fixture")
YAML = CLEAN_ROOT / "dataset.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolov8n.pt",
                   help="pretrained weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8,
                   help="-1 for auto-batch (GPU only); start small on CPU")
    p.add_argument("--device", default=None,
                   help="'cpu', '0' for first GPU, etc. Default: ultralytics auto.")
    return p.parse_args()


def main():
    args = parse_args()
    if not YAML.exists():
        raise SystemExit(f"{YAML} not found. Run prepare_dataset.py first.")

    runs_dir = CLEAN_ROOT / "runs" / "detect"
    model = YOLO(args.base)
    model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name="fixture",
        exist_ok=True,
    )
    best = runs_dir / "fixture" / "weights" / "best.pt"
    print("\nDone. Best weights:")
    print(f"  {best}")
    print("\nRun inference with:")
    print(f'  python yolo_track.py --model "{best}"')


if __name__ == "__main__":
    main()
