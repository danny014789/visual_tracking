"""
Minimal OpenCV labeler for the single-class 'fixture' dataset.

Walks through every img_*.jpg in dataset/raw/, lets you draw bounding
boxes with the mouse, and writes YOLO-format .txt files alongside.

Mouse:
  left-drag  draw a new box (release to commit)
  right-click delete the box under the cursor

Keys (focus the OpenCV window):
  n / d / SPACE / Right    next image
  p / a / Left             previous image
  u                        undo last box on current image
  c                        clear all boxes on current image
  s                        skip this image (no label file written)
  q / Esc                  quit (auto-saves on every navigation)

YOLO format written: one line per box -> "0 cx cy w h"
where coords are normalized to [0, 1].
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "dataset" / "raw"
CLASS_ID = 0
CLASS_NAME = "fixture"
WIN = "label - dataset/raw"


def load_yolo(label_path, w, h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))
        boxes.append((x1, y1, x2, y2))
    return boxes


def save_yolo(label_path, boxes, w, h):
    if not boxes:
        # An empty .txt is a valid YOLO label meaning "no objects in image"
        label_path.write_text("")
        return
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines))


def imread_unicode(path):
    # cv2.imread fails on non-ASCII Windows paths; decode bytes ourselves.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    images = sorted(RAW.glob("img_*.jpg"))
    if not images:
        print(f"No images in {RAW}")
        sys.exit(1)

    state = {
        "i": 0,
        "boxes": [],         # list of (x1,y1,x2,y2) in pixel coords
        "drawing": False,
        "start": None,
        "current": None,     # in-progress (x1,y1,x2,y2) while dragging
        "img": None,
        "img_h": 0,
        "img_w": 0,
    }

    def label_path_for(idx):
        return images[idx].with_suffix(".txt")

    def load_image(idx):
        img = imread_unicode(images[idx])
        if img is None:
            print(f"Failed to read {images[idx]}")
            return False
        h, w = img.shape[:2]
        state["img"] = img
        state["img_h"] = h
        state["img_w"] = w
        state["boxes"] = load_yolo(label_path_for(idx), w, h)
        state["current"] = None
        state["drawing"] = False
        return True

    def save_current():
        save_yolo(label_path_for(state["i"]),
                  state["boxes"], state["img_w"], state["img_h"])

    def on_mouse(event, x, y, flags, _):
        x = max(0, min(state["img_w"] - 1, x))
        y = max(0, min(state["img_h"] - 1, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            sx, sy = state["start"]
            state["current"] = (sx, sy, x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            sx, sy = state["start"]
            x1, x2 = sorted((sx, x))
            y1, y2 = sorted((sy, y))
            state["drawing"] = False
            state["current"] = None
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                state["boxes"].append((x1, y1, x2, y2))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # delete the topmost box containing (x, y)
            for k in range(len(state["boxes"]) - 1, -1, -1):
                bx1, by1, bx2, by2 = state["boxes"][k]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    del state["boxes"][k]
                    break

    if not load_image(0):
        sys.exit(1)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    print(f"Loaded {len(images)} images from {RAW}")
    print("n/d/Right=next  p/a/Left=prev  u=undo  c=clear  s=skip  q/Esc=quit")

    while True:
        display = state["img"].copy()
        for x1, y1, x2, y2 in state["boxes"]:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, CLASS_NAME, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if state["current"] is not None:
            x1, y1, x2, y2 = state["current"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 1)

        labeled_count = sum(1 for p in images if p.with_suffix(".txt").exists())
        hud = (f"[{state['i']+1}/{len(images)}]  boxes:{len(state['boxes'])}  "
               f"labeled total:{labeled_count}/{len(images)}  {images[state['i']].name}")
        cv2.rectangle(display, (0, 0), (state["img_w"], 22), (0, 0, 0), -1)
        cv2.putText(display, hud, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(WIN, display)
        k = cv2.waitKeyEx(20)  # waitKeyEx returns extended codes (arrow keys)

        if k == -1:
            continue
        kl = k & 0xFF

        # Navigation
        if kl in (ord("n"), ord("d"), ord(" ")) or k == 0x270000:  # right arrow
            save_current()
            if state["i"] < len(images) - 1:
                state["i"] += 1
                load_image(state["i"])
        elif kl in (ord("p"), ord("a")) or k == 0x250000:  # left arrow
            save_current()
            if state["i"] > 0:
                state["i"] -= 1
                load_image(state["i"])
        elif kl == ord("u"):
            if state["boxes"]:
                state["boxes"].pop()
        elif kl == ord("c"):
            state["boxes"].clear()
        elif kl == ord("s"):
            # skip without writing a label file
            if state["i"] < len(images) - 1:
                state["i"] += 1
                load_image(state["i"])
        elif kl in (ord("q"), 27):
            save_current()
            break

    cv2.destroyAllWindows()
    labeled = sum(1 for p in images if p.with_suffix(".txt").exists())
    print(f"Done. {labeled}/{len(images)} images have label files.")


if __name__ == "__main__":
    main()
