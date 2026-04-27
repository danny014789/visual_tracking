# Visual Tracking with Intel RealSense D405

Live object tracking with the Intel RealSense D405 depth camera. Combines RGB
detection with the aligned depth stream to report distance and image-plane
position of tracked objects.

## What's in here

| Script | What it does |
| --- | --- |
| `display.py` | Live color + colorized depth side-by-side. |
| `probe.py` | Lists every stream profile the D405 supports. |
| `track.py` | Click-and-drag a box on any object → CSRT tracker follows it; depth at the box center is shown. |
| `yolo_track.py` | YOLOv8 detection + BoT-SORT tracking. Each detection shows class, confidence, track id, distance, and pixel `(x, y)`. |
| `capture.py` | Capture training images of an object from the D405 (SPACE saves a frame). |
| `label.py` | Minimal OpenCV labeler (drag a box, save in YOLO format). One class: `fixture`. |
| `prepare_dataset.py` | Splits labeled images 80/20 into train/val and writes `dataset.yaml`. |
| `train.py` | Fine-tunes YOLOv8n on the labeled dataset. |

## Hardware

- Intel RealSense D405 (passive stereo, ~7 cm – 50 cm sweet spot).
- Tested at 640×480 @ 30 fps for both color and depth streams.

## Install

```bash
pip install pyrealsense2 opencv-contrib-python numpy ultralytics
```

`opencv-contrib-python` (not plain `opencv-python`) is required — the CSRT
tracker used by `track.py` lives in the contrib build.

## Quick start

### Just see the camera
```bash
python display.py
```

### Pick any object with the mouse and track its distance
```bash
python track.py
```
Drag a box → Enter. Press `s` to re-select, `q`/`Esc` to quit.

### Use pretrained YOLOv8 (80 COCO classes)
```bash
python yolo_track.py
```
Click a detection to lock the filter to that class. `c` clears the filter.

### Use the included custom 'fixture' model
```bash
python yolo_track.py --model weights/best.pt --conf 0.3
```

## Training a custom object detector

Pipeline for fine-tuning YOLOv8 on your own object (this repo's `weights/best.pt`
was made this way for a metal fixture, with 80 labeled images).

```bash
# 1. Capture training images from the D405
python capture.py            # SPACE to save → dataset/raw/img_NNNN.jpg

# 2. Label them (one class: 'fixture')
python label.py              # drag boxes; auto-saves YOLO .txt files

# 3. Split + write dataset.yaml
python prepare_dataset.py    # 80/20 train/val split

# 4. Fine-tune YOLOv8n
python train.py              # ~7 min on a modern CPU for 50 epochs / 80 imgs

# 5. Run live tracking with the new weights
python yolo_track.py --model runs/detect/fixture/weights/best.pt --conf 0.3
```

### Tips for good training data

- 80–150 images is a workable minimum.
- Vary pose, distance (10–50 cm for D405), background, and lighting.
- Tight bounding boxes around the object only.

## Known gotchas

- **Non-ASCII paths break ultralytics training.** If your project lives under
  a path containing non-ASCII characters (e.g. Chinese folder names), copy
  the `dataset/` directory to a clean path like `C:\yolo_data\` and edit
  `dataset/dataset.yaml`'s `path:` line accordingly.
- **D405 1280×720 maxes out at 5 fps.** All scripts use 640×480 @ 30 fps.
- **Distance is sampled from a small patch around the box center**, taking
  the median of valid (non-zero) depth pixels — robust to single bad pixels
  but assumes the center of the box actually lies on the object.

## Layout

```
.
├── display.py / probe.py / track.py / yolo_track.py
├── capture.py / label.py / prepare_dataset.py / train.py
├── weights/best.pt        # trained fixture model
├── yolov8n.pt             # baseline COCO pretrained weights
└── dataset/
    ├── images/{train,val}/
    ├── labels/{train,val}/
    ├── raw/               # capture.py writes here, label.py writes labels here
    └── dataset.yaml
```
