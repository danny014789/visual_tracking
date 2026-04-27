"""
YOLOv8 detection + tracking on the Intel RealSense D405 color stream,
with distance read from the aligned depth frame.

Controls (focus the OpenCV window):
  q / Esc  quit
  f        toggle 'filter to single class' on the most-recently-clicked class
  c        clear class filter (show all detections again)

Mouse:
  left-click on a detection to lock the filter to that class.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

WIDTH, HEIGHT, FPS = 640, 480, 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 weights (auto-downloads on first run)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--filter", default=None,
                   help="comma-separated COCO class names to keep, e.g. 'cup,bottle'")
    return p.parse_args()


def sample_distance(depth_image, cx, cy, depth_scale, half=4):
    h, w = depth_image.shape
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def color_for_id(track_id):
    # Stable pseudo-random color per track id
    rng = np.random.default_rng(int(track_id) if track_id is not None else 0)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))


def main():
    args = parse_args()

    model = YOLO(args.model)
    names = model.names  # {id: name}
    name_to_id = {v: k for k, v in names.items()}

    filter_ids = None
    if args.filter:
        wanted = [s.strip() for s in args.filter.split(",")]
        missing = [w for w in wanted if w not in name_to_id]
        if missing:
            raise SystemExit(f"Unknown class names: {missing}\n"
                             f"Valid examples: {list(names.values())[:10]} ...")
        filter_ids = {name_to_id[w] for w in wanted}
        print(f"Filtering to: {wanted}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    win = "D405 + YOLOv8"
    cv2.namedWindow(win)

    # Mouse click sets a one-class filter on whichever box you click into.
    last_boxes = []  # list of (xyxy, cls_id) for current frame, used by mouse cb

    def on_mouse(event, x, y, flags, _userdata):
        nonlocal filter_ids
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), cid in last_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                filter_ids = {cid}
                print(f"Filter -> {names[cid]}")
                return

    cv2.setMouseCallback(win, on_mouse)

    print("Controls: q/Esc=quit  f=lock to clicked class  c=clear filter")

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())

            results = model.track(
                color, persist=True, verbose=False,
                conf=args.conf, tracker="bytetrack.yaml",
            )
            r = results[0]

            display = color.copy()
            last_boxes = []

            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                cls = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                ids = (r.boxes.id.cpu().numpy().astype(int)
                       if r.boxes.id is not None else [None] * len(cls))

                for (x1, y1, x2, y2), cid, cnf, tid in zip(xyxy, cls, confs, ids):
                    if filter_ids is not None and cid not in filter_ids:
                        continue
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    dist = sample_distance(depth, cx, cy, depth_scale)

                    color_box = color_for_id(tid if tid is not None else cid)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                    cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

                    head = f"{names[cid]} {cnf:.2f}"
                    if tid is not None:
                        head = f"#{tid} " + head
                    dist_str = f"{dist*100:.1f}cm" if dist is not None else "no depth"
                    pos_str = f"x={cx} y={cy}"
                    lines = [head, f"{dist_str}  {pos_str}"]

                    sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                             for s in lines]
                    max_w = max(w for w, _ in sizes)
                    line_h = sizes[0][1] + 4
                    box_h = line_h * len(lines) + 4
                    cv2.rectangle(display, (x1, y1 - box_h),
                                  (x1 + max_w + 6, y1), color_box, -1)
                    for li, s in enumerate(lines):
                        cv2.putText(display, s,
                                    (x1 + 2, y1 - box_h + line_h * (li + 1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    last_boxes.append(((x1, y1, x2, y2), int(cid)))

            status = (f"filter: {[names[i] for i in filter_ids]}"
                      if filter_ids else "filter: (all)")
            cv2.putText(display, status, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(win, display)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("c"):
                filter_ids = None
                print("Filter cleared")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
