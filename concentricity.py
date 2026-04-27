"""
concentricity.py — verify a circular fixture is concentric with a circular
hole using the Intel RealSense D405.

Workflow
--------
1. With the hole visible and NO fixture in view, press 'h' to lock the hole.
2. The arm hovers the fixture above the hole.
3. The script detects the fixture in the foreground depth band and reports
   the lateral offset (mm) between fixture and hole centers.
4. After N consecutive frames within tolerance, status flips to PASS —
   the go-signal for the rod.

Controls (focus the OpenCV window):
  h       (re)lock the hole on the current frame
  r       reset (clear hole + counters)
  +/-     adjust tolerance by 0.1 mm
  q/Esc   quit
"""
import argparse
import math

import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="concentricity tolerance in mm (default 1.0)")
    p.add_argument("--stability", type=int, default=5,
                   help="consecutive in-tolerance frames required for PASS")
    p.add_argument("--depth_margin", type=float, default=0.015,
                   help="m: how much closer than the hole the fixture must "
                        "be to count as foreground (default 0.015 = 15 mm)")
    p.add_argument("--min_r", type=int, default=15)
    p.add_argument("--max_r", type=int, default=200)
    return p.parse_args()


def sample_depth_m(depth_image, cx, cy, depth_scale, half=4):
    h, w = depth_image.shape
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def hough_circles(gray, min_r, max_r, param2):
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
        param1=100, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return []
    return [tuple(int(v) for v in c) for c in np.round(circles[0]).astype(int)]


def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx = intr.fx

    hole = None  # (cx, cy, r_px, depth_m)
    stable = 0
    tolerance = args.tolerance
    win = "D405 concentricity"
    cv2.namedWindow(win)

    print("Controls: h=lock hole  r=reset  +/-=tol  q/Esc=quit")
    print(f"Tolerance: {tolerance:.2f} mm  Stability: {args.stability} frames")

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())
            display = color.copy()

            # --- Fixture detection: Hough on full frame, keep candidates whose
            # center lies in the foreground depth band (closer than the hole).
            fixture = None
            if hole is not None:
                _, _, _, hole_depth_m = hole
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                candidates = []
                for cx, cy, r in hough_circles(gray, args.min_r, args.max_r,
                                               param2=25):
                    d = sample_depth_m(depth, cx, cy, depth_scale)
                    if d is not None and 0 < d < hole_depth_m - args.depth_margin:
                        candidates.append((cx, cy, r, d))
                if candidates:
                    fixture = max(candidates, key=lambda t: t[2])  # largest

            # --- Draw hole reference ---
            if hole is not None:
                hx, hy, hr, _ = hole
                cv2.circle(display, (hx, hy), hr, (255, 0, 0), 2)
                cv2.drawMarker(display, (hx, hy), (255, 0, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

            # --- Concentricity ---
            offset_mm = None
            if hole is not None and fixture is not None:
                fxc, fyc, fr, fdepth = fixture
                hxc, hyc, _, _ = hole
                offset_px = math.hypot(fxc - hxc, fyc - hyc)
                offset_mm = offset_px * fdepth * 1000.0 / fx

                if offset_mm <= tolerance:
                    stable += 1
                else:
                    stable = 0

                in_tol = offset_mm <= tolerance
                box_color = (0, 255, 0) if in_tol else (0, 0, 255)
                cv2.circle(display, (fxc, fyc), fr, box_color, 2)
                cv2.drawMarker(display, (fxc, fyc), box_color,
                               markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                cv2.line(display, (hxc, hyc), (fxc, fyc), (0, 255, 255), 1)
            else:
                stable = 0

            # --- HUD ---
            def put(text, y, color=(255, 255, 255), scale=0.55):
                cv2.rectangle(display, (5, y - 16),
                              (5 + int(11 * scale * len(text)) + 6, y + 4),
                              (0, 0, 0), -1)
                cv2.putText(display, text, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

            if hole is None:
                put("Press 'h' with NO fixture in view to lock the hole",
                    22, (0, 200, 255))
            else:
                put(f"Tol: {tolerance:.2f} mm   Stab: {stable}/{args.stability}",
                    22)
                if fixture is None:
                    put("waiting for fixture...", 44, (0, 200, 255))
                elif offset_mm is None:
                    put("fixture detected (no depth)", 44, (0, 200, 255))
                else:
                    put(f"Offset: {offset_mm:.2f} mm", 44)
                    if stable >= args.stability:
                        put("PASS - concentric, rod OK", 70, (0, 255, 0), 0.7)
                    elif offset_mm <= tolerance:
                        put("aligning... (in tolerance)", 70, (0, 255, 255), 0.6)
                    else:
                        put("FAIL - not concentric", 70, (0, 0, 255), 0.7)

            cv2.imshow(win, display)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("h"):
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                circles = hough_circles(gray, max(args.min_r, 20),
                                        max(args.max_r, 250), param2=30)
                if not circles:
                    print("No circle found — adjust lighting/position and try again.")
                else:
                    cx, cy, r = max(circles, key=lambda c: c[2])
                    d = sample_depth_m(depth, cx, cy, depth_scale, half=8)
                    if d is None or d == 0:
                        print("Found a circle but no depth at its center.")
                    else:
                        hole = (cx, cy, r, d)
                        stable = 0
                        print(f"Hole locked: center=({cx},{cy}) "
                              f"r={r}px depth={d*100:.1f}cm")
            elif k == ord("r"):
                hole = None
                stable = 0
                print("Reset.")
            elif k in (ord("+"), ord("=")):
                tolerance += 0.1
                print(f"Tolerance: {tolerance:.2f} mm")
            elif k == ord("-"):
                tolerance = max(0.1, tolerance - 0.1)
                print(f"Tolerance: {tolerance:.2f} mm")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
