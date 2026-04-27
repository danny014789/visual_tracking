"""
Capture training images of the metal fixture from the D405 color stream.

Hold the fixture in front of the camera at varied distances, angles,
backgrounds, and lighting. Aim for at least 80-150 images.

Controls (focus the OpenCV window):
  SPACE   save current frame to dataset/raw/
  q / Esc quit
"""
from pathlib import Path
import time

import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
OUT_DIR = Path(__file__).parent / "dataset" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def next_index():
    existing = sorted(OUT_DIR.glob("img_*.jpg"))
    if not existing:
        return 0
    last = existing[-1].stem.split("_")[-1]
    return int(last) + 1


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)

    idx = next_index()
    saved_this_session = 0
    last_save_t = 0.0
    flash_until = 0.0

    print(f"Saving to: {OUT_DIR}")
    print("SPACE = save frame  |  q / Esc = quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            color = np.asanyarray(cf.get_data())
            display = color.copy()

            total = len(list(OUT_DIR.glob("img_*.jpg")))
            cv2.putText(display, f"saved (this session): {saved_this_session}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"total in dataset/raw: {total}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "SPACE=save  q/Esc=quit",
                        (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if time.time() < flash_until:
                cv2.rectangle(display, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 255, 0), 6)

            cv2.imshow("D405 capture", display)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                # Debounce so a held key doesn't burst-save 30 frames
                if time.time() - last_save_t < 0.15:
                    continue
                path = OUT_DIR / f"img_{idx:04d}.jpg"
                # cv2.imwrite silently fails on Windows paths with non-ASCII chars,
                # so encode in memory and write bytes via Python file IO.
                ok, buf = cv2.imencode(".jpg", color)
                if not ok:
                    print(f"FAILED to encode {path.name}")
                    continue
                path.write_bytes(buf.tobytes())
                idx += 1
                saved_this_session += 1
                last_save_t = time.time()
                flash_until = time.time() + 0.15
                print(f"saved {path.name}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
