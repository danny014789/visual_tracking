import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # meters per raw unit
align = rs.align(rs.stream.color)


def grab_frames():
    frames = align.process(pipeline.wait_for_frames())
    d = frames.get_depth_frame()
    c = frames.get_color_frame()
    if not d or not c:
        return None, None
    return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())


def sample_distance(depth_image, cx, cy, half=3):
    """Median of valid (non-zero) depth pixels in a (2*half+1)^2 patch, in meters."""
    h, w = depth_image.shape
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def select_roi(color_image):
    roi = cv2.selectROI("D405 - select object (ENTER=ok, c=cancel)",
                        color_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("D405 - select object (ENTER=ok, c=cancel)")
    if roi == (0, 0, 0, 0):
        return None
    return roi


def make_tracker():
    return cv2.TrackerCSRT_create()


tracker = None
bbox = None

print("Controls: s = (re)select object  |  q / Esc = quit")

try:
    while True:
        color, depth = grab_frames()
        if color is None:
            continue

        # First-time prompt: pick an ROI
        if tracker is None:
            bbox = select_roi(color)
            if bbox is not None:
                tracker = make_tracker()
                tracker.init(color, bbox)
            else:
                # User cancelled; just show stream until they press s/q
                cv2.imshow("D405 tracker", color)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("s"):
                    tracker = None  # loop back into selection
                continue

        ok, bbox = tracker.update(color)
        display = color.copy()

        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w // 2, y + h // 2
            dist = sample_distance(depth, cx, cy)

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
            label = f"{dist*100:.1f} cm" if dist is not None else "no depth"
            cv2.putText(display, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "tracking lost - press 's' to reselect",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("D405 tracker", display)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            tracker = None  # triggers ROI reselection on next loop
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
