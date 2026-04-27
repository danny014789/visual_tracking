import numpy as np
import cv2
import pyrealsense2 as rs

# D405 supports 1280x720 only at 5fps; 640x480 @ 30fps is the common live mode
WIDTH, HEIGHT, FPS = 640, 480, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)

# Align depth to color so the two images share the same viewpoint
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET
        )

        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("D405 - color | depth", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # Esc or q
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
