import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create directories to save images
os.makedirs("images", exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Create alignment object to align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()

        # Align depth to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display images
        cv2.imshow("RGB", color_image)

        # Apply colormap to depth image (for visualization)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_colormap)

        # Save images when 's' is pressed
        key = cv2.waitKey(1)
        if key == ord("s"):
            frame_number = int(len(os.listdir("images")) / 2)
            # Save RGB image
            cv2.imwrite(
                "images/{}_color.png".format(frame_number),
                color_image,
            )

            # Save raw depth data (16-bit PNG)
            cv2.imwrite(
                "images/{}_depth.png".format(frame_number),
                depth_image,
            )

            print("Images saved!")

        if key == ord("q") or cv2.getWindowProperty("RGB", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
