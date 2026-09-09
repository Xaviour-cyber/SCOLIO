import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.overlay import draw_status_panel
from src.alert import PostureState

# Create a blank frame (simulating a 640x480 webcam feed)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[:] = (50, 50, 50)  # Dark gray background

# Draw status panel
frame = draw_status_panel(
    frame=frame,
    state=PostureState.WARNING,
    angle=8.5,
    elapsed=0.0,
    fps=30.0,
    detection_mode="PRESISI"
)

# Save the frame
output_path = os.path.join(os.path.dirname(__file__), 'preview_overlay_desktop_no_logo.png')
cv2.imwrite(output_path, frame)
print(f"Saved preview to {output_path}")
