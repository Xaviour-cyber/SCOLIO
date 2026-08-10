# Scolio-Scan

Scolio-Scan is a real-time, computer vision-based spinal posture monitoring system designed for the early detection and prevention of posture-related musculoskeletal disorders (such as scoliosis and kyphosis) among adolescents. 

Utilizing the YOLOv26-Pose Nano model for lightweight and efficient pose estimation, the system tracks 17 anatomical keypoints through standard camera feeds. It mathematically calculates spinal deviation angles relative to a baseline calibration and employs a state-machine logic to provide tiered visual alerts when prolonged postural deviation is detected. The architecture implements multi-threading to separate AI inference from the video rendering loop, ensuring high framerate playback even on standard CPU hardware.

## Core Features

*   **Real-time Pose Estimation**: Implements Ultralytics YOLOv26-Pose Nano for high-speed, CPU-optimized human pose tracking.
*   **Mathematical Posture Analysis**: Calculates exact spinal inclination angles using shoulder-to-mid-hip vector geometry.
*   **Tiered Alert System**: Employs a state machine with three levels of posture classification (Normal, Warning, Alert) based on angle thresholds and duration constraints.
*   **Asynchronous Inference**: Utilizes a multi-threaded architecture to decouple heavy AI inference from the main display loop, maintaining a smooth rendering framerate (15-20+ FPS on CPU).
*   **Auto-Calibration**: Dynamically establishes a user-specific posture baseline upon initialization.

## Prerequisites

*   Windows Operating System (10 or 11)
*   Python 3.10 or higher installed with PATH configured

## Installation

The repository includes automated batch scripts to streamline the deployment process on Windows environments.

1.  Clone the repository:
    ```bash
    git clone https://github.com/Xaviour-cyber/SCOLIO.git
    cd SCOLIO
    ```
2.  Execute the setup script to create a virtual environment and install dependencies:
    ```bash
    setup.bat
    ```

## Usage

Once the setup is complete and dependencies are installed, you can launch the application directly.

1.  Execute the main application script:
    ```bash
    run_app.bat
    ```
2.  **Calibration Phase**: Upon launching, sit upright and face the camera directly for 3 seconds to allow the system to establish a baseline posture.
3.  **Real-time Monitoring**: The system will begin analyzing posture deviations.
4.  **Controls**:
    *   Press `R` to force a recalibration.
    *   Press `Q` to terminate the application safely.

---
*Developed for Scientific Research and Academic Purposes - 2026*
