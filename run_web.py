"""
run_web.py — Titik Masuk Utama Scolio-Scan (Versi Web App)
"""

import cv2
import time
import sys
import os
import threading
import copy
from flask import Flask, render_template, Response, jsonify

from src.camera import ThreadedCamera
from src.detector import PoseDetector
from src.alert import PostureAlertSystem, PostureState
from src.logger import SessionLogger
from src.overlay import draw_skeleton

# ── Konfigurasi ──────────────────────────────────────────────
CAMERA_SOURCE = "VID_20260623_210838.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_DISPLAY_HEIGHT = 720
ANGLE_THRESHOLD = 10.0       # Derajat
ALERT_DURATION = 10.0        # Detik
CALIBRATION_DURATION = 3.0   # Detik

app = Flask(__name__)

# Global state untuk dikirim ke frontend via API
global_state = {
    "angle": 0.0,
    "state": "CALIBRATING",
    "elapsed": 0.0,
    "fps": 0.0,
    "calibration_remaining": 0.0,
    "timer_progress": 0.0,
    "detection_mode": "full_body",
    "is_detecting": False
}

class InferenceThread:
    def __init__(self, detector: PoseDetector):
        self.detector = detector
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ready = False
        self._result = None
        self._ai_fps = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()
            self._frame_ready = True

    def get_result(self):
        with self._lock:
            return copy.deepcopy(self._result), self._ai_fps

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)

    def _worker(self):
        ai_fps_timer = time.time()
        ai_frame_count = 0
        while self._running:
            frame_to_process = None
            with self._lock:
                if self._frame_ready:
                    frame_to_process = self._frame
                    self._frame_ready = False
            if frame_to_process is None:
                time.sleep(0.005)
                continue
            keypoints = self.detector.detect(frame_to_process)
            with self._lock:
                if keypoints is not None:
                    self._result = keypoints
            ai_frame_count += 1
            now = time.time()
            if now - ai_fps_timer >= 1.0:
                with self._lock:
                    self._ai_fps = ai_frame_count / (now - ai_fps_timer)
                ai_frame_count = 0
                ai_fps_timer = now

# Inisialisasi global
detector = PoseDetector(model_path="yolo26n-pose.pt", confidence=0.5)
ai_thread = InferenceThread(detector)
alert_system = PostureAlertSystem(ANGLE_THRESHOLD, ALERT_DURATION, CALIBRATION_DURATION)
logger = None
camera = None
cap = None
is_video_file = False
video_fps = 30.0

def init_system(student_id, screen_time):
    global camera, cap, is_video_file, logger, video_fps
    
    if isinstance(CAMERA_SOURCE, str) and os.path.exists(CAMERA_SOURCE):
        _, ext = os.path.splitext(CAMERA_SOURCE.lower())
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video_file = True

    if is_video_file:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print("Gagal membuka video!")
            sys.exit(1)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 30.0
    else:
        camera = ThreadedCamera(source=CAMERA_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    
    logger = SessionLogger(student_id=student_id, screen_time_min=screen_time)
    alert_system.start_calibration()

def generate_frames():
    global global_state
    display_fps = 0.0
    fps_timer = time.time()
    fps_frame_count = 0
    last_keypoints = None

    while True:
        if is_video_file:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Video selesai, loop ulang untuk keperluan demo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Kontrol kecepatan video file
            time.sleep(1.0 / video_fps)
        else:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)

        # Resize
        h, w = frame.shape[:2]
        if is_video_file:
            target_h = min(h, MAX_DISPLAY_HEIGHT)
            scale = target_h / h
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Async Inference
        ai_thread.submit_frame(frame)
        keypoints, ai_fps = ai_thread.get_result()
        if keypoints is not None:
            last_keypoints = keypoints

        # Proses State
        if last_keypoints is not None:
            frame = draw_skeleton(frame, last_keypoints)
            state, angle, elapsed = alert_system.update(last_keypoints)
            active_mode = alert_system.get_active_mode()
            
            global_state["is_detecting"] = True
            global_state["angle"] = angle
            global_state["state"] = state.name
            global_state["elapsed"] = elapsed
            global_state["detection_mode"] = active_mode
            global_state["fps"] = display_fps
            
            if state == PostureState.CALIBRATING:
                global_state["calibration_remaining"] = alert_system.get_calibration_remaining()
            else:
                global_state["timer_progress"] = alert_system.get_timer_progress()
                logger.log(angle, state.value, active_mode)
        else:
            global_state["is_detecting"] = False
            global_state["state"] = "CALIBRATING"
            global_state["fps"] = display_fps

        fps_frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            display_fps = fps_frame_count / (now - fps_timer)
            fps_frame_count = 0
            fps_timer = now

        # Encode to JPEG (TIDAK ADA TEKS UI, HANYA SKELETON)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify(global_state)

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    alert_system.start_calibration()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("=" * 50)
    print("  SCOLIO-SCAN WEB APP")
    print("=" * 50)
    student_id = input("  Masukkan Nama/ID Siswa : ").strip() or "unknown"
    try:
        screen_time = float(input("  Screen Time Harian (Menit) : ").strip() or "0")
    except ValueError:
        screen_time = 0.0
    
    init_system(student_id, screen_time)
    print("\n[INFO] Menjalankan Server Web di http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
