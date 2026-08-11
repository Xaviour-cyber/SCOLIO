"""
run.py — Titik Masuk Utama Scolio-Scan

Menjalankan pipeline lengkap dengan arsitektur multi-thread:
  - Main Thread  : Baca frame → resize → overlay → display (smooth FPS)
  - AI Thread    : YOLO inference di background (tidak menghalangi display)

Kontrol:
  - Tekan Q untuk keluar
  - Tekan R untuk kalibrasi ulang
  - Tekan M untuk toggle Mode Render (Async ↔ Sync)
"""

import cv2
import time
import sys
import os
import threading
import copy

from src.camera import ThreadedCamera
from src.detector import PoseDetector
from src.alert import PostureAlertSystem, PostureState
from src.logger import SessionLogger
from src.overlay import (
    draw_skeleton,
    draw_status_panel,
    draw_timer_bar,
    draw_calibration_overlay,
    draw_controls_hint,
)


# ── Konfigurasi ──────────────────────────────────────────────
WINDOW_NAME = "Scolio-Scan | Monitoring Postur"
# Gunakan URL DroidCam (HP sebagai webcam via WiFi)
# Ganti IP sesuai yang tampil di app DroidCam HP kamu
# Untuk webcam biasa, ganti ke: CAMERA_SOURCE = 0
# Untuk video file, ganti ke: CAMERA_SOURCE = "video.mp4"
CAMERA_SOURCE = "VID_20260623_210838.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_DISPLAY_HEIGHT = 720     # Batas tinggi tampilan agar muat di layar
ANGLE_THRESHOLD = 15.0       # Derajat
ALERT_DURATION = 10.0        # Detik
CALIBRATION_DURATION = 3.0   # Detik


class InferenceThread:
    """
    Thread terpisah untuk menjalankan YOLO inference di background.

    Main thread hanya perlu memasukkan frame terbaru via `submit_frame()`,
    dan mengambil hasil deteksi terakhir via `get_result()`.
    Tidak ada blocking — main thread tetap berjalan smooth.
    """

    def __init__(self, detector: PoseDetector):
        self.detector = detector
        self._lock = threading.Lock()
        self._frame = None             # Frame terbaru untuk diproses
        self._frame_ready = False      # Flag ada frame baru
        self._result = None            # Hasil deteksi terakhir
        self._ai_fps = 0.0             # FPS inferensi AI
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        """Kirim frame terbaru ke AI thread (non-blocking, overwrite jika belum diproses)."""
        with self._lock:
            self._frame = frame.copy()
            self._frame_ready = True

    def get_result(self):
        """Ambil hasil deteksi terakhir (thread-safe). Returns (keypoints, ai_fps)."""
        with self._lock:
            return copy.deepcopy(self._result), self._ai_fps

    def stop(self):
        """Hentikan thread."""
        self._running = False
        self._thread.join(timeout=2.0)

    def _worker(self):
        """Loop utama AI thread."""
        ai_fps_timer = time.time()
        ai_frame_count = 0

        while self._running:
            # Ambil frame terbaru (jika ada)
            frame_to_process = None
            with self._lock:
                if self._frame_ready:
                    frame_to_process = self._frame
                    self._frame_ready = False

            if frame_to_process is None:
                # Tidak ada frame baru, tunggu sebentar agar tidak busy-loop
                time.sleep(0.005)
                continue

            # ── Jalankan YOLO inference (bagian yang lambat) ──
            keypoints = self.detector.detect(frame_to_process)

            # Simpan hasil
            with self._lock:
                if keypoints is not None:
                    self._result = keypoints

            # Hitung AI FPS
            ai_frame_count += 1
            now = time.time()
            if now - ai_fps_timer >= 1.0:
                with self._lock:
                    self._ai_fps = ai_frame_count / (now - ai_fps_timer)
                ai_frame_count = 0
                ai_fps_timer = now


def main():
    """Loop utama aplikasi."""
    print("=" * 50)
    print("  SCOLIO-SCAN")
    print("  Sistem Monitoring Postur Tulang Belakang")
    print("  [Mode: Async AI Inference (Default)]")
    print("  Tekan M untuk Toggle Mode Render")
    print("=" * 50)
    print()

    # ── Input Data Siswa ─────────────────────────────────────
    print("─" * 50)
    student_id = input("  Masukkan Nama/ID Siswa : ").strip() or "unknown"
    try:
        screen_time = float(input("  Screen Time Harian (Menit) : ").strip() or "0")
    except ValueError:
        screen_time = 0.0
    print("─" * 50)
    print()

    # ── Cek Tipe Sumber Video ────────────────────────────────
    is_video_file = False
    if isinstance(CAMERA_SOURCE, str) and os.path.exists(CAMERA_SOURCE):
        _, ext = os.path.splitext(CAMERA_SOURCE.lower())
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video_file = True

    # ── Inisialisasi Sumber Kamera/Video ─────────────────────
    print("[INFO] Membuka sumber video...")
    camera = None
    cap = None
    video_fps = 30.0

    try:
        if is_video_file:
            print(f"[INFO] Membuka file video: {CAMERA_SOURCE}")
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            if not cap.isOpened():
                raise RuntimeError(f"Gagal membuka file video: {CAMERA_SOURCE}")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0
            print(f"[OK] Video berhasil dibuka. FPS asli: {video_fps:.1f}")
        else:
            camera = ThreadedCamera(
                source=CAMERA_SOURCE,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
            print("[OK] Kamera berhasil dibuka.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Inisialisasi Detektor & AI Thread ────────────────────
    print("[INFO] Memuat model YOLOv26-Pose Nano...")
    detector = PoseDetector(model_path="yolo26n-pose.pt", confidence=0.5)
    print("[OK] Model berhasil dimuat.")

    print("[INFO] Memulai AI Inference Thread...")
    ai_thread = InferenceThread(detector)
    print("[OK] AI Thread aktif — inferensi berjalan di background.")

    # ── Inisialisasi Sistem Alert ────────────────────────────
    alert_system = PostureAlertSystem(
        angle_threshold=ANGLE_THRESHOLD,
        alert_duration=ALERT_DURATION,
        calibration_duration=CALIBRATION_DURATION,
    )
    alert_system.start_calibration()
    print("[INFO] Memulai kalibrasi — Duduk tegak dan menghadap kamera!")
    print()

    # ── Inisialisasi Data Logger ──────────────────────────────
    logger = SessionLogger(student_id=student_id, screen_time_min=screen_time)

    # ── Variabel loop ────────────────────────────────────────
    display_fps = 0.0
    fps_timer = time.time()
    fps_frame_count = 0
    last_keypoints = None
    sync_mode = False          # False = Async (mulus), True = Sync (presisi)

    # ── Window OpenCV ────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    try:
        while True:
            # ── Ambil frame terbaru ──────────────────────────
            if is_video_file:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[INFO] Video selesai diputar.")
                    break
            else:
                ret, frame = camera.get_frame()
                if not ret or frame is None:
                    continue

            if not is_video_file:
                # Flip horizontal (mirror) agar lebih intuitif untuk kamera langsung
                frame = cv2.flip(frame, 1)

            # ── Resize frame agar pas di layar ───────────────
            h, w = frame.shape[:2]
            if is_video_file:
                target_h = min(h, MAX_DISPLAY_HEIGHT)
                scale = target_h / h
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            else:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ── Deteksi Pose (Mode tergantung sync_mode) ──────
            if sync_mode:
                # SYNC MODE: Proses langsung di main thread
                # Skeleton 100% akurat tapi video jadi lambat
                keypoints = detector.detect(frame)
                if keypoints is not None:
                    last_keypoints = keypoints
                ai_fps = display_fps  # Di sync mode, AI FPS = Display FPS
            else:
                # ASYNC MODE (default): Kirim ke AI Thread
                # Video mulus tapi skeleton agak delay
                ai_thread.submit_frame(frame)
                keypoints, ai_fps = ai_thread.get_result()
                if keypoints is not None:
                    last_keypoints = keypoints

            # ── Proses keypoints & gambar overlay ────────────
            if last_keypoints is not None:
                # Gambar skeleton
                frame = draw_skeleton(frame, last_keypoints)

                # Update sistem alert
                state, angle, elapsed = alert_system.update(last_keypoints)
                active_mode = alert_system.get_active_mode()

                # Gambar UI berdasarkan state
                if state == PostureState.CALIBRATING:
                    remaining = alert_system.get_calibration_remaining()
                    frame = draw_calibration_overlay(frame, remaining)
                    frame = draw_status_panel(frame, state, 0.0, 0.0, display_fps,
                                              detection_mode=active_mode)
                else:
                    frame = draw_status_panel(frame, state, angle, elapsed, display_fps,
                                              detection_mode=active_mode)
                    # Timer bar
                    progress = alert_system.get_timer_progress()
                    frame = draw_timer_bar(frame, progress)

                    # ── Catat data ke logger ──────────────
                    logger.log(angle, state.value, active_mode)
            else:
                # Tidak ada deteksi — tampilkan status informatif
                frame = draw_status_panel(
                    frame, PostureState.CALIBRATING, 0.0, 0.0, display_fps
                )
                # Pesan "tidak terdeteksi"
                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    "Tidak ada orang terdeteksi",
                    (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # ── Kontrol hint ─────────────────────────────────
            render_mode_label = "SYNC (Presisi)" if sync_mode else "ASYNC (Mulus)"
            frame = draw_controls_hint(frame, render_mode=render_mode_label)

            # ── Hitung Display FPS ───────────────────────────
            fps_frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                display_fps = fps_frame_count / (now - fps_timer)
                fps_frame_count = 0
                fps_timer = now

            # ── Tampilkan frame ──────────────────────────────
            cv2.imshow(WINDOW_NAME, frame)

            # ── Kontrol keyboard & Pacing ────────────────────
            if is_video_file:
                frame_delay = max(1, int(1000 / video_fps))
                key = cv2.waitKey(frame_delay) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] Keluar dari Scolio-Scan...")
                break

            if key == ord("r") or key == ord("R"):
                print("[INFO] Kalibrasi ulang — Duduk tegak dan menghadap kamera!")
                alert_system.start_calibration()
                last_keypoints = None

            if key == ord("m") or key == ord("M"):
                sync_mode = not sync_mode
                mode_name = "SYNC (Presisi)" if sync_mode else "ASYNC (Mulus)"
                print(f"[INFO] Mode Render diubah ke: {mode_name}")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna.")

    finally:
        # Cleanup — hentikan AI thread dan lepas resources
        print("[INFO] Menghentikan AI Thread...")
        ai_thread.stop()
        if is_video_file and cap is not None:
            cap.release()
        elif camera is not None:
            camera.release()
        cv2.destroyAllWindows()

        # ── Generate laporan & grafik otomatis ────────────
        logger.close_and_report()

        print("[OK] Scolio-Scan ditutup dengan bersih.")


if __name__ == "__main__":
    main()
