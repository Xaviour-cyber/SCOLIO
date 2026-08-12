"""
alert.py — Modul Thresholding, State Machine, dan Kalibrasi

Mengelola kondisi postur (NORMAL/WARNING/ALERT) dan
auto-kalibrasi posisi duduk tegak sebagai referensi nol.

Mendukung dua mode deteksi:
  - FULL_BODY  : Menggunakan sudut spine (neck → mid_hip)
  - UPPER_BODY : Menggunakan sudut komposit (kemiringan bahu + kepala)
"""

import time
import winsound
from enum import Enum
from dataclasses import dataclass, field

from .geometry import (
    calculate_spine_angle,
    calculate_relative_angle,
    calculate_composite_upper_body_angle,
)


class PostureState(Enum):
    """Status postur pengguna."""
    CALIBRATING = "CALIBRATING"
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ALERT = "ALERT"


@dataclass
class CalibrationData:
    """Data kalibrasi posisi duduk tegak."""
    baseline_spine_angle: float = 0.0
    baseline_upper_body_angle: float = 0.0
    baseline_shoulder_tilt: float = 0.0
    is_calibrated: bool = False
    samples_spine: list = field(default_factory=list)
    samples_upper: list = field(default_factory=list)
    detection_mode: str = ""  # Mode yang digunakan saat kalibrasi


class PostureAlertSystem:
    """
    Sistem alert postur dengan state machine dan auto-kalibrasi.

    State transitions:
        CALIBRATING → NORMAL (setelah kalibrasi selesai)
        NORMAL → WARNING (sudut > threshold)
        WARNING → ALERT (sudut > threshold selama ≥ alert_duration detik)
        WARNING → NORMAL (sudut ≤ threshold selama hysteresis frames)
        ALERT → NORMAL (sudut ≤ threshold selama hysteresis frames)
    """

    def __init__(
        self,
        angle_threshold: float = 15.0,
        alert_duration: float = 10.0,
        calibration_duration: float = 3.0,
        hysteresis_frames: int = 2,
        beep_interval: float = 5.0,
    ):
        """
        Inisialisasi sistem alert.

        Args:
            angle_threshold: Sudut maksimum (derajat) sebelum dianggap postur buruk
            alert_duration: Durasi (detik) postur buruk sebelum status ALERT
            calibration_duration: Durasi (detik) untuk auto-kalibrasi
            hysteresis_frames: Jumlah frame berturut-turut untuk transisi state
            beep_interval: Interval (detik) antara bunyi beep saat ALERT
        """
        self.angle_threshold = angle_threshold
        self.alert_duration = alert_duration
        self.calibration_duration = calibration_duration
        self.hysteresis_frames = hysteresis_frames
        self.beep_interval = beep_interval

        # State
        self.state = PostureState.CALIBRATING
        self.calibration = CalibrationData()
        self.calibration_start_time: float | None = None

        # Tracking postur buruk
        self._bad_posture_start: float | None = None
        self._bad_posture_elapsed: float = 0.0
        self._consecutive_good: int = 0
        self._consecutive_bad: int = 0

        # Audio
        self._last_beep_time: float = 0.0

        # Mode deteksi aktif
        self._active_mode: str = ""

    def start_calibration(self):
        """Mulai proses kalibrasi."""
        self.state = PostureState.CALIBRATING
        self.calibration = CalibrationData()
        self.calibration_start_time = time.time()
        self._bad_posture_start = None
        self._bad_posture_elapsed = 0.0
        self._consecutive_good = 0
        self._consecutive_bad = 0
        self._active_mode = ""

    def get_calibration_remaining(self) -> float:
        """Sisa waktu kalibrasi dalam detik."""
        if self.calibration_start_time is None:
            return self.calibration_duration
        elapsed = time.time() - self.calibration_start_time
        remaining = self.calibration_duration - elapsed
        return max(0.0, remaining)

    def get_active_mode(self) -> str:
        """Kembalikan mode deteksi yang sedang aktif."""
        return self._active_mode

    def _compute_angle(self, keypoints: dict) -> tuple[float, str]:
        """
        Hitung sudut postur berdasarkan mode deteksi.

        Args:
            keypoints: Dictionary keypoint dari PoseDetector

        Returns:
            Tuple (angle, mode) — sudut saat ini dan mode yang digunakan
        """
        mode = keypoints.get("detection_mode", "full_body")

        if mode == "full_body":
            # Menggunakan nose sebagai titik atas tulang belakang, sehingga stabil walau bahu diangkat
            angle = calculate_spine_angle(keypoints["nose"], keypoints["mid_hip"])
        else:
            # Upper body mode — gunakan sudut komposit
            angle = calculate_composite_upper_body_angle(
                keypoints["nose"],
                keypoints["neck"],
                keypoints["left_shoulder"],
                keypoints["right_shoulder"],
            )

        return angle, mode

    def update_calibration(self, keypoints: dict) -> bool:
        """
        Update data kalibrasi dengan keypoints baru.

        Args:
            keypoints: Dictionary keypoint dari PoseDetector

        Returns:
            True jika kalibrasi selesai, False jika masih berlangsung
        """
        if self.state != PostureState.CALIBRATING:
            return True

        if self.calibration_start_time is None:
            self.calibration_start_time = time.time()

        # Kumpulkan sampel sudut
        angle, mode = self._compute_angle(keypoints)

        if mode == "full_body":
            self.calibration.samples_spine.append(angle)
        else:
            self.calibration.samples_upper.append(angle)

        # Catat mode terakhir yang digunakan
        self.calibration.detection_mode = mode

        # Periksa apakah durasi kalibrasi sudah habis
        elapsed = time.time() - self.calibration_start_time
        if elapsed >= self.calibration_duration:
            # Hitung rata-rata sebagai baseline
            if self.calibration.samples_spine:
                self.calibration.baseline_spine_angle = (
                    sum(self.calibration.samples_spine)
                    / len(self.calibration.samples_spine)
                )
            if self.calibration.samples_upper:
                self.calibration.baseline_upper_body_angle = (
                    sum(self.calibration.samples_upper)
                    / len(self.calibration.samples_upper)
                )

            self.calibration.is_calibrated = True
            self._active_mode = self.calibration.detection_mode
            self.state = PostureState.NORMAL
            print(f"[OK] Kalibrasi selesai. Mode: {self._active_mode}")
            return True

        return False

    def update(self, keypoints: dict) -> tuple[PostureState, float, float]:
        """
        Update status postur berdasarkan keypoints terbaru.

        Args:
            keypoints: Dictionary keypoint dari PoseDetector

        Returns:
            Tuple (state, relative_angle, bad_posture_elapsed)
        """
        # Jika masih kalibrasi
        if self.state == PostureState.CALIBRATING:
            self.update_calibration(keypoints)
            return self.state, 0.0, 0.0

        # Hitung sudut saat ini
        current_angle, mode = self._compute_angle(keypoints)
        self._active_mode = mode

        # Pilih baseline yang sesuai
        if mode == "full_body":
            baseline = self.calibration.baseline_spine_angle
        else:
            baseline = self.calibration.baseline_upper_body_angle

        # Hitung sudut relatif terhadap baseline
        relative_angle = calculate_relative_angle(current_angle, baseline)

        # Evaluasi apakah postur buruk
        is_bad_posture = relative_angle > self.angle_threshold

        if is_bad_posture:
            self._consecutive_bad += 1
            self._consecutive_good = 0

            # Mulai timer postur buruk (dengan hysteresis)
            if self._consecutive_bad >= self.hysteresis_frames:
                if self._bad_posture_start is None:
                    self._bad_posture_start = time.time()
                self._bad_posture_elapsed = time.time() - self._bad_posture_start

                # Tentukan state berdasarkan durasi
                if self._bad_posture_elapsed >= self.alert_duration:
                    self.state = PostureState.ALERT
                    self._trigger_audio_alert()
                else:
                    self.state = PostureState.WARNING
        else:
            self._consecutive_good += 1
            self._consecutive_bad = 0

            # Kembali ke NORMAL (dengan hysteresis)
            if self._consecutive_good >= self.hysteresis_frames:
                self.state = PostureState.NORMAL
                self._bad_posture_start = None
                self._bad_posture_elapsed = 0.0

        return self.state, relative_angle, self._bad_posture_elapsed

    def _trigger_audio_alert(self):
        """Mainkan bunyi beep jika interval sudah tercapai."""
        now = time.time()
        if now - self._last_beep_time >= self.beep_interval:
            self._last_beep_time = now
            try:
                # Windows-native beep: frequency=1000Hz, duration=500ms
                winsound.Beep(1000, 500)
            except Exception:
                pass  # Abaikan error audio di lingkungan tanpa audio

    def get_timer_progress(self) -> float:
        """
        Progress bar postur buruk (0.0 s/d 1.0).

        Returns:
            Nilai 0.0 (baru mulai buruk) hingga 1.0 (sudah ALERT).
        """
        if self._bad_posture_elapsed <= 0:
            return 0.0
        return min(1.0, self._bad_posture_elapsed / self.alert_duration)
