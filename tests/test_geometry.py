"""
test_geometry.py — Unit Tests untuk Modul Perhitungan Sudut

Memvalidasi calculate_spine_angle, calculate_shoulder_tilt,
calculate_head_tilt, dan calculate_relative_angle dengan
input geometri yang diketahui.
"""

import math
import pytest
from src.geometry import (
    calculate_spine_angle,
    calculate_shoulder_tilt,
    calculate_head_tilt,
    calculate_relative_angle,
)


class TestCalculateSpineAngle:
    """Uji coba fungsi calculate_spine_angle."""

    def test_perfectly_vertical(self):
        """Tulang belakang tegak sempurna → 0°."""
        neck = (320, 100)
        mid_hip = (320, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert angle == 0.0

    def test_15_degree_tilt(self):
        """Kemiringan ~15° → hasilnya mendekati 15°."""
        # tan(15°) ≈ 0.2679
        # Jika dy = 300, maka dx = 300 * tan(15°) ≈ 80.38
        dy = 300
        dx = dy * math.tan(math.radians(15))
        neck = (320 + dx, 100)
        mid_hip = (320, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert abs(angle - 15.0) < 0.5  # Toleransi 0.5°

    def test_45_degree_tilt(self):
        """Kemiringan 45° → hasilnya mendekati 45°."""
        # tan(45°) = 1, jadi dx = dy
        neck = (620, 100)
        mid_hip = (320, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert abs(angle - 45.0) < 0.5

    def test_identical_points(self):
        """Titik identik → 0°."""
        neck = (320, 200)
        mid_hip = (320, 200)
        angle = calculate_spine_angle(neck, mid_hip)
        assert angle == 0.0

    def test_small_deviation(self):
        """Deviasi kecil (~5°) → hasilnya mendekati 5°."""
        dy = 300
        dx = dy * math.tan(math.radians(5))
        neck = (320 + dx, 100)
        mid_hip = (320, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert abs(angle - 5.0) < 0.5

    def test_neck_left_of_hip(self):
        """Neck di kiri hip → tetap sudut positif."""
        dy = 300
        dx = dy * math.tan(math.radians(20))
        neck = (320 - dx, 100)  # kiri
        mid_hip = (320, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert abs(angle - 20.0) < 0.5

    def test_always_positive(self):
        """Sudut selalu positif (abs)."""
        neck = (100, 100)
        mid_hip = (400, 400)
        angle = calculate_spine_angle(neck, mid_hip)
        assert angle >= 0


class TestCalculateShoulderTilt:
    """Uji coba fungsi calculate_shoulder_tilt."""

    def test_level_shoulders(self):
        """Bahu sejajar horizontal → 0°."""
        left = (200, 200)
        right = (440, 200)
        tilt = calculate_shoulder_tilt(left, right)
        assert tilt == 0.0

    def test_left_higher(self):
        """Bahu kiri lebih tinggi (y lebih kecil) → sudut negatif."""
        left = (200, 180)   # lebih tinggi
        right = (440, 200)
        tilt = calculate_shoulder_tilt(left, right)
        # dy = 200-180 = 20 (positif, artinya turun ke kanan)
        assert tilt > 0  # atan2(20, 240) > 0

    def test_right_higher(self):
        """Bahu kanan lebih tinggi (y lebih kecil) → sudut negatif."""
        left = (200, 200)
        right = (440, 180)  # lebih tinggi
        tilt = calculate_shoulder_tilt(left, right)
        # dy = 180-200 = -20 (negatif), tapi rumus baru pakai abs() jadi hasil selalu positif
        assert tilt > 0
        assert abs(tilt - 4.76) < 0.5

    def test_identical_points(self):
        """Titik identik → 0°."""
        pt = (320, 200)
        tilt = calculate_shoulder_tilt(pt, pt)
        assert tilt == 0.0


class TestCalculateHeadTilt:
    """Uji coba fungsi calculate_head_tilt."""

    def test_head_centered(self):
        """Kepala tepat di atas leher → 0°."""
        nose = (320, 50)
        neck = (320, 150)
        tilt = calculate_head_tilt(nose, neck)
        assert tilt == 0.0

    def test_head_forward(self):
        """Kepala maju ke depan → sudut > 0."""
        nose = (370, 50)  # geser ke samping (simulasi maju dari sisi)
        neck = (320, 150)
        tilt = calculate_head_tilt(nose, neck)
        assert tilt > 0

    def test_identical_points(self):
        """Titik identik → 0°."""
        pt = (320, 100)
        tilt = calculate_head_tilt(pt, pt)
        assert tilt == 0.0


class TestCalculateRelativeAngle:
    """Uji coba fungsi calculate_relative_angle."""

    def test_no_deviation(self):
        """Sudut sama dengan baseline → 0°."""
        assert calculate_relative_angle(10.0, 10.0) == 0.0

    def test_positive_deviation(self):
        """Sudut lebih besar dari baseline."""
        assert calculate_relative_angle(25.0, 10.0) == 15.0

    def test_negative_deviation(self):
        """Sudut lebih kecil dari baseline → tetap positif."""
        result = calculate_relative_angle(5.0, 10.0)
        assert result == 5.0
        assert result >= 0

    def test_zero_baseline(self):
        """Baseline nol → angle langsung."""
        assert calculate_relative_angle(12.5, 0.0) == 12.5
