"""
geometry.py — Modul Perhitungan Sudut Tulang Belakang & Trigonometri

Menghitung sudut deviasi lateral tulang belakang, kemiringan bahu,
dan kemiringan kepala menggunakan koordinat keypoint.
"""

import math


def calculate_spine_angle(neck: tuple, mid_hip: tuple) -> float:
    """
    Menghitung sudut deviasi lateral (θ) tulang belakang.

    Sudut dihitung antara garis vertikal dan garis neck→mid_hip.
    0° = posisi tegak sempurna (neck tepat di atas mid_hip).

    Menggunakan rumus:
        θ = atan2(|neck.x - hip.x|, |neck.y - hip.y|)

    Catatan: Pada koordinat gambar, sumbu Y terbalik (ke bawah positif),
    sehingga neck.y < hip.y untuk postur normal.

    Args:
        neck: Tuple (x, y) posisi leher (titik tengah bahu)
        mid_hip: Tuple (x, y) posisi tengah pinggul

    Returns:
        Sudut deviasi dalam derajat (selalu positif). 0° = tegak.
    """
    dx = abs(neck[0] - mid_hip[0])
    dy = abs(neck[1] - mid_hip[1])

    # Hindari division by zero jika titik identik
    if dy == 0 and dx == 0:
        return 0.0

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 2)


def calculate_shoulder_tilt(left_shoulder: tuple, right_shoulder: tuple) -> float:
    """
    Menghitung kemiringan bahu (asymmetry).

    0° = bahu sejajar horizontal.
    Nilai positif = bahu kiri lebih tinggi, negatif = bahu kanan lebih tinggi.

    Args:
        left_shoulder: Tuple (x, y) posisi bahu kiri
        right_shoulder: Tuple (x, y) posisi bahu kanan

    Returns:
        Sudut kemiringan dalam derajat (bisa positif/negatif).
    """
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]

    # Hindari division by zero
    if dx == 0 and dy == 0:
        return 0.0

    # Sudut dari horizontal
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 2)


def calculate_head_tilt(nose: tuple, neck: tuple) -> float:
    """
    Menghitung kemiringan kepala ke depan (forward head posture).

    Mengukur sudut antara garis vertikal dan garis neck→nose.
    0° = kepala tepat di atas leher.

    Args:
        nose: Tuple (x, y) posisi hidung
        neck: Tuple (x, y) posisi leher (titik tengah bahu)

    Returns:
        Sudut deviasi dalam derajat (selalu positif). 0° = tegak.
    """
    dx = abs(nose[0] - neck[0])
    dy = abs(nose[1] - neck[1])

    if dy == 0 and dx == 0:
        return 0.0

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 2)


def calculate_relative_angle(current_angle: float, baseline_angle: float) -> float:
    """
    Menghitung sudut relatif terhadap baseline kalibrasi.

    Args:
        current_angle: Sudut saat ini (derajat)
        baseline_angle: Sudut baseline dari kalibrasi (derajat)

    Returns:
        Selisih sudut (derajat, selalu positif).
    """
    return round(abs(current_angle - baseline_angle), 2)


def calculate_composite_upper_body_angle(
    nose: tuple, neck: tuple,
    left_shoulder: tuple, right_shoulder: tuple,
) -> float:
    """
    Menghitung sudut deviasi postur komposit dari upper body saja.

    Digunakan ketika pinggul tidak terdeteksi oleh kamera (misalnya
    saat pengguna duduk dekat dengan webcam laptop).

    Rumus:
        θ_shoulder = |atan2(Δy_bahu, Δx_bahu)|   → kemiringan bahu
        θ_head     = atan2(|Δx_nose|, |Δy_nose|)  → kemiringan kepala lateral
        θ_composite = max(θ_shoulder, θ_head)

    Mengambil nilai MAKSIMUM dari kedua indikator, karena:
    - Jika bahu miring ≥ threshold → postur buruk (indikasi skoliosis)
    - Jika kepala miring ≥ threshold → postur buruk (indikasi text neck)
    - Cukup salah satu yang melebihi batas untuk memicu peringatan.

    Args:
        nose: Tuple (x, y) posisi hidung
        neck: Tuple (x, y) posisi leher (titik tengah bahu)
        left_shoulder: Tuple (x, y) posisi bahu kiri
        right_shoulder: Tuple (x, y) posisi bahu kanan

    Returns:
        Sudut deviasi komposit dalam derajat (selalu positif). 0° = tegak.
    """
    shoulder_tilt = abs(calculate_shoulder_tilt(left_shoulder, right_shoulder))
    head_tilt = calculate_head_tilt(nose, neck)

    composite = max(shoulder_tilt, head_tilt)
    return round(composite, 2)
