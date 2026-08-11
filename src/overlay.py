"""
overlay.py — Modul Gambar OpenCV (Skeleton, Teks Status, UI)

Menampilkan overlay skeleton, panel status, gauge sudut,
timer bar, dan UI kalibrasi di atas feed video.

Semua elemen UI menggunakan ukuran relatif terhadap dimensi frame
sehingga tampil proporsional di berbagai resolusi dan rasio aspek.
"""

import cv2
import numpy as np
from .alert import PostureState


# ── Warna (BGR) ──────────────────────────────────────────────
COLOR_GREEN = (0, 220, 100)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (0, 60, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SKELETON = (255, 200, 100)
COLOR_JOINT = (0, 180, 255)
COLOR_PANEL_BG = (30, 30, 30)
COLOR_ACCENT_BLUE = (200, 160, 40)

# ── Font ─────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ── Koneksi skeleton untuk overlay ────────────────────────────
# Index COCO: nose=0, Leye=1, Reye=2, Lear=3, Rear=4,
#   Lsho=5, Rsho=6, Lelb=7, Relb=8, Lwri=9, Rwri=10,
#   Lhip=11, Rhip=12, Lkne=13, Rkne=14, Lank=15, Rank=16
SKELETON_CONNECTIONS = [
    (5, 6),    # bahu kiri → bahu kanan
    (5, 7),    # bahu kiri → siku kiri
    (7, 9),    # siku kiri → pergelangan kiri
    (6, 8),    # bahu kanan → siku kanan
    (8, 10),   # siku kanan → pergelangan kanan
    (5, 11),   # bahu kiri → pinggul kiri
    (6, 12),   # bahu kanan → pinggul kanan
    (11, 12),  # pinggul kiri → pinggul kanan
    (11, 13),  # pinggul kiri → lutut kiri
    (13, 15),  # lutut kiri → ankle kiri
    (12, 14),  # pinggul kanan → lutut kanan
    (14, 16),  # lutut kanan → ankle kanan
]

# Koneksi tambahan untuk garis tulang belakang (neck → mid_hip)
# Digambar terpisah dengan warna khusus


def _scale(value: float, frame_height: int, ref: int = 480) -> int:
    """Skalakan nilai berdasarkan tinggi frame relatif terhadap referensi 480px."""
    return max(1, int(value * frame_height / ref))


def _font_scale(base: float, frame_height: int, ref: int = 480) -> float:
    """Skalakan ukuran font berdasarkan tinggi frame."""
    return max(0.3, base * frame_height / ref)


def draw_skeleton(frame: np.ndarray, keypoints: dict, confidences: np.ndarray = None,
                  min_conf: float = 0.3) -> np.ndarray:
    """
    Gambar overlay skeleton pada frame.

    Args:
        frame: Frame BGR dari OpenCV
        keypoints: Dictionary dari PoseDetector (harus punya 'all_keypoints')
        confidences: Array confidence per keypoint (opsional)
        min_conf: Minimum confidence untuk menggambar

    Returns:
        Frame dengan overlay skeleton
    """
    all_kps = keypoints.get("all_keypoints")
    all_conf = keypoints.get("all_confidences")

    if all_kps is None:
        return frame

    h_frame = frame.shape[0]
    line_thick = _scale(2, h_frame)
    spine_thick = _scale(3, h_frame)
    joint_r = _scale(5, h_frame)

    overlay = frame.copy()

    # Gambar koneksi skeleton
    for (i, j) in SKELETON_CONNECTIONS:
        if all_conf is not None and (all_conf[i] < min_conf or all_conf[j] < min_conf):
            continue
        pt1 = (int(all_kps[i][0]), int(all_kps[i][1]))
        pt2 = (int(all_kps[j][0]), int(all_kps[j][1]))
        cv2.line(overlay, pt1, pt2, COLOR_SKELETON, line_thick, cv2.LINE_AA)

    # Gambar garis tulang belakang (neck → mid_hip) dengan warna khusus
    neck = keypoints.get("neck")
    mid_hip = keypoints.get("mid_hip")
    if neck and mid_hip:
        pt_neck = (int(neck[0]), int(neck[1]))
        pt_hip = (int(mid_hip[0]), int(mid_hip[1]))
        cv2.line(overlay, pt_neck, pt_hip, COLOR_GREEN, spine_thick, cv2.LINE_AA)

    # Gambar garis nose → neck
    nose = keypoints.get("nose")
    if nose and neck:
        pt_nose = (int(nose[0]), int(nose[1]))
        pt_neck = (int(neck[0]), int(neck[1]))
        cv2.line(overlay, pt_nose, pt_neck, COLOR_GREEN, spine_thick, cv2.LINE_AA)

    # Gambar titik sendi
    for i in range(len(all_kps)):
        if all_conf is not None and all_conf[i] < min_conf:
            continue
        pt = (int(all_kps[i][0]), int(all_kps[i][1]))
        cv2.circle(overlay, pt, joint_r, COLOR_JOINT, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, joint_r, COLOR_WHITE, 1, cv2.LINE_AA)

    # Blend overlay dengan frame asli (semi-transparan)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    return frame


def draw_status_panel(frame: np.ndarray, state: PostureState,
                      angle: float, elapsed: float,
                      fps: float = 0.0,
                      detection_mode: str = "") -> np.ndarray:
    """
    Gambar panel status di bagian atas frame.
    Ukuran font dan layout adaptif terhadap dimensi frame.

    Args:
        frame: Frame BGR
        state: Status postur saat ini
        angle: Sudut deviasi relatif (derajat)
        elapsed: Waktu postur buruk berturut-turut (detik)
        fps: Frame per second saat ini

    Returns:
        Frame dengan panel status
    """
    h, w = frame.shape[:2]
    margin = _scale(10, h)
    panel_height = _scale(80, h)

    # Ukuran font adaptif
    title_scale = _font_scale(0.7, h)
    info_scale = _font_scale(0.45, h)
    title_thick = max(1, _scale(2, h))
    info_thick = max(1, _scale(1, h))

    # Tentukan warna dan teks berdasarkan state
    if state == PostureState.NORMAL:
        color = COLOR_GREEN
        status_text = "POSTUR: NORMAL"
    elif state == PostureState.WARNING:
        color = COLOR_YELLOW
        status_text = "! PERBAIKI POSTUR"
    elif state == PostureState.ALERT:
        color = COLOR_RED
        status_text = "!! PERBAIKI POSTUR!"
    elif state == PostureState.CALIBRATING:
        color = (200, 180, 50)
        status_text = "KALIBRASI..."
    else:
        color = COLOR_WHITE
        status_text = "UNKNOWN"

    # Panel background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Garis warna status di bawah panel
    bar_h = max(2, _scale(4, h))
    cv2.rectangle(frame, (0, panel_height - bar_h), (w, panel_height), color, -1)

    # ── Baris 1: Status utama (kiri) + FPS (kanan) ──
    row1_y = margin + _scale(25, h)

    # Cek apakah teks status muat di lebar frame
    status_size = cv2.getTextSize(status_text, FONT_BOLD, title_scale, title_thick)[0]
    while status_size[0] > w * 0.7 and title_scale > 0.35:
        title_scale -= 0.05
        status_size = cv2.getTextSize(status_text, FONT_BOLD, title_scale, title_thick)[0]

    cv2.putText(frame, status_text, (margin, row1_y),
                FONT_BOLD, title_scale, color, title_thick, cv2.LINE_AA)

    # FPS di pojok kanan (baris 1)
    fps_text = f"FPS:{fps:.0f}"
    fps_size = cv2.getTextSize(fps_text, FONT, info_scale, info_thick)[0]
    fps_x = w - fps_size[0] - margin
    # Pastikan FPS tidak menabrak teks status
    if fps_x > status_size[0] + margin * 3:
        cv2.putText(frame, fps_text, (fps_x, row1_y),
                    FONT, info_scale, (180, 180, 180), info_thick, cv2.LINE_AA)

    # ── Baris 2: Sudut (kiri) + Durasi (kanan) ──
    row2_y = row1_y + _scale(25, h)

    # Mode deteksi (jika tersedia)
    mode_label = ""
    if detection_mode == "upper_body":
        mode_label = " [Upper Body]"
    elif detection_mode == "full_body":
        mode_label = " [Full Body]"

    angle_text = f"Sudut: {angle:.1f} deg{mode_label}"
    cv2.putText(frame, angle_text, (margin, row2_y),
                FONT, info_scale, COLOR_WHITE, info_thick, cv2.LINE_AA)

    # Timer durasi postur buruk (kanan baris 2)
    if elapsed > 0:
        timer_text = f"Durasi: {elapsed:.1f}s / 10s"
        timer_size = cv2.getTextSize(timer_text, FONT, info_scale, info_thick)[0]
        timer_x = w - timer_size[0] - margin
        cv2.putText(frame, timer_text, (timer_x, row2_y),
                    FONT, info_scale, COLOR_YELLOW, info_thick, cv2.LINE_AA)

    return frame


def draw_timer_bar(frame: np.ndarray, progress: float) -> np.ndarray:
    """
    Gambar progress bar timer postur buruk.

    Args:
        frame: Frame BGR
        progress: Nilai 0.0 hingga 1.0

    Returns:
        Frame dengan progress bar
    """
    if progress <= 0:
        return frame

    h, w = frame.shape[:2]
    panel_height = _scale(80, h)
    bar_height = max(3, _scale(6, h))
    bar_width = int(w * progress)

    # Warna gradasi: kuning → merah
    if progress < 0.5:
        color = COLOR_YELLOW
    elif progress < 0.8:
        color = (0, 140, 255)  # oranye
    else:
        color = COLOR_RED

    cv2.rectangle(frame, (0, panel_height), (bar_width, panel_height + bar_height), color, -1)

    return frame


def draw_calibration_overlay(frame: np.ndarray, remaining: float) -> np.ndarray:
    """
    Gambar overlay hitung mundur kalibrasi.
    Panel adaptif terhadap ukuran frame.

    Args:
        frame: Frame BGR
        remaining: Sisa waktu kalibrasi (detik)

    Returns:
        Frame dengan overlay kalibrasi
    """
    h, w = frame.shape[:2]

    # Panel adaptif — maksimum 80% lebar frame
    panel_w = min(_scale(420, h), int(w * 0.8))
    panel_h = _scale(160, h)
    x1 = (w - panel_w) // 2
    y1 = (h - panel_h) // 2
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    # Semi-transparan background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Border
    border_thick = max(1, _scale(2, h))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 180, 50), border_thick)

    # Ukuran font adaptif
    title_scale = _font_scale(0.7, h)
    sub_scale = _font_scale(0.45, h)
    timer_scale = _font_scale(1.0, h)
    title_thick = max(1, _scale(2, h))

    # Teks instruksi
    text1 = "KALIBRASI POSTUR"
    text2 = "Duduk tegak dan menghadap kamera"
    text3 = f"{remaining:.1f} detik"

    ts1 = cv2.getTextSize(text1, FONT_BOLD, title_scale, title_thick)[0]
    ts2 = cv2.getTextSize(text2, FONT, sub_scale, 1)[0]
    ts3 = cv2.getTextSize(text3, FONT_BOLD, timer_scale, title_thick)[0]

    # Pastikan teks muat dalam panel
    while ts1[0] > panel_w - 20 and title_scale > 0.3:
        title_scale -= 0.05
        ts1 = cv2.getTextSize(text1, FONT_BOLD, title_scale, title_thick)[0]

    while ts2[0] > panel_w - 20 and sub_scale > 0.25:
        sub_scale -= 0.05
        ts2 = cv2.getTextSize(text2, FONT, sub_scale, 1)[0]

    # Vertikal spacing dalam panel
    y_title = y1 + int(panel_h * 0.25)
    y_sub = y1 + int(panel_h * 0.50)
    y_timer = y1 + int(panel_h * 0.82)

    cv2.putText(frame, text1, ((w - ts1[0]) // 2, y_title),
                FONT_BOLD, title_scale, (200, 180, 50), title_thick, cv2.LINE_AA)
    cv2.putText(frame, text2, ((w - ts2[0]) // 2, y_sub),
                FONT, sub_scale, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, text3, ((w - ts3[0]) // 2, y_timer),
                FONT_BOLD, timer_scale, COLOR_WHITE, title_thick, cv2.LINE_AA)

    return frame


def draw_controls_hint(frame: np.ndarray, render_mode: str = "ASYNC (Mulus)") -> np.ndarray:
    """
    Gambar petunjuk kontrol di bagian bawah frame.

    Args:
        frame: Frame BGR
        render_mode: Label mode render aktif ("ASYNC (Mulus)" atau "SYNC (Presisi)")

    Returns:
        Frame dengan petunjuk kontrol
    """
    h, w = frame.shape[:2]

    hint = f"Q: Keluar  |  R: Kalibrasi Ulang  |  M: Mode [{render_mode}]"
    hint_scale = _font_scale(0.35, h)
    hint_thick = max(1, _scale(1, h))
    bar_h = _scale(25, h)

    ts = cv2.getTextSize(hint, FONT, hint_scale, hint_thick)[0]

    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, hint, ((w - ts[0]) // 2, h - _scale(8, h)),
                FONT, hint_scale, (180, 180, 180), hint_thick, cv2.LINE_AA)

    return frame

