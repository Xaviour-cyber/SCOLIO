"""
overlay.py — Modul Gambar OpenCV (Skeleton, Teks Status, Modern Clinical HUD)

Menampilkan overlay skeleton modern bergaya cyber-klinis, panel status HUD
glassmorphism, logo resmi, gauge sudut, dan UI kalibrasi di atas feed video.
"""

import cv2
import numpy as np
import os
from .alert import PostureState


# ── Palet Warna Modern (BGR) ─────────────────────────────────
COLOR_EMERALD = (80, 215, 60)      # Hijau klinis cerah (Normal)
COLOR_AMBER = (0, 175, 255)        # Oranye / Kuning peringatan (Warning)
COLOR_CRIMSON = (50, 50, 240)      # Merah medis (Alert/High Risk)
COLOR_CYAN = (240, 180, 0)         # Cyan aksen sendi
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY_TEXT = (190, 195, 205)
COLOR_GLASS_BG = (18, 20, 24)      # Dark Slate Glass
COLOR_GLASS_BORDER = (55, 60, 72)
COLOR_BONE = (210, 185, 130)       # Tulang skeleton lembut anti-aliased

# ── Font ─────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ── Koneksi skeleton untuk overlay ────────────────────────────
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

# Cache logo
_logo_img = None
_logo_checked = False


def _get_logo():
    global _logo_img, _logo_checked
    if not _logo_checked:
        for p in [
            os.path.join("static", "img", "logo_transparent.png"),
            os.path.join("static", "img", "logo.png"),
        ]:
            if os.path.exists(p):
                _logo_img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                break
        _logo_checked = True
    return _logo_img


def _scale(value: float, frame_height: int, ref: int = 480) -> int:
    """Skalakan nilai berdasarkan tinggi frame relatif terhadap referensi 480px."""
    return max(1, int(value * frame_height / ref))


def _font_scale(base: float, frame_height: int, ref: int = 480) -> float:
    """Skalakan ukuran font berdasarkan tinggi frame."""
    return max(0.32, base * frame_height / ref)


def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple,
                      radius: int = 10, thickness: int = -1) -> np.ndarray:
    """Menggambar persegi panjang dengan sudut membulat (rounded corners)."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    radius = min(radius, w // 2, h // 2)

    if radius <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return img

    if thickness < 0:
        # Filled rounded rect
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        # Border rounded rect
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)

    return img


def overlay_alpha(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
    """Tempel citra transparan (RGBA) di atas background BGR dengan aman."""
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]

    if x >= w_bg or y >= h_bg or x + w_fg <= 0 or y + h_fg <= 0:
        return bg

    x1_bg, y1_bg = max(0, x), max(0, y)
    x2_bg, y2_bg = min(w_bg, x + w_fg), min(h_bg, y + h_fg)

    x1_fg, y1_fg = max(0, -x), max(0, -y)
    x2_fg = x1_fg + (x2_bg - x1_bg)
    y2_fg = y1_fg + (y2_bg - y1_bg)

    fg_crop = fg[y1_fg:y2_fg, x1_fg:x2_fg]
    bg_crop = bg[y1_bg:y2_bg, x1_bg:x2_bg]

    if fg.shape[2] == 4:
        alpha = (fg_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
        fg_bgr = fg_crop[:, :, :3]
        bg[y1_bg:y2_bg, x1_bg:x2_bg] = (alpha * fg_bgr + (1.0 - alpha) * bg_crop).astype(np.uint8)
    else:
        bg[y1_bg:y2_bg, x1_bg:x2_bg] = fg_crop

    return bg


def draw_skeleton(frame: np.ndarray, keypoints: dict, confidences: np.ndarray = None,
                  min_conf: float = 0.3) -> np.ndarray:
    """
    Gambar overlay skeleton bergaya sensor biomekanik modern.
    """
    all_kps = keypoints.get("all_keypoints")
    all_conf = keypoints.get("all_confidences")

    if all_kps is None:
        return frame

    h_frame = frame.shape[0]
    line_thick = max(1, _scale(2, h_frame))
    spine_thick = max(2, _scale(3, h_frame))
    joint_r = _scale(5, h_frame)

    overlay = frame.copy()

    # 1. Gambar tulang tubuh (soft anti-aliased)
    for (i, j) in SKELETON_CONNECTIONS:
        if all_conf is not None and (all_conf[i] < min_conf or all_conf[j] < min_conf):
            continue
        pt1 = (int(all_kps[i][0]), int(all_kps[i][1]))
        pt2 = (int(all_kps[j][0]), int(all_kps[j][1]))
        cv2.line(overlay, pt1, pt2, COLOR_BONE, line_thick, cv2.LINE_AA)

    # 2. Gambar tulang belakang vertikal (neck → mid_hip)
    neck = keypoints.get("neck")
    mid_hip = keypoints.get("mid_hip")
    if neck and mid_hip:
        pt_neck = (int(neck[0]), int(neck[1]))
        pt_hip = (int(mid_hip[0]), int(mid_hip[1]))
        # Spine line bercahaya
        cv2.line(overlay, pt_neck, pt_hip, COLOR_EMERALD, spine_thick, cv2.LINE_AA)

    # 3. Garis nose → neck
    nose = keypoints.get("nose")
    if nose and neck:
        pt_nose = (int(nose[0]), int(nose[1]))
        pt_neck = (int(neck[0]), int(neck[1]))
        cv2.line(overlay, pt_nose, pt_neck, COLOR_EMERALD, spine_thick, cv2.LINE_AA)

    # 4. Titik Sendi (High-Tech Concentric Dots)
    for i in range(len(all_kps)):
        if all_conf is not None and all_conf[i] < min_conf:
            continue
        pt = (int(all_kps[i][0]), int(all_kps[i][1]))
        # Lingkaran luar putih halus
        cv2.circle(overlay, pt, joint_r + 1, COLOR_WHITE, 1, cv2.LINE_AA)
        # Lingkaran dalam warna cyan/amber
        cv2.circle(overlay, pt, joint_r, COLOR_CYAN, -1, cv2.LINE_AA)

    # Blend transparan
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame


def draw_status_panel(frame: np.ndarray, state: PostureState,
                      angle: float, elapsed: float,
                      fps: float = 0.0,
                      detection_mode: str = "") -> np.ndarray:
    """
    Gambar panel status di bagian atas frame.
    Ukuran font dan layout adaptif terhadap dimensi frame.
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
        color = COLOR_EMERALD
        status_text = "POSTUR: NORMAL"
    elif state == PostureState.WARNING:
        color = COLOR_AMBER
        status_text = "! PERBAIKI POSTUR"
    elif state == PostureState.ALERT:
        color = COLOR_CRIMSON
        status_text = "!! PERBAIKI POSTUR!"
    elif state == PostureState.CALIBRATING:
        color = (200, 180, 50)
        status_text = "KALIBRASI..."
    else:
        color = COLOR_WHITE
        status_text = "UNKNOWN"

    # Panel background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), COLOR_GLASS_BG, -1)
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

    angle_text = f"Sudut: {angle:.1f} deg"
    cv2.putText(frame, angle_text, (margin, row2_y),
                FONT, info_scale, COLOR_WHITE, info_thick, cv2.LINE_AA)

    # Timer durasi postur buruk (kanan baris 2)
    if elapsed > 0:
        timer_text = f"Durasi: {elapsed:.1f}s / 10s"
        timer_size = cv2.getTextSize(timer_text, FONT, info_scale, info_thick)[0]
        timer_x = w - timer_size[0] - margin
        cv2.putText(frame, timer_text, (timer_x, row2_y),
                    FONT, info_scale, COLOR_AMBER, info_thick, cv2.LINE_AA)

    return frame


def draw_timer_bar(frame: np.ndarray, progress: float) -> np.ndarray:
    """
    Gambar progress bar timer postur buruk.
    """
    if progress <= 0:
        return frame

    h, w = frame.shape[:2]
    panel_height = _scale(80, h)
    bar_height = max(3, _scale(6, h))
    bar_width = int(w * progress)

    # Warna gradasi: kuning → merah
    if progress < 0.5:
        color = COLOR_AMBER
    elif progress < 0.8:
        color = (0, 140, 255)  # oranye
    else:
        color = COLOR_CRIMSON

    cv2.rectangle(frame, (0, panel_height), (bar_width, panel_height + bar_height), color, -1)

    return frame


def draw_calibration_overlay(frame: np.ndarray, remaining: float) -> np.ndarray:
    """
    Gambar overlay hitung mundur kalibrasi.
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
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GLASS_BG, -1)
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


def draw_controls_hint(frame: np.ndarray, render_mode: str = "SYNC (Presisi)") -> np.ndarray:
    """
    Gambar petunjuk kontrol di bagian bawah frame.
    """
    h, w = frame.shape[:2]

    hint = "Q: Keluar  |  R: Kalibrasi Ulang"
    hint_scale = _font_scale(0.35, h)
    hint_thick = max(1, _scale(1, h))
    bar_h = _scale(25, h)

    ts = cv2.getTextSize(hint, FONT, hint_scale, hint_thick)[0]

    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_GLASS_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, hint, ((w - ts[0]) // 2, h - _scale(8, h)),
                FONT, hint_scale, (180, 180, 180), hint_thick, cv2.LINE_AA)

    return frame
