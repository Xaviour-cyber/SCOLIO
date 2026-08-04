"""
detector.py — Modul Deteksi Pose YOLO & Ekstraksi Keypoint

Memuat model YOLOv26-Pose Nano dan mengekstrak keypoint anatomi
dari setiap frame video untuk analisis postur tulang belakang.
"""

from ultralytics import YOLO
import numpy as np

# Peta index keypoint COCO (17 titik)
# Ref: https://docs.ultralytics.com/tasks/pose/
COCO_KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Keypoint yang kita butuhkan untuk analisis postur
POSTURE_KEYPOINTS = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]

# Threshold confidence minimum untuk keypoint
MIN_CONFIDENCE = 0.5


class PoseDetector:
    """Detektor pose menggunakan YOLOv26-Pose Nano."""

    def __init__(self, model_path: str = "yolo26n-pose.pt", confidence: float = 0.5):
        """
        Inisialisasi detektor pose.

        Args:
            model_path: Path ke file model YOLO pose (.pt)
            confidence: Threshold confidence minimum untuk deteksi
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Jalankan deteksi pose pada satu frame.

        Args:
            frame: Frame BGR dari OpenCV (numpy array)

        Returns:
            Dictionary berisi keypoint terstruktur, atau None jika tidak terdeteksi.
            Format:
            {
                "nose": (x, y),
                "neck": (x, y),           # Diturunkan dari bahu
                "left_shoulder": (x, y),
                "right_shoulder": (x, y),
                "left_hip": (x, y),
                "right_hip": (x, y),
                "mid_hip": (x, y),         # Titik tengah pinggul
                "all_keypoints": np.array,  # Semua 17 keypoints mentah
                "all_confidences": np.array # Semua 17 confidence scores
            }
        """
        # Jalankan inferensi (verbose=False untuk menghindari log spam)
        results = self.model.predict(
            source=frame,
            device="cpu",
            conf=self.confidence,
            verbose=False,
        )

        # Ambil hasil pertama
        result = results[0]

        # Periksa apakah ada deteksi
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Ambil keypoints dari orang pertama yang terdeteksi (paling besar/dekat)
        # Shape: (num_persons, 17, 3) → [x, y, confidence]
        keypoints_data = result.keypoints.data

        if len(keypoints_data) == 0:
            return None

        # Pilih orang dengan bounding box terbesar (paling dekat ke kamera)
        if result.boxes is not None and len(result.boxes) > 0:
            areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * \
                    (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
            best_idx = int(areas.argmax())
        else:
            best_idx = 0

        person_kps = keypoints_data[best_idx].cpu().numpy()  # Shape: (17, 3)

        coords = person_kps[:, :2]       # (17, 2) — x, y
        confidences = person_kps[:, 2]   # (17,)   — confidence

        # Validasi keypoint yang kita butuhkan
        extracted = {}
        valid = True

        for name in POSTURE_KEYPOINTS:
            idx = COCO_KEYPOINT_INDICES[name]
            conf = confidences[idx]

            if conf < MIN_CONFIDENCE:
                valid = False
                break

            extracted[name] = (float(coords[idx][0]), float(coords[idx][1]))

        if not valid:
            return None

        # Turunkan posisi "leher" = titik tengah kedua bahu
        lsh = extracted["left_shoulder"]
        rsh = extracted["right_shoulder"]
        neck = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
        extracted["neck"] = neck

        # Turunkan "mid_hip" = titik tengah kedua pinggul
        lhip = extracted["left_hip"]
        rhip = extracted["right_hip"]
        mid_hip = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
        extracted["mid_hip"] = mid_hip

        # Simpan data mentah juga
        extracted["all_keypoints"] = coords
        extracted["all_confidences"] = confidences

        return extracted
