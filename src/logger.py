"""
logger.py — Modul Perekaman Data & Pembuatan Grafik Otomatis

Merekam data postur ke file CSV selama sesi berlangsung,
lalu secara otomatis menghasilkan grafik dan ringkasan
saat sesi selesai.

Output:
  - CSV individual per sesi (data detail per-frame)
  - CSV master kumulatif (ringkasan semua sesi)
  - Line chart (sudut vs waktu)
  - Scatter plot (persebaran data berwarna per status)
  - Scatter plot gabungan seluruh sesi (jika multi-sesi)
"""

import csv
import os
import time
from datetime import datetime


# Folder output
LOGS_DIR = "logs"


class SessionLogger:
    """Mencatat data postur ke CSV dan menghasilkan grafik otomatis."""

    def __init__(self, student_id: str = "", screen_time_min: float = 0.0):
        """
        Args:
            student_id: ID/Nama siswa (contoh: "S-01")
            screen_time_min: Durasi screen time harian siswa (menit)
        """
        self.student_id = student_id or "unknown"
        self.screen_time_min = screen_time_min

        # Buat folder logs jika belum ada
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Nama file berbasis waktu
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = self.student_id.replace(" ", "_").replace("/", "-")
        self._session_name = f"session_{safe_id}_{ts}"
        self._csv_path = os.path.join(LOGS_DIR, f"{self._session_name}.csv")
        self._master_csv = os.path.join(LOGS_DIR, "all_sessions.csv")

        # Buka file CSV dan tulis header
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp_sec", "angle_deg", "state", "detection_mode"
        ])

        # Throttle: maksimum 2 log per detik
        self._last_log_time = 0.0
        self._log_interval = 0.5  # detik

        # Timer sesi
        self._session_start = time.time()

        # Data in-memory untuk ringkasan cepat
        self._data_rows = []

        print(f"[LOG] Perekaman data dimulai → {self._csv_path}")

    def log(self, angle: float, state: str, detection_mode: str = ""):
        """
        Catat satu baris data. Otomatis di-throttle ke 2x/detik.

        Args:
            angle: Sudut deviasi relatif (derajat)
            state: Status postur (NORMAL/WARNING/ALERT)
            detection_mode: Mode deteksi (full_body/upper_body)
        """
        now = time.time()
        if now - self._last_log_time < self._log_interval:
            return  # Skip — belum waktunya

        self._last_log_time = now
        elapsed = round(now - self._session_start, 2)

        row = [elapsed, round(angle, 2), state, detection_mode]
        self._writer.writerow(row)
        self._data_rows.append(row)

    def close_and_report(self):
        """
        Tutup file CSV dan hasilkan grafik + ringkasan.
        Dipanggil saat program ditutup.
        """
        # Tutup file CSV
        self._file.close()

        if len(self._data_rows) < 3:
            print("[LOG] Data terlalu sedikit untuk membuat grafik.")
            return

        print(f"[LOG] Sesi selesai. Total {len(self._data_rows)} data points.")
        print("[LOG] Membuat grafik...")

        try:
            self._generate_charts()
            self._append_to_master()
            self._generate_master_scatter()
        except Exception as e:
            print(f"[LOG] Error saat membuat grafik: {e}")

    def _generate_charts(self):
        """Buat line chart dan scatter plot dari data sesi ini."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Parse data
        times = [row[0] for row in self._data_rows]
        angles = [row[1] for row in self._data_rows]
        states = [row[2] for row in self._data_rows]

        color_map = {
            "NORMAL": "#2ecc71",
            "WARNING": "#f39c12",
            "ALERT": "#e74c3c",
            "CALIBRATING": "#3498db",
        }

        # ── Grafik 1: Line Chart (BERSIH, tanpa titik-titik) ──
        fig, ax = plt.subplots(figsize=(10, 5))

        # Gambar garis per segmen, diwarnai sesuai status
        for i in range(len(times) - 1):
            c = color_map.get(states[i], "#95a5a6")
            ax.plot(times[i:i+2], angles[i:i+2], color=c, linewidth=2.0,
                    solid_capstyle="round")

        # Garis batas threshold
        ax.axhline(y=15, color="#e74c3c", linestyle="--", alpha=0.6,
                    label="Ambang Batas (15°)")

        # Area warna (transparan)
        ax.fill_between(times, 0, angles,
                         where=[a <= 15 for a in angles],
                         color="#2ecc71", alpha=0.08, interpolate=True)
        ax.fill_between(times, 0, angles,
                         where=[a > 15 for a in angles],
                         color="#e74c3c", alpha=0.08, interpolate=True)

        # Legend manual untuk warna status
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color="#2ecc71", lw=3, label="Normal"),
            Line2D([0], [0], color="#f39c12", lw=3, label="Warning"),
            Line2D([0], [0], color="#e74c3c", lw=3, label="Alert"),
            Line2D([0], [0], color="#e74c3c", lw=1, ls="--", label="Ambang Batas (15°)"),
        ]
        ax.legend(handles=legend_items, loc="upper left", fontsize=9)

        ax.set_xlabel("Waktu (detik)", fontsize=11)
        ax.set_ylabel("Sudut Deviasi Relatif (°)", fontsize=11)
        ax.set_title(
            f"Perubahan Sudut Postur Terhadap Waktu — {self.student_id}",
            fontsize=13, fontweight="bold"
        )
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path1 = os.path.join(LOGS_DIR, f"{self._session_name}_linechart.png")
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"[LOG] Line chart → {path1}")

        # ── Grafik 2: Scatter Plot (1 titik per 5 detik) ─────
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        # Sample data: ambil 1 titik setiap ~5 detik agar tidak terlalu padat
        sample_interval = 5.0  # detik
        sampled_t, sampled_a, sampled_s = [], [], []
        last_sample = -sample_interval
        for i in range(len(times)):
            if times[i] - last_sample >= sample_interval:
                sampled_t.append(times[i])
                sampled_a.append(angles[i])
                sampled_s.append(states[i])
                last_sample = times[i]

        for state_name, color in color_map.items():
            mask = [s == state_name for s in sampled_s]
            t = [sampled_t[i] for i in range(len(mask)) if mask[i]]
            a = [sampled_a[i] for i in range(len(mask)) if mask[i]]
            if t:
                ax2.scatter(t, a, c=color, s=60, label=state_name,
                            alpha=0.8, edgecolors="black", linewidths=0.8,
                            zorder=5)

        ax2.axhline(y=15, color="#e74c3c", linestyle="--", alpha=0.6,
                     label="Ambang Batas (15°)")

        ax2.set_xlabel("Waktu (detik)", fontsize=11)
        ax2.set_ylabel("Sudut Deviasi Relatif (°)", fontsize=11)
        ax2.set_title(
            f"Persebaran Data Postur — {self.student_id}",
            fontsize=13, fontweight="bold"
        )
        ax2.legend(loc="upper left", fontsize=9)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        path2 = os.path.join(LOGS_DIR, f"{self._session_name}_scatter.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"[LOG] Scatter plot → {path2}")

    def _append_to_master(self):
        """Tambahkan ringkasan sesi ini ke master CSV."""
        if not self._data_rows:
            return

        angles = [row[1] for row in self._data_rows]
        states = [row[2] for row in self._data_rows]
        total = len(states)
        duration = self._data_rows[-1][0]  # timestamp terakhir

        avg_angle = sum(angles) / len(angles)
        max_angle = max(angles)

        pct_normal = states.count("NORMAL") / total * 100
        pct_warning = states.count("WARNING") / total * 100
        pct_alert = states.count("ALERT") / total * 100

        # Cek apakah master CSV sudah ada (perlu header?)
        need_header = not os.path.exists(self._master_csv)

        with open(self._master_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow([
                    "student_id", "screen_time_min", "duration_sec",
                    "avg_angle", "max_angle",
                    "pct_normal", "pct_warning", "pct_alert",
                    "detection_mode", "date"
                ])
            # Mode yang paling sering digunakan
            modes = [row[3] for row in self._data_rows if row[3]]
            dominant_mode = max(set(modes), key=modes.count) if modes else "unknown"

            writer.writerow([
                self.student_id,
                self.screen_time_min,
                round(duration, 1),
                round(avg_angle, 2),
                round(max_angle, 2),
                round(pct_normal, 1),
                round(pct_warning, 1),
                round(pct_alert, 1),
                dominant_mode,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ])

        # Print ringkasan ke terminal
        print()
        print("=" * 50)
        print(f"  RINGKASAN SESI — {self.student_id}")
        print("=" * 50)
        print(f"  Durasi Tes       : {duration:.1f} detik")
        print(f"  Screen Time/Hari : {self.screen_time_min:.0f} menit")
        print(f"  Rata-rata Sudut  : {avg_angle:.1f}°")
        print(f"  Sudut Maksimal   : {max_angle:.1f}°")
        print(f"  % Normal         : {pct_normal:.1f}%")
        print(f"  % Warning        : {pct_warning:.1f}%")
        print(f"  % Alert          : {pct_alert:.1f}%")
        print("=" * 50)
        print()

    def _generate_master_scatter(self):
        """Baca all_sessions.csv dan buat scatter plot gabungan semua siswa."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as csv_module

        if not os.path.exists(self._master_csv):
            return

        # Baca master CSV
        students = []
        with open(self._master_csv, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                students.append(row)

        if len(students) < 2:
            print("[LOG] Belum cukup data siswa untuk scatter plot gabungan (min. 2).")
            return

        # Parse data
        screen_times = [float(s["screen_time_min"]) for s in students]
        avg_angles = [float(s["avg_angle"]) for s in students]
        labels = [s["student_id"] for s in students]

        # Warna berdasarkan rata-rata sudut
        colors = []
        for a in avg_angles:
            if a < 10:
                colors.append("#2ecc71")   # hijau = aman
            elif a < 15:
                colors.append("#f39c12")   # kuning = hati-hati
            else:
                colors.append("#e74c3c")   # merah = berisiko

        # Buat scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(screen_times, avg_angles, c=colors, s=100,
                   edgecolors="black", linewidths=0.8, alpha=0.8, zorder=5)

        # Label setiap titik
        for i, label in enumerate(labels):
            ax.annotate(label, (screen_times[i], avg_angles[i]),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=8, alpha=0.7)

        # Garis batas
        ax.axhline(y=15, color="#e74c3c", linestyle="--", alpha=0.5,
                    label="Ambang Batas Postur (15°)")

        ax.set_xlabel("Durasi Screen Time Harian (Menit)", fontsize=12)
        ax.set_ylabel("Rata-rata Sudut Deviasi Postur (°)", fontsize=12)
        ax.set_title(
            "Korelasi Screen Time Harian terhadap Kemiringan Postur Siswa",
            fontsize=13, fontweight="bold"
        )
        ax.legend(loc="upper left")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(LOGS_DIR, "scatter_all_sessions.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[LOG] Scatter plot gabungan ({len(students)} siswa) → {path}")
