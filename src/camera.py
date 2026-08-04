"""
camera.py — Modul Pengambilan Webcam dengan Threading

Thread terpisah membaca frame dari webcam secara terus-menerus,
sehingga main thread dapat mengambil frame terbaru tanpa blocking.
Mendukung webcam lokal (index) dan IP camera (URL DroidCam, dll).
"""

import cv2
import threading
import time


class ThreadedCamera:
    """
    Webcam capture dengan thread terpisah untuk performa optimal.

    Thread background terus membaca frame dari kamera.
    Main thread mengambil frame terbaru via get_frame() (non-blocking).

    Mendukung:
    - Webcam lokal: source = 0, 1, 2, ...
    - IP Camera / DroidCam: source = "http://192.168.x.x:4747/video"
    """

    def __init__(self, source=0, width: int = 640, height: int = 480):
        """
        Inisialisasi kamera dengan threading.

        Args:
            source: Index kamera (int) atau URL stream (str)
            width: Lebar resolusi frame (hanya untuk webcam lokal)
            height: Tinggi resolusi frame (hanya untuk webcam lokal)
        """
        self._source = source
        self._is_url = isinstance(source, str)

        print(f"[INFO] Menghubungkan ke: {source}")
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Tidak dapat membuka kamera (source={source}). "
                "Pastikan webcam/DroidCam terhubung dan IP sudah benar."
            )

        # Set resolusi hanya untuk webcam lokal (bukan URL stream)
        if not self._is_url:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Buffer kecil untuk mengurangi delay pada IP camera
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Baca frame pertama
        self._ret = False
        self._frame = None

        # Coba beberapa kali untuk IP camera (bisa lambat connect)
        for attempt in range(10):
            self._ret, self._frame = self.cap.read()
            if self._ret:
                break
            print(f"[INFO] Menunggu frame... (percobaan {attempt + 1}/10)")
            time.sleep(0.5)

        if not self._ret:
            self.cap.release()
            raise RuntimeError(
                "Gagal membaca frame dari kamera. "
                "Pastikan DroidCam aktif di HP dan IP benar."
            )

        print(f"[OK] Stream aktif — resolusi: {self._frame.shape[1]}x{self._frame.shape[0]}")

        # Lock untuk thread safety
        self._lock = threading.Lock()

        # Start thread background
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Loop utama thread: baca frame terus-menerus."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._ret = ret
                    self._frame = frame
            else:
                # Jika gagal baca, tunggu sebentar sebelum retry
                time.sleep(0.03)

    def get_frame(self):
        """
        Ambil frame terbaru (non-blocking).

        Returns:
            Tuple (success: bool, frame: np.ndarray)
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        """Hentikan thread dan lepaskan kamera."""
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        """Cleanup otomatis saat objek dihapus."""
        try:
            self.release()
        except Exception:
            pass
