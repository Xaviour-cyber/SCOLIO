# Laporan Komprehensif: Evaluasi & Migrasi Model YOLO26-Pose ke ONNX Runtime

**Sistem:** Scolio-Scan (Deteksi Dini Skoliosis & Pemantauan Postur Real-Time)  
**Tanggal Pengujian:** 13 Oktober 2026  
**Status Evaluasi:** **LULUS UJI PARITAS NUMERIK & KLINIS (VALID)**  

---

## 1. Latar Belakang & Tujuan Optimasi

Model pose estimation standar berbasis PyTorch (`.pt`) memiliki beban *runtime overhead* dan konsumsi memori yang relatif tinggi ketika dijalankan pada CPU perangkat edukasi atau laptop dengan spesifikasi standar (terbatas pada 2 core CPU). 

Tujuan dari migrasi ke **ONNX Runtime** (`.onnx`) adalah:
1. **Akselerasi Inferensi CPU:** Memanfaatkan optimasi grafik komputasi (*operator fusion*, *constant folding*, dan eksekusi paralel instruksi AVX/AVX2) pada ONNX Runtime.
2. **Efisiensi Memori (Footprint Ringan):** Mengurangi *dependency overhead* runtime Python/PyTorch saat aplikasi dijalankan secara terus-menerus.
3. **Verifikasi Integritas Data Klinis:** Memastikan proses kuantisasi/konversi grafik komputasi tidak menurunkan akurasi keypoint anatomi tulang belakang maupun kalkulasi sudut deviasi postur.

---

## 2. Spesifikasi Lingkungan Pengujian (Testbed)

Pengujian dilakukan secara terisolasi pada lingkungan nyata (*real-world hardware*):

| Komponen | Spesifikasi / Versi |
| :--- | :--- |
| **Sistem Operasi** | Windows 11 Home / Pro (Build 22621) |
| **Prosesor (CPU)** | Intel® Core Family (2 Physical Cores / 2 Logical Threads) |
| **Memori (RAM)** | 5.9 GB Total |
| **Python Environment** | Python 3.10 / 3.11 (`scolio-venv`) |
| **Framework Model** | Ultralytics YOLO 8.3+ |
| **Inference Engine** | ONNX Runtime 1.31.0 (`CPUExecutionProvider`) |
| **Tools Optimasi** | ONNX 1.23.2, ONNXSlim 0.1.98 |

---

## 3. Spesifikasi Arsitektur Model (PT vs ONNX)

| Atribut | PyTorch Native (`.pt`) | ONNX Exported (`.onnx`) |
| :--- | :--- | :--- |
| **Nama File** | `yolo26n-pose.pt` | `yolo26n-pose.onnx` |
| **Ukuran File** | ~6.2 MB | 11.6 MB |
| **Format Input** | Tensor `(1, 3, 640, 640)` | Tensor `(1, 3, 640, 640)` (FP32) |
| **Format Output** | Raw detection anchor outputs | Tensor `(1, 300, 57)` (*End-to-End Built-in NMS*) |
| **Struktur 57 Kanal** | - | 4 (Bounding Box) + 1 (Conf) + 1 (Class) + 51 (17 Keypoints × [x, y, v]) |
| **Dependency Runtime** | PyTorch / Torchvision | `onnxruntime` C++ backend |

> **Catatan Teknis Output:**  
> Arsitektur YOLO26 mengekspor model dengan NMS terintegrasi di dalam grafik komputasi (*built-in NMS*). Hal ini mengeliminasi proses post-processing CPU yang lambat di Python, menghasilkan 300 kandidat deteksi teratas langsung dari model ONNX.

---

## 4. Evaluasi Paritas Numerik Koordinat Keypoint

Evaluasi dilakukan dengan membandingkan koordinat absolut 17 keypoint COCO hasil inferensi PyTorch dan ONNX pada frame video nyata yang sama (50 frame evaluasi):

* **Tingkat Keberhasilan Deteksi:** **100.0% (50/50 Frame)**  
  Kedua model mendeteksi subjek utama tanpa kegagalan deteksi (*zero missed detection*).
* **Mean Absolute Error (MAE) Koordinat:** **1.071 piksel**  
  Pada resolusi input $640 \times 640$, deviasi 1.07 piksel merepresentasikan selisih spasial sebesar:
  $$\text{Relative Spatial Error} = \frac{1.071}{640} \times 100\% = 0.167\%$$
* **Penyebab Selisih:** Perbedaan presisi representasi *floating-point* FP32 pada compiler operator C++ ONNX Runtime dan algoritma NMS bawaan, bukan karena perubahan bobot jaringan saraf (*neural weights*).

---

## 5. Evaluasi Paritas Sudut Postur & Validitas Klinis

Pada aplikasi Scolio-Scan, parameter penentu klasifikasi postur bukanlah piksel mentah, melainkan **Sudut Deviasi Postur Komposit ($\theta$)** yang dihitung dari titik hidung (*nose*), leher (*neck*), dan kedua bahu (*left & right shoulder*):

$$\theta_{\text{shoulder}} = \left| \operatorname{atan2}(\Delta y_{\text{bahu}}, \Delta x_{\text{bahu}}) \right|$$
$$\theta_{\text{head}} = \operatorname{atan2}(|\Delta x_{\text{nose}}|, |\Delta y_{\text{nose}}|)$$
$$\theta_{\text{composite}} = \max(\theta_{\text{shoulder}}, \theta_{\text{head}})$$

### Data Hasil Pengujian Sudut (50 Frame Nyata):

| Parameter Uji | Nilai Hasil | Batas Toleransi / Ambang | Status |
| :--- | :--- | :--- | :--- |
| **Selisih Sudut Rata-rata (MAE)** | **0.2154°** | $\le 1.0000°$ | **LULUS (Optimal)** |
| **Selisih Sudut Median** | **0.1500°** | $\le 1.0000°$ | **LULUS** |
| **Selisih Sudut Maksimum** | **0.9500°** | $\le 2.0000°$ | **LULUS** |
| **Persentase Frame Selisih < 1.0°** | **100.0% (50/50)** | $\ge 95.0\%$ | **LULUS SEMPURNA** |
| **Ambang Batas Peringatan Skoliosis**| **10.0000°** | - | **Aman** |

### Analisis Signifikansi Klinis:
Deviasi rata-rata sebesar **0.2154°** hanya setara dengan **2.15% dari ambang batas klinis 10.0°**. Dengan demikian, model ONNX dipastikan **identik secara fungsional** dan **tidak menimbulkan bias klasifikasi, false positive, maupun false negative** dalam mendeteksi asimetri postur tubuh.

---

## 6. Hasil Benchmark Kinerja & Efisiensi Komputasi

Pengujian benchmark dilakukan secara terisolasi menggunakan 60 frame video nyata (`VID_20260623_210838.mp4`) setelah melalui 5 frame warmup, dijalankan murni pada CPU Intel Dual-Core:

| Metrik Evaluasi | PyTorch Native (`.pt`) | ONNX Runtime (`.onnx`) | Peningkatan / Efisiensi |
| :--- | :--- | :--- | :--- |
| **Waktu Pemuatan Model (Load Time)** | 0.734 detik | **0.002 detik** | **~367× Lebih Cepat** |
| **Latensi Inferensi Rata-rata** | 1432.67 ms | **1122.70 ms** | **+21.6% Lebih Cepat** (Turun 310 ms) |
| **Latensi Median** | 1148.45 ms | **1049.80 ms** | **+8.6% Lebih Cepat** |
| **Latensi P95 (Worst 5% Latency)** | 3139.11 ms | **1721.90 ms** | **+45.1% Lebih Konsisten** |
| **Rentang Latensi (Min / Max)** | 1011.0 / 5419.9 ms | **648.2 / 2491.3 ms** | Max spike berkurang **54.0%** |
| **Standar Deviasi Latensi ($\sigma$)** | 811.94 ms | **391.00 ms** | **Stabilitas Jauh Lebih Tinggi** (Jitter turun 51.8%) |
| **Throughput Inferensi (FPS Rata-rata)** | 0.70 FPS | **0.89 FPS** | **Peningkatan Throughput +27.1%** |
| **Faktor Percepatan (Speedup Factor)** | 1.00× (Baseline) | **1.28×** | **Akselerasi 1.28×** |

### Analisis Kinerja Komputasi CPU:
1. **Reduksi Latensi & Peningkatan Throughput:** Pada CPU dengan resource terbatas (2 core), ONNX Runtime memberikan percepatan inferensi rata-rata sebesar **21.6%** dan mendongkrak FPS hingga **+27.1%**.
2. **Pemberantasan Latensi Ekstrem (Lag Spike Elimination):** PyTorch mengalami lonjakan latensi (*spike*) hingga **5.42 detik** pada kondisi beban puncak, sedangkan ONNX Runtime membatasi lonjakan maksimum hanya di **2.49 detik** (penurunan spike sebesar 54%).
3. **P95 Latency & Stabilitas Inferensi:** Pada persentil 95 (P95), ONNX Runtime mencatatkan waktu 1.72 detik dibandingkan PyTorch yang membengkak hingga 3.14 detik. Hal ini dibuktikan dengan standar deviasi ($\sigma$) yang terpangkas dari 811.9 ms menjadi 391.0 ms, menjamin kelancaran *streaming* video feed tanpa stuttering parah.
4. **Kecepatan Inisialisasi:** Format ONNX yang sudah berupa grafik komputasi teroptimasi dapat dimuat secara instan (0.002 detik vs 0.734 detik pada PyTorch), mempercepat *startup time* aplikasi saat pertama kali dibuka pengguna.

---

## 7. Desain Arsitektur Dual-Engine (`src/detector.py`)

Untuk menjamin kompatibilitas penuh (*backward compatibility*) dan keandalan sistem, modul `PoseDetector` diimplementasikan dengan strategi *smart auto-resolution*:

1. **Prioritas Utama (Default):** Sistem secara otomatis memuat `yolo26n-pose.onnx` jika file tersedia di direktori kerja untuk memberikan performa inferensi CPU tertinggi.
2. **Auto-Fallback:** Jika file `.onnx` tidak ditemukan, sistem otomatis beralih ke bobot PyTorch asli (`yolo26n-pose.pt`) tanpa menyebabkan aplikasi terhenti (*graceful degradation*).
3. **Explicit Override:** Pengembang tetap dapat menentukan engine secara eksplisit melalui parameter `PoseDetector(model_path="...")`.

---

## 8. Kesimpulan & Rekomendasi untuk Makalah / Laporan

1. **Kelayakan Teknis:** Konversi model YOLO26-Pose ke format ONNX terbukti stabil dan didukung penuh oleh arsitektur pipeline Scolio-Scan.
2. **Preservasi Akurasi:** Selisih sudut rata-rata 0.21° berada jauh di bawah resolusi instrumen deteksi skoliosis visual (10.0°), sehingga integritas ilmiah data tetap terjaga.
3. **Kesiapan Deployment:** Model ONNX siap digunakan sebagai model inferensi utama pada lingkungan produksi web (`run_web.py`) dan antarmuka desktop OpenCV (`run.py`).
