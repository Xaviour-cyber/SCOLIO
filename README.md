# Scolio-Scan 👨‍💻🩺

> **Scolio-Scan** is a real-time computer vision-based spinal posture monitoring system. Utilizing the YOLOv26-Pose Nano model, it tracks 17 anatomical keypoints to calculate spinal deviation angles. Designed to prevent musculoskeletal disorders among students, it provides tiered visual alerts (Normal, Warning, Alert) when slouching is detected for extended periods.

---

## 📌 Apa itu Scolio-Scan?
Scolio-Scan adalah proyek karya ilmiah yang bertujuan untuk mendeteksi kebiasaan postur duduk yang buruk (seperti membungkuk) yang berisiko memicu kelainan tulang belakang (Skoliosis/Kifosis) pada remaja akibat *screen time* yang berlebihan.

Sistem ini berjalan **sepenuhnya secara lokal** menggunakan CPU komputer (tanpa perlu internet) dan dioptimalkan dengan *Multi-Threading* sehingga video tetap berjalan mulus meskipun AI sedang memproses data di latar belakang.

## ✨ Fitur Utama
- **YOLOv26-Pose Nano:** AI ringan yang mampu mendeteksi kerangka tubuh manusia secara akurat meski hanya menggunakan CPU standar.
- **Deteksi Sudut Tulang Belakang:** Menghitung sudut kemiringan secara matematis dari titik leher (antara dua bahu) hingga titik tengah pinggul.
- **Peringatan Bertingkat (Alert System):** 
  - 🟢 **Normal (Aman)**: Postur tegak (< 15°).
  - 🟡 **Warning**: Terdeteksi mulai membungkuk, bar timer akan berjalan.
  - 🔴 **Alert**: Membungkuk melebihi 10 detik, peringatan merah menyala.
- **Threaded Inference:** Arsitektur yang memisahkan pemrosesan AI dengan tampilan UI, menghasilkan FPS video yang sangat mulus (15-20+ FPS).

## 🚀 Cara Instalasi (Untuk Pengguna Windows)

Sangat mudah! Tidak perlu mengetik perintah rumit, cukup jalankan *script* otomatis yang telah disediakan.

1. **Pastikan Python sudah terinstall:**
   Buka terminal/CMD dan ketik `python --version`. Jika belum ada, download dari [python.org](https://www.python.org/) (pastikan centang *Add Python to PATH* saat instalasi).
2. Buka folder proyek ini.
3. Klik ganda (Double-click) pada file **`setup.bat`**.
4. Tunggu beberapa saat (5-15 menit) hingga proses instalasi AI dan pembuatan *virtual environment* selesai.

## 🎮 Cara Menjalankan Aplikasi

Jika instalasi sudah selesai, Anda hanya perlu melakukan langkah ini setiap kali ingin menggunakan aplikasi:

1. Klik ganda pada file **`run_app.bat`**.
2. Aplikasi akan terbuka. **Duduklah dengan tegak dan menghadap kamera** selama 3 detik pertama untuk proses kalibrasi.
3. Setelah kalibrasi selesai, AI akan mulai memantau postur Anda secara *real-time*!
4. **Kontrol Keyboard:**
   - Tekan `R` untuk Kalibrasi Ulang (jika posisi kursi Anda berubah).
   - Tekan `Q` untuk Menutup aplikasi.

---
*Dibuat untuk keperluan Penelitian dan Karya Ilmiah - 2026.*
