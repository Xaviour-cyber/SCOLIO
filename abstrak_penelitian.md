# Abstrak Penelitian — Scolio-Scan

## Abstrak

Skoliosis merupakan gangguan muskuloskeletal dengan prevalensi 2–5% pada populasi remaja usia 10–15 tahun secara global. Di Indonesia, studi di Surabaya (2010) mencatat prevalensi *Adolescent Idiopathic Scoliosis* sebesar 2,93% pada siswa usia 9–16 tahun, sementara penelitian di Jakarta menunjukkan angka kecurigaan skoliosis mencapai 7% pada siswa sekolah dasar. Meningkatnya durasi *screen time* dan kebiasaan postur duduk yang buruk pada era digital memperburuk risiko kelainan tulang belakang pada remaja. Metode skrining konvensional seperti *Adam's Forward Bending Test* memiliki keterbatasan berupa subjektivitas pemeriksa dan tidak dapat dilakukan secara kontinu. Penelitian ini mengembangkan **Scolio-Scan**, sebuah sistem monitoring postur tulang belakang berbasis *computer vision* menggunakan model YOLOv26-Pose Nano untuk estimasi pose secara *real-time*. Sistem mengekstraksi 17 *keypoints* anatomis melalui kamera atau rekaman video, menghitung sudut deviasi tulang belakang relatif terhadap posisi kalibrasi, dan memberikan peringatan bertingkat (*Normal*, *Warning*, *Alert*) ketika kemiringan melebihi ambang batas 15°. Hasil pengujian menunjukkan bahwa sistem mampu mendeteksi perubahan postur secara responsif dengan antarmuka visual *real-time* yang informatif, menjadikannya solusi skrining awal yang aksesibel dan non-invasif untuk pencegahan gangguan postur pada remaja.

---

> [!NOTE]
> **Jumlah kata: ~170 kata** (di bawah batas maksimal 200 kata)

## Kata Kunci
`Skoliosis`, `Postur Tulang Belakang`, `Computer Vision`, `YOLO Pose Estimation`, `Real-time Monitoring`, `Skrining Kesehatan`

---

## Referensi Pendukung yang Digunakan

| No | Sumber | Detail |
|----|--------|--------|
| 1 | Studi AIS Surabaya (2010) | Prevalensi 2,93% pada siswa usia 9–16 tahun (sumber: FK Universitas Airlangga / RSUD Dr. Soetomo) |
| 2 | Studi Skoliosis Jakarta | Prevalensi kecurigaan skoliosis 7% pada siswa SD (sumber: Universitas Katolik Atma Jaya) |
| 3 | Scoliosis Research Society (SRS) | Prevalensi global skoliosis idiopatik remaja 2–4% |
| 4 | Ultralytics / YOLO | YOLOv26-Pose — model pose estimation real-time berbasis deep learning (ultralytics.com) |
| 5 | Riset Ergonomi Duduk + YOLO | YOLOv8-based sitting posture detection untuk pencegahan gangguan muskuloskeletal (rcf-indonesia.org, 2024) |
| 6 | Dampak Screen Time | Korelasi durasi penggunaan gawai dengan nyeri punggung dan risiko skoliosis pada remaja |

> [!IMPORTANT]
> Tabel referensi di atas adalah **ringkasan sumber** untuk keperluan penulisan abstrak. Untuk makalah final, setiap referensi perlu dilengkapi dengan format sitasi penuh (APA/IEEE) beserta DOI/URL.
