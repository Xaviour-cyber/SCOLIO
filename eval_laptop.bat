@echo off
title Scolio-Scan - Evaluasi & Benchmark Sistem
echo ================================================================
echo   SCOLIO-SCAN: AUTO EVALUATION & BENCHMARK SUITE
echo   Pengujian Paritas Numerik (MAE) & Benchmark Hardware
echo ================================================================
echo.

set PYTHON_EXE=

REM 1. Cek venv khusus think
if exist "C:\Users\think\scolio-venv\Scripts\python.exe" (
    set PYTHON_EXE="C:\Users\think\scolio-venv\Scripts\python.exe"
    echo [INFO] Terdeteksi lingkungan: scolio-venv
) else if exist ".venv\Scripts\python.exe" (
    REM 2. Cek venv lokal proyek (standar laptop teman)
    set PYTHON_EXE=".venv\Scripts\python.exe"
    echo [INFO] Terdeteksi lingkungan lokal: .venv
) else (
    REM 3. Cek Python global sistem
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_EXE=python
        echo [INFO] Terdeteksi Python Sistem
    ) else (
        echo [ERROR] Python tidak ditemukan!
        echo         Pastikan virtual environment .venv sudah dibuat via setup.bat
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Memeriksa dependensi ONNX Runtime...
%PYTHON_EXE% -c "import onnxruntime" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] onnxruntime belum terpasang. Menginstall otomatis...
    %PYTHON_EXE% -m pip install "onnxruntime>=1.18.0"
    if %errorlevel% neq 0 (
        echo [ERROR] Gagal menginstall onnxruntime. Cek koneksi internet.
        pause
        exit /b 1
    )
    echo [OK] onnxruntime berhasil terpasang.
) else (
    echo [OK] onnxruntime siap digunakan.
)

echo.
echo [INFO] Memulai proses evaluasi dan pengujian performa...
echo ----------------------------------------------------------------
%PYTHON_EXE% eval_system.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Terjadi kendala saat menjalankan evaluasi.
) else (
    echo.
    echo ================================================================
    echo [SUKSES] Pengujian selesai! Laporan telah tersimpan di folder logs/
    echo ================================================================
)

echo.
pause
