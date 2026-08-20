@echo off
echo ==================================================
echo   SCOLIO-SCAN - Setup Otomatis
echo   Instalasi Virtual Environment dan Dependensi
echo ==================================================
echo.

REM Cek apakah Python terinstall
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python belum terinstall!
    echo         Download di: https://www.python.org/downloads/
    echo         Pastikan centang "Add Python to PATH" saat install.
    pause
    exit /b 1
)

echo [INFO] Python ditemukan:
python --version
echo.

REM Buat virtual environment
echo [INFO] Membuat virtual environment...
if exist .venv (
    echo [INFO] Folder .venv sudah ada, menghapus yang lama...
    rmdir /s /q .venv
)
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Gagal membuat virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment berhasil dibuat.
echo.

REM Aktivasi dan install dependensi
echo [INFO] Menginstall dependensi (ini bisa memakan waktu 5-15 menit)...
echo [INFO] Pastikan koneksi internet stabil.
echo.
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install Flask
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Gagal menginstall dependensi!
    echo         Pastikan koneksi internet stabil dan coba lagi.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   SETUP SELESAI!
echo ==================================================
echo.
echo   Cara menjalankan Scolio-Scan:
echo     1. Buka terminal/cmd di folder ini
echo     2. Ketik: .venv\Scripts\activate
echo     3. Ketik: python run.py
echo.
echo   Atau langsung jalankan: run_app.bat
echo.
echo.
pause
