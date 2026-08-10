@echo off
echo [INFO] Menjalankan Scolio-Scan...
echo.

set PYTHON_EXE=
set DEV_ENV= partner

REM Cek venv khusus (di luar OneDrive untuk laptop Anda)
if exist "C:\Users\think\scolio-venv\Scripts\python.exe" (
    set PYTHON_EXE="C:\Users\think\scolio-venv\Scripts\python.exe"
    echo [INFO] Menggunakan venv khusus: C:\Users\think\scolio-venv
) else if exist ".venv\Scripts\python.exe" (
    REM Cek venv lokal (untuk laptop partner)
    set PYTHON_EXE=".venv\Scripts\python.exe"
    echo [INFO] Menggunakan venv lokal: .venv
) else (
    echo [ERROR] Virtual environment belum di-setup!
    echo         Jalankan setup.bat terlebih dahulu.
    pause
    exit /b 1
)

%PYTHON_EXE% run.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program keluar dengan error.
    pause
)


echo.
pause
