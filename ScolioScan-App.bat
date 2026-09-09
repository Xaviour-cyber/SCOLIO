@echo off
echo =======================================
echo    MENGHIDUPKAN SCOLIO-SCAN AI...
echo =======================================
echo.
echo Menyiapkan Engine AI di latar belakang...
start wscript.exe launcher.vbs

echo Menunggu Server siap...
ping 127.0.0.1 -n 4 > nul

echo Membuka Aplikasi...
start chrome.exe --app=http://127.0.0.1:5000
   