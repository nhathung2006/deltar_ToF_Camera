@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow README.md in the project root.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u -m apps.check_camera_tof --port COM9 --baud 921600 --show-serial %*
set "DELTAR_EXIT_CODE=%ERRORLEVEL%"
if not "%DELTAR_EXIT_CODE%"=="0" pause
exit /b %DELTAR_EXIT_CODE%
