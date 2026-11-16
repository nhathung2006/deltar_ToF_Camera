@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow README.md in the project root.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u -m apps.live_depth --camera 1 --port COM9 --baud 921600 --sigma-m 0.02 --extrinsics configs/calibration/rgb_tof_extrinsics_25mm.json --smoothing combined --guided-radius 48 --no-edge-aware-tof --view comparison %*
set "DELTAR_EXIT_CODE=%ERRORLEVEL%"
if not "%DELTAR_EXIT_CODE%"=="0" pause
exit /b %DELTAR_EXIT_CODE%
