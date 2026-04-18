@echo off
echo ========================================
echo   Sign Language To Text Conversion
echo ========================================
echo.
echo Starting enhanced application...
echo.

python Application.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo Application encountered an error!
    echo ========================================
    echo.
    echo Please check:
    echo 1. Camera is connected and working
    echo 2. All model files are in Models/ folder
    echo 3. Python dependencies are installed
    echo.
) else (
    echo.
    echo Application closed successfully.
)

pause
