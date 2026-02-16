@echo off
REM ============================================================
REM run_train.bat - Train Pothole Detection Model (Windows)
REM ============================================================
REM This script trains the YOLOv8 model on the MWPD dataset.
REM Make sure you've completed the setup steps in README.md first.
REM
REM Quick start: Just double-click this file or run from terminal.
REM ============================================================

echo.
echo ============================================================
echo   POTHOLE DETECTION - TRAINING
echo   GPU: RTX 3060 12GB Optimized
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo   Virtual environment activated.
) else (
    echo   No virtual environment found. Using system Python.
    echo   (Recommended: python -m venv venv)
)

REM Run training with default settings
python scripts/train.py %*

echo.
echo Training complete! Check the runs/train/ folder for results.
pause
