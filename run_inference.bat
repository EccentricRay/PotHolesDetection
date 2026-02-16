@echo off
REM ============================================================
REM run_inference.bat - Run Pothole Detection (Windows)
REM ============================================================
REM Runs the trained model on images, videos, or webcam.
REM
REM USAGE:
REM   run_inference.bat                      (prompts for input)
REM   run_inference.bat test_media\road.jpg  (specific file)
REM   run_inference.bat test_media\video.mp4 (video file)
REM   run_inference.bat 0                    (webcam)
REM ============================================================

echo.
echo ============================================================
echo   POTHOLE DETECTION - INFERENCE
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Find the best model weights
set MODEL_PATH=runs\train\pothole_detection\weights\best.pt

if not exist "%MODEL_PATH%" (
    echo   ERROR: Trained model not found at %MODEL_PATH%
    echo   Please train the model first: run_train.bat
    pause
    exit /b 1
)

REM Check if input argument was provided
if "%~1"=="" (
    echo   No input specified.
    echo.
    echo   Usage examples:
    echo     run_inference.bat test_media\image.jpg
    echo     run_inference.bat test_media\video.mp4
    echo     run_inference.bat test_media\             (folder of images)
    echo     run_inference.bat 0                       (webcam)
    echo.
    set /p INPUT_PATH="  Enter path to image/video (or 0 for webcam): "
) else (
    set INPUT_PATH=%~1
)

echo.
echo   Model: %MODEL_PATH%
echo   Input: %INPUT_PATH%
echo.

python scripts/inference.py --model %MODEL_PATH% --input %INPUT_PATH% --show

pause
