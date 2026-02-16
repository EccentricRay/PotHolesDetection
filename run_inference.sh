#!/bin/bash
# ============================================================
# run_inference.sh - Run Pothole Detection (Linux/Mac)
# ============================================================
# Usage: ./run_inference.sh test_media/road.jpg
#        ./run_inference.sh test_media/drone_video.mp4
#        ./run_inference.sh 0    (webcam)
# ============================================================

echo ""
echo "============================================================"
echo "  POTHOLE DETECTION - INFERENCE"
echo "============================================================"
echo ""

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODEL_PATH="runs/train/pothole_detection/weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "  ERROR: Trained model not found at $MODEL_PATH"
    echo "  Please train the model first: ./run_train.sh"
    exit 1
fi

if [ -z "$1" ]; then
    echo "  Usage: ./run_inference.sh <input_path>"
    echo ""
    echo "  Examples:"
    echo "    ./run_inference.sh test_media/image.jpg"
    echo "    ./run_inference.sh test_media/video.mp4"
    echo "    ./run_inference.sh test_media/          (folder)"
    echo "    ./run_inference.sh 0                    (webcam)"
    exit 1
fi

python scripts/inference.py --model "$MODEL_PATH" --input "$1" --show "${@:2}"
