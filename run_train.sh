#!/bin/bash
# ============================================================
# run_train.sh - Train Pothole Detection Model (Linux/Mac)
# ============================================================
# Usage: ./run_train.sh
#        ./run_train.sh --epochs 50 --batch 8
# ============================================================

echo ""
echo "============================================================"
echo "  POTHOLE DETECTION - TRAINING"
echo "  GPU: RTX 3060 12GB Optimized"
echo "============================================================"
echo ""

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "  Virtual environment activated."
fi

# Run training (pass any extra arguments through)
python scripts/train.py "$@"

echo ""
echo "Training complete! Check the runs/train/ folder for results."
