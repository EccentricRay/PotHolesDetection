# Pothole Detection on Drone Video Footage (YOLOv8)

A streamlined pothole detection pipeline using YOLOv8, optimized for **RTX 3060 (12GB VRAM)**. Trained on the [Multi-Weather Pothole Detection (MWPD)](https://www.kaggle.com/datasets/jocelyndumlao/multi-weather-pothole-detection-mwpd/data) dataset. Designed for drone video footage analysis — GPS integration will come later.

Inspired by [yolo-training-template](https://github.com/computer-vision-with-marco/yolo-training-template), stripped down to only what you need for pothole detection.

---

## Project Structure

```
pothole-detection/
├── configs/
│   ├── pothole_dataset.yaml      # Dataset paths & class definition
│   └── training_config.yaml      # Hyperparameters (GPU-optimized)
├── scripts/
│   ├── quick_test.py             # Verify your setup works
│   ├── train.py                  # Training pipeline
│   ├── inference.py              # Run detection on images/video
│   └── validate.py               # Evaluate model metrics
├── data/                         # Dataset (auto-downloaded)
├── runs/                         # Training outputs & results
├── test_media/                   # Put test images/videos here
├── run_train.bat / .sh           # One-click training scripts
├── run_inference.bat / .sh       # One-click inference scripts
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## Step-by-Step Setup Guide

### Step 1: Install Python & Create Virtual Environment

Open a terminal (or VSCode terminal: `` Ctrl+` ``) and navigate to the project folder:

```bash
cd pothole-detection

# Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
```

### Step 2: Install PyTorch with CUDA Support

This is the most important step. You need PyTorch compiled for your GPU.

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and select:
- OS: Windows / Linux
- Package: pip
- Language: Python
- Compute Platform: **CUDA 12.x** (match your NVIDIA driver)

Then run the generated command. Example for CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **How to check your CUDA version:** Run `nvidia-smi` in terminal. Look at the "CUDA Version" in the top-right corner.

### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Kaggle API (for dataset download)

The training script downloads the MWPD dataset automatically from Kaggle. You need an API key:

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in (or create a free account)
2. Click your profile icon (top-right) → **Settings**
3. Scroll to **API** section → Click **"Create New Token"**
4. This downloads a `kaggle.json` file
5. Move it to the right location:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

> **Alternative:** If you prefer, download the dataset manually from the Kaggle page and extract it to `data/MWPD/`. The script will detect it and skip downloading.

### Step 5: Verify Your Setup

Run the setup checker to make sure everything works:

```bash
python scripts/quick_test.py
```

This checks: Python packages, GPU/CUDA, YOLO loading, Kaggle API, and project files. Fix any issues it reports before proceeding.

---

## Training

### Quick Start (recommended defaults)

```bash
# Windows:
run_train.bat

# Linux/Mac:
chmod +x run_train.sh
./run_train.sh

# Or directly:
python scripts/train.py
```

This will:
1. Download the MWPD dataset from Kaggle (~500MB)
2. Auto-detect the dataset structure
3. Train YOLOv8m for 100 epochs with early stopping
4. Save the best model to `runs/train/pothole_detection/weights/best.pt`

### Custom Training Options

```bash
# Quick test run (10 epochs, smaller batch)
python scripts/train.py --epochs 10 --batch 8

# Use a faster but less accurate model
python scripts/train.py --model yolov8s.pt

# Use a more accurate model (may need batch=8 to fit in VRAM)
python scripts/train.py --model yolov8l.pt --batch 8

# Resume a training run that was interrupted
python scripts/train.py --resume runs/train/pothole_detection/weights/last.pt

# Skip download if dataset is already there
python scripts/train.py --skip-download
```

### If You Get "Out of Memory" (OOM) Errors

Your RTX 3060 has 12GB VRAM. If training crashes with CUDA OOM:

```bash
# Option 1: Reduce batch size
python scripts/train.py --batch 8

# Option 2: Use a smaller model
python scripts/train.py --model yolov8s.pt

# Option 3: Both
python scripts/train.py --model yolov8s.pt --batch 8
```

### Monitor Training with TensorBoard

While training is running, open another terminal:

```bash
tensorboard --logdir runs/train
```

Then open `http://localhost:6006` in your browser to see live training curves.

---

## Running Inference (Detection)

After training, use the model to detect potholes:

### On a Single Image

```bash
python scripts/inference.py \
    --model runs/train/pothole_detection/weights/best.pt \
    --input test_media/road_photo.jpg
```

### On Drone Video

```bash
python scripts/inference.py \
    --model runs/train/pothole_detection/weights/best.pt \
    --input test_media/drone_flight.mp4
```

### On a Folder of Images

```bash
python scripts/inference.py \
    --model runs/train/pothole_detection/weights/best.pt \
    --input test_media/
```

### Live Webcam

```bash
python scripts/inference.py \
    --model runs/train/pothole_detection/weights/best.pt \
    --input 0 --show
```

### Tuning Detection Sensitivity

```bash
# More sensitive (catches faint potholes, but more false positives)
python scripts/inference.py --model best.pt --input video.mp4 --conf 0.15

# More strict (fewer false alarms, may miss some potholes)
python scripts/inference.py --model best.pt --input video.mp4 --conf 0.5
```

Results are saved to `runs/inference/pothole_results/`.

---

## Evaluating the Model

Run formal evaluation on the test set:

```bash
python scripts/validate.py \
    --model runs/train/pothole_detection/weights/best.pt

# Evaluate on validation set instead
python scripts/validate.py \
    --model runs/train/pothole_detection/weights/best.pt \
    --split val
```

This outputs: **mAP@50**, **mAP@50-95**, **Precision**, **Recall**, plus confusion matrices and PR curves in `runs/validate/`.

---

## Understanding the Key Files

### `configs/training_config.yaml` — What to Tweak

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `model` | `yolov8m.pt` | Model size. `n`=nano, `s`=small, `m`=medium, `l`=large, `x`=extra |
| `epochs` | `100` | How many times to train over the full dataset |
| `batch` | `16` | Images per training step. Lower if OOM |
| `imgsz` | `640` | Input resolution. Higher = more accurate, more VRAM |
| `amp` | `true` | Mixed precision. **Keep this ON** for 3060 |
| `patience` | `20` | Stop early if no improvement for N epochs |
| `flipud` | `0.2` | Vertical flip probability (good for aerial/drone views) |

### `configs/pothole_dataset.yaml` — Dataset Config

This tells YOLO where images and labels are. The `path` field gets auto-updated by the training script. You only have **1 class**: `pothole` (index 0).

---

## GPU Optimization Notes (RTX 3060)

The project includes several optimizations for your hardware:

- **Mixed Precision (AMP):** Enabled by default. Uses FP16 where possible, cutting VRAM usage by ~40% and boosting speed.
- **TF32:** Automatically enabled for Ampere architecture (RTX 30xx). Gives ~2x speedup on matrix operations with minimal accuracy loss.
- **FP16 Inference:** The inference script uses `half=True` on GPU for faster processing.
- **Batch Size 16:** Sweet spot for YOLOv8m on 12GB VRAM at 640px. Drop to 8 if you use a larger model.
- **Stream Mode:** Video inference processes frame-by-frame instead of loading everything into memory.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Lower `--batch` to 8 or use `yolov8s.pt` |
| `No module named 'kaggle'` | `pip install kaggle` |
| `kaggle.json not found` | See Step 4 above |
| `No GPU detected` | Check CUDA install: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"` |
| Training is very slow | Make sure `amp: true` in config and GPU is being used (check with `nvidia-smi` during training) |
| Poor detection results | Train for more epochs, try `yolov8l.pt`, or add more training data |
| `train/images not found` | Dataset structure issue. Check that `data/MWPD/train/images/` exists |

---

## What's Next (Future Additions)

- GPS data integration with detection coordinates
- Real-time drone feed processing
- Pothole severity classification
- Export to ONNX/TensorRT for edge deployment
