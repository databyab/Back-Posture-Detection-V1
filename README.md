# Posture Detection

Real-time posture detection using YOLOv8. Detects shoulder, hip, and knee keypoints to calculate back angle and classify posture as GOOD (≥160°) or BAD (<160°).

## Setup

```bash
# Clone and install
git clone https://github.com/databyab/Back-Posture-Detection-V1.git
cd Back-Posture-Detection-V1

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

The YOLOv8 model downloads automatically on first run.

## Usage

### Web Dashboard
```bash
streamlit run frontend/streamlit_app.py
```
Upload images/videos, view real-time posture analysis, adjust threshold slider.

Access at http://localhost:8501

### Local Webcam
```bash
python webcam.py
```
Real-time detection from webcam. Press `q` to quit, `s` to save, `r` to reset stats.

Options:
```bash
python webcam.py --threshold 160 --camera 0 --width 1280 --height 720
```

### REST API
```bash
uvicorn app.main:app --reload
```
Access docs at http://localhost:8000/docs

**Endpoints:**
- `GET /health` - Health check
- `POST /api/predict/image` - Analyze image
- `POST /api/predict/image/annotated` - Get annotated image
- `POST /api/predict/video` - Analyze video file
- `GET /api/config` - Get current threshold
- `POST /api/config?threshold=165` - Update threshold

## Architecture

**Core Module** (`core/posture_model.py`)
- Single source of truth for all interfaces
- Handles YOLOv8 inference, keypoint extraction, angle calculation

**Interfaces**
- `frontend/streamlit_app.py` - Web dashboard
- `app/main.py` - REST API backend
- `webcam.py` - Local CLI mode

## Requirements

- Python 3.8+
- YOLOv8 (ultralytics)
- OpenCV, FastAPI, Streamlit, NumPy

Full list in `requirements.txt`

## Model

Uses YOLOv8 nano pretrained on COCO with pose estimation. Model auto-downloads to `models/yolov8n-pose.pt` (55MB).

For custom models:
```python
detector = PostureDetector(model_path="path/to/custom_model.pt")
```

## Performance

- **CPU:** ~50-100ms per frame
- **GPU:** ~10-20ms per frame
- Threshold: 100-180 degrees (default 160)

## Testing

```bash
python verify_setup.py
```
Checks all dependencies and project structure.

## File Structure

```
core/
  posture_model.py      # Detection engine
app/
  main.py               # FastAPI REST API
frontend/
  streamlit_app.py      # Web dashboard
webcam.py              # Webcam runner
requirements.txt       # Dependencies
setup.py               # Package config
verify_setup.py        # Environment check
```

