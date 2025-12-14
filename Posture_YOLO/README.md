# YOLOv8 Posture Detection


## Setup
1. Create a Python venv (recommended) on Python 3.10+ or 3.13.
2. Install requirements: `pip install -r requirements.txt`
3. Download the YOLOv8 pose model `yolov8n-pose.pt` and place it in `models/`.
- You can download from Ultralytics model zoo or use `ultralytics` to fetch.



## Run
- For webcam realtime: `python app.py --source 0`
- For image test: `python app.py --source assets/sample.png`# YOLOv8 Posture Detection


## Setup
1. Create a Python venv (recommended) on Python 3.10+ or 3.13.
2. Install requirements: `pip install -r requirements.txt`
3. Download the YOLOv8 pose model `yolov8n-pose.pt` and place it in `models/`.
- You can download from Ultralytics model zoo or use `ultralytics` to fetch.
4. (Optional) Put a sample image in `assets/sample.png` or use the included sample at `/mnt/data/25d81cb9-dac0-4074-88ef-54392d805ddb.png`.


## Run
- For webcam realtime: `python app.py --source 0`
- For image test: `python app.py --source assets/sample.png`