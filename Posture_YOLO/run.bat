@echo off
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py --source 0