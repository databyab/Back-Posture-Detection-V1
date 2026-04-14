#!/usr/bin/env python3
"""
Quick start verification script
Checks if all components are installed and working
"""

import sys
import importlib
from pathlib import Path

def check_import(package_name, friendly_name=None):
    """Check if a package is installed"""
    friendly_name = friendly_name or package_name
    try:
        importlib.import_module(package_name)
        print(f"✅ {friendly_name}")
        return True
    except ImportError:
        print(f"❌ {friendly_name} - NOT INSTALLED")
        return False

def check_file(filepath):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("🧍 Posture Detection - Verification Script")
    print("=" * 60)
    
    print("\n📦 Checking Python Packages...")
    packages = [
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv8"),
        ("numpy", "NumPy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("streamlit", "Streamlit"),
    ]
    
    all_packages_ok = True
    for pkg, name in packages:
        if not check_import(pkg, name):
            all_packages_ok = False
    
    print("\n📁 Checking Project Structure...")
    files = [
        "core/posture_model.py",
        "core/__init__.py",
        "app/main.py",
        "app/__init__.py",
        "frontend/streamlit_app.py",
        "frontend/__init__.py",
        "webcam.py",
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
    ]
    
    all_files_ok = True
    for filepath in files:
        if not check_file(filepath):
            all_files_ok = False
    
    print("\n📋 Checking Model Files...")
    if not Path("models/yolov8n-pose.pt").exists():
        print("❌ models/yolov8n-pose.pt - NOT FOUND")
        print("   💡 Download with: python -c \"from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt')\"")
        all_files_ok = False
    else:
        print("✅ models/yolov8n-pose.pt")
    
    print("\n" + "=" * 60)
    
    if all_packages_ok and all_files_ok:
        print("✅ Everything is set up! Ready to run:")
        print("\n   🎥 Local webcam (original):")
        print("      python webcam.py")
        print("\n   🌐 Web dashboard (new):")
        print("      streamlit run frontend/streamlit_app.py")
        print("\n   📡 REST API (new):")
        print("      uvicorn app.main:app --reload")
        print("\n" + "=" * 60)
        return 0
    else:
        if not all_packages_ok:
            print("❌ Missing packages. Install with:")
            print("   pip install -r requirements.txt")
        if not all_files_ok:
            print("❌ Missing files. Check project structure.")
        print("\n" + "=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
