"""
Setup configuration for Posture Detection package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="posture-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade posture detection system using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/posture-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "streamlit>=1.28.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "posture-webcam=webcam:main",
        ],
    },
)
