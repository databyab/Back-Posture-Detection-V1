"""
FastAPI backend for Posture Detection
Provides REST endpoints for image and video processing
"""

import io
import os
import logging
import tempfile
import hashlib
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.posture_model import PostureDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Posture Detection API",
    description="Real-time posture detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware - restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Initialize model
try:
    model_path = "models/yolov8n-pose.pt"
    detector = PostureDetector(model_path=model_path, threshold=160.0)
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None


# Security constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    base_name = os.path.basename(filename)
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    _, ext = os.path.splitext(base_name)
    return f"{file_hash}{ext}"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None
    }


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict posture from an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, BMP, WebP)
    
    Returns:
        JSON with posture detection results
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed: JPG, PNG, BMP, WebP")
    
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail=f"Image too large. Max: {MAX_IMAGE_SIZE // 1024 // 1024} MB")
    
    try:
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect posture
        result = detector.detect(frame)
        
        if not result["success"]:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": result.get("error", "Detection failed")}
            )
        
        # Encode annotated frame to bytes
        _, buffer = cv2.imencode('.jpg', result["annotated_frame"])
        annotated_base64 = buffer.tobytes()
        
        return {
            "success": True,
            "posture": result["posture"],
            "keypoints": {
                "shoulder": [float(x) for x in result["keypoints"]["shoulder"]],
                "hip": [float(x) for x in result["keypoints"]["hip"]],
                "knee": [float(x) for x in result["keypoints"]["knee"]]
            } if result["keypoints"] else None
        }
    
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/image/annotated")
async def predict_image_annotated(file: UploadFile = File(...)):
    """
    Predict posture and return annotated image.
    
    Args:
        file: Image file (JPG, PNG, BMP, WebP)
    
    Returns:
        Annotated image with detection overlays
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type. Allowed: JPG, PNG, BMP, WebP")
    
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail=f"Image too large. Max: {MAX_IMAGE_SIZE // 1024 // 1024} MB")
    
    try:
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect posture
        result = detector.detect(frame)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Detection failed"))
        
        # Encode annotated frame
        _, buffer = cv2.imencode('.jpg', result["annotated_frame"])
        
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg"
        )
    
    except Exception as e:
        logger.error(f"Image annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Analyze posture in a video file.
    
    Args:
        file: Video file (MP4, AVI, MOV, MKV)
    
    Returns:
        JSON with video-level statistics
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid video type. Allowed: MP4, AVI, MOV, MKV")
    
    try:
        # Read video file
        contents = await file.read()
        if len(contents) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=413, detail=f"Video too large. Max: {MAX_VIDEO_SIZE // 1024 // 1024} MB")
        
        # Create secure temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            os.write(temp_fd, contents)
            os.close(temp_fd)
            
            # Process video
            stats = detector.process_video(temp_path)
            
            if not stats.get("success"):
                raise HTTPException(status_code=400, detail=stats.get("error", "Video processing failed"))
            
            return {
                "success": True,
                "statistics": stats
            }
        finally:
            # Ensure temp file cleanup
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        raise HTTPException(status_code=500, detail="Video processing failed")


@app.get("/api/config")
async def get_config():
    """Get current detector configuration"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "threshold": detector.threshold,
        "model_path": "models/yolov8n-pose.pt"
    }


@app.post("/api/config")
async def update_config(threshold: float = 160.0):
    """Update detector configuration"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not 100 <= threshold <= 180:
        raise HTTPException(status_code=400, detail="Threshold must be between 100 and 180")
    
    detector.threshold = threshold
    return {"success": True, "threshold": detector.threshold}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
