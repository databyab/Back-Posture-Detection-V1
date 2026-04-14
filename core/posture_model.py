"""
Core Posture Detection Model
Handles YOLOv8-based pose estimation and posture classification
"""

import math
import logging
from typing import Tuple, Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PostureDetector:
    """
    Detects human posture using YOLOv8 pose estimation.
    Calculates back angle and classifies posture as GOOD or BAD.
    """
    
    # YOLOv8 keypoint indices (17 keypoints model)
    LEFT_SHOULDER = 5
    LEFT_HIP = 11
    LEFT_KNEE = 13
    RIGHT_SHOULDER = 6
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    
    def __init__(self, model_path: str = "models/yolov8n-pose.pt", threshold: float = 160.0):
        """
        Initialize the PostureDetector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            threshold: Back angle threshold for GOOD posture (degrees)
        """
        try:
            self.model = YOLO(model_path)
            self.threshold = threshold
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def calculate_angle(a: Tuple[float, float], 
                       b: Tuple[float, float], 
                       c: Tuple[float, float]) -> float:
        """
        Calculate angle ABC (in degrees) where B is the vertex.
        
        Args:
            a: Point A coordinates (x, y)
            b: Point B coordinates (x, y) - vertex
            c: Point C coordinates (x, y)
        
        Returns:
            Angle in degrees (0-180)
        """
        ax, ay = a
        bx, by = b
        cx, cy = c
        
        # Vectors from B to A and B to C
        ab = (ax - bx, ay - by)
        cb = (cx - bx, cy - by)
        
        # Dot product and magnitudes
        dot = ab[0] * cb[0] + ab[1] * cb[1]
        mag_ab = math.hypot(ab[0], ab[1])
        mag_cb = math.hypot(cb[0], cb[1])
        
        # Avoid division by zero
        if mag_ab == 0 or mag_cb == 0:
            return 0.0
        
        # Cosine formula
        cos_angle = dot / (mag_ab * mag_cb)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Numerical safety
        
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def classify_posture(self, angle: float) -> Dict[str, any]:
        """
        Classify posture based on back angle.
        
        Args:
            angle: Back angle in degrees
        
        Returns:
            Dictionary with status and details
        """
        status = "GOOD" if angle >= self.threshold else "BAD"
        
        return {
            "status": status,
            "angle": round(angle, 1),
            "threshold": self.threshold,
            "confidence": min(100, abs(angle - self.threshold) / self.threshold * 100)
        }
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect posture in a single frame.
        
        Args:
            frame: Input image (BGR format from OpenCV)
        
        Returns:
            Dictionary with detection results
        """
        try:
            # Run inference
            results = self.model(frame)
            
            output = {
                "success": False,
                "frame": frame.copy(),
                "annotated_frame": results[0].plot(),
                "keypoints": None,
                "posture": None,
                "angle": None,
                "error": None
            }
            
            # Extract keypoints if person detected
            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                output["error"] = "No person detected"
                return output
            
            # Get first person's keypoints
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Extract key joints
            shoulder = tuple(kpts[self.LEFT_SHOULDER])
            hip = tuple(kpts[self.LEFT_HIP])
            knee = tuple(kpts[self.LEFT_KNEE])
            
            # Skip if any keypoint is invalid (0, 0)
            if any(np.allclose(pt, [0, 0]) for pt in [shoulder, hip, knee]):
                output["error"] = "Incomplete keypoints detected"
                return output
            
            # Calculate back angle
            angle = self.calculate_angle(shoulder, hip, knee)
            posture_info = self.classify_posture(angle)
            
            # Annotate frame
            annotated = self._annotate_frame(
                output["annotated_frame"],
                shoulder, hip, knee,
                angle, posture_info["status"]
            )
            
            output.update({
                "success": True,
                "annotated_frame": annotated,
                "keypoints": {
                    "shoulder": shoulder,
                    "hip": hip,
                    "knee": knee
                },
                "posture": posture_info
            })
            
            return output
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                "success": False,
                "frame": frame.copy(),
                "annotated_frame": frame.copy(),
                "error": str(e)
            }
    
    @staticmethod
    def _annotate_frame(frame: np.ndarray,
                       shoulder: Tuple[int, int],
                       hip: Tuple[int, int],
                       knee: Tuple[int, int],
                       angle: float,
                       status: str) -> np.ndarray:
        """
        Add visual annotations to frame.
        
        Args:
            frame: Input frame
            shoulder, hip, knee: Keypoint coordinates
            angle: Back angle
            status: Posture status (GOOD/BAD)
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color based on posture
        color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)
        
        # Draw keypoint circles
        for point in [shoulder, hip, knee]:
            cv2.circle(annotated, (int(point[0]), int(point[1])), 6, (255, 0, 0), -1)
        
        # Draw skeleton line
        cv2.line(annotated, (int(shoulder[0]), int(shoulder[1])),
                (int(hip[0]), int(hip[1])), (0, 255, 255), 2)
        cv2.line(annotated, (int(hip[0]), int(hip[1])),
                (int(knee[0]), int(knee[1])), (0, 255, 255), 2)
        
        # Display angle
        cv2.putText(annotated, f"Back Angle: {angle:.1f}°", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display posture status
        cv2.putText(annotated, f"Posture: {status}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return annotated
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Detection results
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return {"success": False, "error": "Could not read image file"}
            
            return self.detect(frame)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def process_video(self, video_path: str, callback=None) -> Dict:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to video file
            callback: Optional callback function called for each frame
                     Should accept (frame_num, result) and return True to continue
        
        Returns:
            Summary of video processing
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Could not open video file"}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            good_count = 0
            bad_count = 0
            angles = []
            frame_num = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.detect(frame)
                
                if result["success"]:
                    good_count += 1 if result["posture"]["status"] == "GOOD" else 0
                    bad_count += 1 if result["posture"]["status"] == "BAD" else 0
                    angles.append(result["posture"]["angle"])
                
                frame_num += 1
                
                # Call callback if provided
                if callback:
                    if not callback(frame_num, result):
                        break
            
            cap.release()
            
            return {
                "success": True,
                "total_frames": frame_num,
                "duration_seconds": frame_count / fps if fps > 0 else 0,
                "good_posture_frames": good_count,
                "bad_posture_frames": bad_count,
                "good_posture_percentage": (good_count / frame_num * 100) if frame_num > 0 else 0,
                "average_angle": sum(angles) / len(angles) if angles else 0
            }
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {"success": False, "error": str(e)}
