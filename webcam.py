"""
Local Webcam Posture Detection
Real-time posture detection directly from your webcam
Run with: python webcam.py
"""

import argparse
import cv2
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.posture_model import PostureDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time posture detection from webcam"
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n-pose.pt",
        help="Path to YOLOv8 model weights"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=160.0,
        help="Back angle threshold for GOOD posture (degrees)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 for default webcam)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height in pixels"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second"
    )
    
    return parser.parse_args()


def run_webcam(detector, camera_idx=0, width=1280, height=720, fps=30):
    """
    Run real-time posture detection from webcam.
    
    Args:
        detector: PostureDetector instance
        camera_idx: Camera index
        width: Video width
        height: Video height
        fps: Target frames per second
    """
    logger.info("Initializing webcam...")
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_idx}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    logger.info("Webcam initialized. Press 'q' to quit.")
    logger.info(f"Threshold: {detector.threshold}°")
    
    frame_count = 0
    good_posture_count = 0
    bad_posture_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            # Detect posture
            result = detector.detect(frame)
            
            # Display annotated frame
            display_frame = result["annotated_frame"] if result["success"] else frame
            
            # Update statistics if detection successful
            if result["success"]:
                frame_count += 1
                if result["posture"]["status"] == "GOOD":
                    good_posture_count += 1
                else:
                    bad_posture_count += 1
            
            # Add statistics to frame
            stats_text = f"Good: {good_posture_count} | Bad: {bad_posture_count}"
            cv2.putText(
                display_frame,
                stats_text,
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )
            
            # Display frame
            cv2.imshow("Posture Detection - Press 'q' to quit", display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"posture_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"Frame saved as {filename}")
            elif key == ord('r'):
                # Reset statistics
                frame_count = 0
                good_posture_count = 0
                bad_posture_count = 0
                logger.info("Statistics reset")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Print final statistics
        logger.info("\n" + "="*50)
        logger.info("Session Summary:")
        logger.info(f"  Total frames processed: {frame_count}")
        logger.info(f"  Good posture frames: {good_posture_count}")
        logger.info(f"  Bad posture frames: {bad_posture_count}")
        
        if frame_count > 0:
            good_percentage = (good_posture_count / frame_count) * 100
            logger.info(f"  Good posture percentage: {good_percentage:.1f}%")
        
        logger.info("="*50 + "\n")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam closed")


def main():
    """Main entry point"""
    args = parse_args()
    
    logger.info(f"Loading model from {args.model}...")
    
    try:
        detector = PostureDetector(
            model_path=args.model,
            threshold=args.threshold
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info("Model loaded successfully")
    
    # Run webcam detection
    run_webcam(
        detector,
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
