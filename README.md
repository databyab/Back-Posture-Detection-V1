##Back Posture Detection - YOLOv8(OpenCV)

A real time human posture detection system using YOLOv8 Pose Estimation and OpenCV.
The application analyzes body keypoints (shoulder, hip, knee) to calculate the back angle and classifies posture as GOOD or BAD in real time.

Features:

  -Real-time posture detection using webcam
  -Image-based posture analysis
  -Uses YOLOv8 pose keypoints
  -Computes back angle using geometry
  -Live posture classification (GOOD / BAD)
  -Lightweight YOLOv8 Nano pose model


Working:

YOLOv8 detects human pose keypoints

Extracts:
 - Left Shoulder
 - Left Hip
 - Left Knee

Calculates the angle at the hip

Classifies posture:
 - GOOD → Angle ≥ 160°
 - BAD → Angle < 160°

Model Setup
Download the YOLOv8 pose model: https://docs.ultralytics.com/models/yolov8/


Output

Live video with:
 - Pose skeleton
 - Back angle (degrees)
 - Posture label (GOOD / BAD)

Color Coding:
🟢 GOOD posture
🔴 BAD posture

Notes
 - Only one person is evaluated (first detected)
 - Works best when the full body is visible

Future Improvements

 - Multi-person posture tracking
 - Side / front posture analysis
 - Sitting posture detection
 - Posture score over time
 - GUI / Web dashboard
