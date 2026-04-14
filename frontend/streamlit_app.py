"""
Streamlit Web Dashboard for Posture Detection
Real-time and batch processing interface
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.posture_model import PostureDetector

# Page config
st.set_page_config(
    page_title="Posture Detection",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0;
    }
    .good-posture { color: #09ab3b; font-weight: bold; }
    .bad-posture { color: #ff2b2b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🧍 Posture Detection System")
st.markdown("Real-time posture analysis using YOLOv8 Pose Estimation")

# Initialize session state
if "detector" not in st.session_state:
    try:
        st.session_state.detector = PostureDetector(
            model_path="models/yolov8n-pose.pt",
            threshold=160.0
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

if "threshold" not in st.session_state:
    st.session_state.threshold = 160.0

detector = st.session_state.detector

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threshold slider
    new_threshold = st.slider(
        "Posture Threshold (°)",
        min_value=100,
        max_value=180,
        value=int(st.session_state.threshold),
        step=1,
        help="Angles ≥ threshold are considered GOOD posture"
    )
    
    if new_threshold != st.session_state.threshold:
        detector.threshold = float(new_threshold)
        st.session_state.threshold = float(new_threshold)
    
    st.divider()
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["🎥 Live Webcam", "📸 Image Upload", "🎥 Video Upload", "📊 Video Analysis"],
        help="Choose how you want to analyze posture"
    )
    
    st.divider()
    
    st.markdown("""
    ### 📖 How It Works
    1. The system detects human poses using YOLOv8
    2. Extracts shoulder, hip, and knee positions
    3. Calculates the back angle at the hip
    4. Classifies posture as GOOD or BAD
    
    **GOOD**: Angle ≥ {} degrees  
    **BAD**: Angle < {} degrees
    """.format(int(st.session_state.threshold), int(st.session_state.threshold)))


# Main content based on mode
if mode == "🎥 Live Webcam":
    st.header("Live Webcam Feed")
    st.markdown("Real-time posture detection from your camera")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Camera Feed")
        camera_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📊 Live Stats")
        metric_placeholder = st.empty()
        feedback_placeholder = st.empty()
    
    # Start webcam
    st.markdown("#### ▶️ Start Camera")
    
    if st.button("🎯 Start Live Detection"):
        st.info("Starting camera... Press ESC in camera window to stop")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check permissions.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            good_count = 0
            bad_count = 0
            
            # Create stop button
            stop_button = st.checkbox("Stop", value=False, key="stop_webcam")
            
            while not stop_button:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Detect posture
                result = detector.detect(frame)
                
                frame_count += 1
                
                if result["success"]:
                    if result["posture"]["status"] == "GOOD":
                        good_count += 1
                    else:
                        bad_count += 1
                    
                    # Display annotated frame
                    camera_placeholder.image(
                        result["annotated_frame"],
                        channels="BGR"
                    )
                    
                    # Display live metrics
                    posture_info = result["posture"]
                    angle = posture_info["angle"]
                    status = posture_info["status"]
                    status_emoji = "🟢" if status == "GOOD" else "🔴"
                    
                    with metric_placeholder.container():
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Status", f"{status_emoji} {status}")
                        with col_b:
                            st.metric("Angle", f"{angle:.1f}°")
                    
                    # Display statistics
                    with stats_placeholder.container():
                        if frame_count > 0:
                            good_pct = (good_count / frame_count) * 100
                            st.progress(min(good_pct / 100, 1.0))
                            st.caption(f"Good Posture: {good_pct:.1f}% ({good_count}/{frame_count})")
                    
                    # Feedback
                    if angle >= 165:
                        feedback_placeholder.success("✅ Perfect posture!")
                    elif angle >= 160:
                        feedback_placeholder.info("👍 Good posture!")
                    elif angle >= 150:
                        feedback_placeholder.warning("⚠️ Posture could improve")
                    else:
                        feedback_placeholder.error("🔴 Bad posture - straighten up!")
                else:
                    camera_placeholder.image(frame, channels="BGR")
            
            cap.release()
            st.success("✅ Camera closed")
            
            if frame_count > 0:
                st.divider()
                st.markdown("### 📈 Session Summary")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    st.metric("Total Frames", frame_count)
                with col_y:
                    st.metric("Good Posture", good_count)
                with col_z:
                    good_pct = (good_count / frame_count) * 100
                    st.metric("Good %", f"{good_pct:.1f}%")

elif mode == "📸 Image Upload":
    st.header("Image Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear image with visible body keypoints"
        )
        
        if uploaded_file is not None:
            # Read and display uploaded image
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            st.image(image, channels="BGR", caption="Original Image")
    
    with col2:
        if uploaded_file is not None:
            # Process image
            st.info("Processing image...")
            result = detector.detect(image)
            
            if result["success"]:
                st.image(
                    result["annotated_frame"],
                    channels="BGR",
                    caption="Posture Detection Results"
                )
                
                # Display results
                st.divider()
                st.markdown("### 📊 Results")
                
                posture = result["posture"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    angle = posture["angle"]
                    status_color = "🟢" if posture["status"] == "GOOD" else "🔴"
                    st.metric(
                        "Posture Status",
                        f"{status_color} {posture['status']}",
                        delta=f"{angle:.1f}°"
                    )
                
                with col2:
                    st.metric("Back Angle", f"{angle:.1f}°", delta=f"Threshold: {posture['threshold']}°")
                
                with col3:
                    diff = angle - posture['threshold']
                    st.metric("Angle Difference", f"{diff:+.1f}°")
                
                # Keypoints info
                if result["keypoints"]:
                    st.divider()
                    st.markdown("### 📍 Keypoints Detected")
                    
                    keypoints_data = {
                        "Joint": ["Shoulder", "Hip", "Knee"],
                        "X": [
                            f"{result['keypoints']['shoulder'][0]:.0f}",
                            f"{result['keypoints']['hip'][0]:.0f}",
                            f"{result['keypoints']['knee'][0]:.0f}"
                        ],
                        "Y": [
                            f"{result['keypoints']['shoulder'][1]:.0f}",
                            f"{result['keypoints']['hip'][1]:.0f}",
                            f"{result['keypoints']['knee'][1]:.0f}"
                        ]
                    }
                    st.table(keypoints_data)
            else:
                st.error(f"Detection failed: {result.get('error', 'Unknown error')}")


elif mode == "🎥 Video Upload":
    st.header("Video Frame Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video for posture analysis"
    )
    
    if uploaded_file is not None:
        st.info("Video uploaded! Use Video Analysis mode to process and get statistics.")
        st.video(uploaded_file)


elif mode == "📊 Video Analysis":
    st.header("Video Posture Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a video for detailed analysis",
        type=["mp4", "avi", "mov", "mkv"],
        help="Process entire video and get statistics"
    )
    
    if uploaded_file is not None:
        # Save to secure temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            os.write(temp_fd, uploaded_file.read())
            os.close(temp_fd)
            
            st.info("Analyzing video... This may take a moment.")
            
            progress_bar = st.progress(0)
            
            # Process video with progress callback
            frame_count_dict = {"count": 0}
            def progress_callback(frame_num, result):
                frame_count_dict["count"] = frame_num
                progress_bar.progress(min(frame_num / 300, 1.0))  # Assume ~300 frames
                return True
            
            stats = detector.process_video(temp_path, callback=progress_callback)
            progress_bar.empty()
            
            if stats.get("success"):
                st.divider()
                st.markdown("### 📈 Video Statistics")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Frames",
                        f"{stats['total_frames']}",
                        help="Total frames processed"
                    )
                
                with col2:
                    duration = stats.get("duration_seconds", 0)
                    st.metric(
                        "Duration",
                        f"{duration:.1f}s",
                        help="Video duration in seconds"
                    )
                
                with col3:
                    good_pct = stats.get("good_posture_percentage", 0)
                    status = "🟢" if good_pct >= 50 else "🔴"
                    st.metric(
                        "Good Posture",
                        f"{status} {good_pct:.1f}%",
                        help="Percentage of frames with good posture"
                    )
                
                with col4:
                    avg_angle = stats.get("average_angle", 0)
                    st.metric(
                        "Avg Angle",
                        f"{avg_angle:.1f}°",
                        help="Average back angle across video"
                    )
                
                # Detailed stats
                st.divider()
                st.markdown("### 📊 Detailed Statistics")
                
                stats_data = {
                    "Metric": [
                        "Good Posture Frames",
                        "Bad Posture Frames",
                        "Good Posture %",
                        "Average Angle"
                    ],
                    "Value": [
                        f"{stats['good_posture_frames']}",
                        f"{stats['bad_posture_frames']}",
                        f"{stats['good_posture_percentage']:.1f}%",
                        f"{stats['average_angle']:.1f}°"
                    ]
                }
                st.dataframe(data=stats_data, use_container_width=True)
                
                # Recommendations
                st.divider()
                st.markdown("### 💡 Recommendations")
                
                good_pct = stats.get("good_posture_percentage", 0)
                if good_pct >= 80:
                    st.success("✅ Excellent posture! Keep it up!")
                elif good_pct >= 60:
                    st.info("📌 Good job! Try to improve a bit more. Focus on maintaining your back angle.")
                elif good_pct >= 40:
                    st.warning("⚠️ Your posture needs attention. Try to maintain a straighter back.")
                else:
                    st.error("🔴 Poor posture detected. Adjust your position and try again.")
                
            else:
                st.error(f"Video processing failed: {stats.get('error', 'Unknown error')}")
        
        finally:
            # Ensure temp file cleanup
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                pass  # Silently ignore cleanup errors


# Footer
st.divider()
st.markdown("""
---
**Posture Detection System** | Built with YOLOv8 + Streamlit  
For more information, visit: [GitHub Repository](https://github.com)
""")
