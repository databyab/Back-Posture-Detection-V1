import argparse
import cv2
import math
from ultralytics import YOLO


# -------- ANGLE FUNCTION -------- #
def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])

    if mag_ab == 0 or mag_cb == 0:
        return 0

    val = dot / (mag_ab * mag_cb)
    val = max(-1.0, min(1.0, val))

    return math.degrees(math.acos(val))


# YOLO keypoint indices
LEFT_SHOULDER = 5
LEFT_HIP = 11
LEFT_KNEE = 13


# -------- ARGUMENTS -------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0=webcam or path to video/image")
    parser.add_argument("--model", default="models/yolov8n-pose.pt", help="YOLO model path")
    return parser.parse_args()


# -------- MAIN -------- #
def main():
    args = parse_args()
    model = YOLO(args.model)
    src = args.source

    # Webcam mode
    if src == "0":
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            # Get keypoints
            if results[0].keypoints is not None:
                kpts = results[0].keypoints.xy[0].cpu().numpy()

                try:
                    shoulder = tuple(kpts[LEFT_SHOULDER])
                    hip = tuple(kpts[LEFT_HIP])
                    knee = tuple(kpts[LEFT_KNEE])

                    # Draw markers
                    for p in [shoulder, hip, knee]:
                        cv2.circle(annotated, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)

                    # Calculate back angle
                    angle = calculate_angle(shoulder, hip, knee)

                    cv2.putText(annotated, f"Back Angle: {angle:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Posture classification
                    if angle < 160:
                        status = "BAD"
                        color = (0, 0, 255)
                    else:
                        status = "GOOD"
                        color = (0, 255, 0)

                    cv2.putText(annotated, f"Posture: {status}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                except Exception:
                    pass

            cv2.imshow("YOLO Posture Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # ---------- IMAGE MODE ---------- #
        frame = cv2.imread(src)
        results = model(frame)
        annotated = results[0].plot()

        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            try:
                shoulder = tuple(kpts[LEFT_SHOULDER])
                hip = tuple(kpts[LEFT_HIP])
                knee = tuple(kpts[LEFT_KNEE])

                angle = calculate_angle(shoulder, hip, knee)

                cv2.putText(annotated, f"Back Angle: {angle:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                status = "GOOD" if angle >= 160 else "BAD"
                color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)

                cv2.putText(annotated, f"Posture: {status}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            except:
                pass

        cv2.imshow("YOLO Posture Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
