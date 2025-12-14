import math


# Indices mapping used by YOLOv8-pose (17 keypoints typical)
# This mapping is illustrative — confirm with model docs if different.
LEFT_SHOULDER = 5
LEFT_HIP = 11
LEFT_KNEE = 13


def calculate_angle(a, b, c):
    """Calculate angle ABC (in degrees) where b is the vertex."""
    ax, ay = a
    bx, by = b
    cx, cy = c


    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)


    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])

    if mag_ab * mag_cb == 0:
        return 0.0


    val = dot / (mag_ab * mag_cb)
    # numeric safety
    val = max(-1.0, min(1.0, val))
    angle = math.degrees(math.acos(val))
    return angle




def posture_feedback(angle, threshold=160.0):
    """Return a simple label for back posture."""
    return "GOOD" if angle >= threshold else "BAD"