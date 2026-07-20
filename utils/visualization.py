import cv2


def draw_status(
    frame,
    label,
    collected,
    target,
    fps,
    hands_detected
):

    cv2.putText(
        frame,
        f"Label : {label}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Collected : {collected}/{target}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    status = "YES" if hands_detected else "NO"

    color = (0,255,0) if hands_detected else (0,0,255)

    cv2.putText(
        frame,
        f"Hands : {status}",
        (20,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )