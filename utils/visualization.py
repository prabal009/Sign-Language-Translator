import cv2


def draw_status(frame, lines):
    """Draw a readable three-line collector HUD over the mirrored frame."""
    if frame.shape[1] < 24:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = frame.shape[1] - 20
    for row, text in enumerate(lines):
        (full_width, _), _ = cv2.getTextSize(text, font, 0.65, 1)
        scale = 0.65 * min(1, width / max(1, full_width))
        y = 26 * (row + 1)
        cv2.putText(frame, text, (10, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)