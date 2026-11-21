"""MediaPipe landmarks in raw camera space, with a separate mirrored preview."""

from math import floor

import numpy as np

from .observations import Detection, Hand, valid_hand, validate_hands


class HandDetector:
    def __init__(self, *, static_image_mode: bool = False):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        hands, self._hands = self._hands, None
        if hands is not None:
            hands.close()

    def detect(self, frame: np.ndarray) -> Detection:
        import cv2

        if self._hands is None:
            raise RuntimeError("HandDetector is closed")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb)

    def detect_rgb(self, rgb: np.ndarray) -> Detection:
        if self._hands is None:
            raise RuntimeError("HandDetector is closed")
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        try:
            landmarks = results.multi_hand_landmarks or ()
            handedness = results.multi_handedness or ()
            if len(landmarks) != len(handedness) or len(landmarks) > 2:
                return Detection(valid=False)
            hands = tuple(_raw_hand(points, side) for points, side in zip(landmarks, handedness))
        except (AttributeError, TypeError, ValueError):
            return Detection(valid=False)
        if validate_hands(hands):
            return Detection(hands)
        # Valid points can still be drawn, but duplicate slots are never classified.
        return Detection(tuple(hand for hand in hands if valid_hand(hand)), valid=False)


def _raw_hand(landmarks, handedness) -> Hand:
    if len(handedness.classification) != 1:
        raise ValueError("Expected one raw handedness classification")
    classification = handedness.classification[0]
    if hasattr(classification, "HasField") and not classification.HasField("score"):
        raise ValueError("Missing raw handedness score")
    points = np.array([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float64)
    return Hand(points, classification.label, classification.score)


def _draw_mirrored(
    frame: np.ndarray, detection: Detection, *, indices: bool = False, rgb: bool = False, mirror: bool = True,
) -> np.ndarray:
    import cv2

    preview = cv2.flip(frame, 1) if mirror else frame.copy()
    if not detection.hands:
        return preview

    from mediapipe import solutions

    line_color = (100, 220, 100)
    point_color = (55, 195, 230) if rgb else (230, 195, 55)
    height, width = preview.shape[:2]
    for hand in detection.hands:
        pixels = [
            (
                (width - 1 - min(floor(x * width), width - 1)) if mirror else min(floor(x * width), width - 1),
                min(floor(y * height), height - 1),
            )
            if 0 <= x <= 1 and 0 <= y <= 1 else None
            for x, y, _ in hand.points
        ]
        for start, end in solutions.hands.HAND_CONNECTIONS:
            if pixels[start] is not None and pixels[end] is not None:
                cv2.line(preview, pixels[start], pixels[end], line_color, 2)
        for index, point in enumerate(pixels):
            if point is not None:
                cv2.circle(preview, point, 3, point_color, -1)
                if indices:
                    cv2.putText(
                        preview, str(index), point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA,
                    )
        if indices:
            anchor = next((point for point in pixels if point is not None), None)
            if anchor is not None:
                cv2.putText(
                    preview, f"{hand.label} (raw)", (max(0, anchor[0] - 45), max(16, anchor[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1, cv2.LINE_AA,
                )
    return preview


def render_preview(frame: np.ndarray, detection: Detection, *, indices: bool = False) -> np.ndarray:
    """Mirror only the BGR preview; points and raw handedness are not modified."""
    return _draw_mirrored(frame, detection, indices=indices, rgb=False)


def render_preview_rgb(
    rgb: np.ndarray, detection: Detection, *, indices: bool = False, mirror: bool = True,
) -> np.ndarray:
    """Render landmarks onto an RGB frame, mirroring unless already mirrored for input."""
    return _draw_mirrored(rgb, detection, indices=indices, rgb=True, mirror=mirror)