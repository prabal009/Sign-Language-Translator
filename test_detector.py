import cv2
from utils.detector import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame, left, right = detector.detect(frame)

    print(
        "Left detected:", sum(left) != 0,
        "| Right detected:", sum(right) != 0,
        end="\r"
    )

    cv2.imshow("Detector Test", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()