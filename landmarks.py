import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print all landmarks
            for id, lm in enumerate(hand_landmarks.landmark):

                h, w, c = frame.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                print(f"ID: {id}  X:{lm.x:.3f}  Y:{lm.y:.3f}  Z:{lm.z:.3f}")

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                cv2.putText(
                    frame,
                    str(id),
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )

    cv2.imshow("Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''
1. hand_landmarks.landmark
for id, lm in enumerate(hand_landmarks.landmark):

This loops through all 21 landmarks.

enumerate() gives you:

id → Landmark number (0 to 20)
lm → The landmark object
2. Landmark coordinates
lm.x
lm.y
lm.z

MediaPipe gives normalized coordinates.

That means:

0.0 ----------- 1.0

instead of pixels.

Example:

x = 0.50

means:

Middle of image
3. Convert to pixels
cx = int(lm.x * w)
cy = int(lm.y * h)

Suppose:

Image Width = 640

lm.x = 0.50

Then

cx = 320 pixels

Now OpenCV knows exactly where to draw.

4. Draw circles
cv2.circle(...)

We draw a small red dot on every landmark so you can clearly see each detected point.

5. Draw IDs
cv2.putText(...)

Instead of only drawing points,

you'll see

0
1
2
3
...
20

on your hand.
'''