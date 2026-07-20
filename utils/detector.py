import cv2
import mediapipe as mp

class HandDetector:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        left_hand = [0] * 63
        right_hand = [0] * 63

        if results.multi_hand_landmarks:

            for landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):

                hand_type = handedness.classification[0].label

                features = []

                for lm in landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                if hand_type == "Left":
                    left_hand = features
                else:
                    right_hand = features

                self.drawer.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, left_hand, right_hand