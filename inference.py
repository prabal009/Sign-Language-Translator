import cv2
import time
import joblib
import numpy as np
from collections import Counter

from utils.detector import HandDetector
from utils.extractor import FeatureExtractor
from utils.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, LABELS
)

def main():
    # 1. Load the freshly trained alphanumeric + spaces + deletions package
    try:
        model_package = joblib.load("models/best_sign_model.pkl")
        model = model_package["model"]
        winner_name = model_package.get("model_name", "AI")
        print(f"🧠 Optimized Intelligence Model ({winner_name}) successfully active.")
    except FileNotFoundError:
        print("❌ Core Error: 'models/best_sign_model.pkl' not found. Please run train_model.py first.")
        return

    # Setup Core Video Stream Engine
    detector = HandDetector()
    extractor = FeatureExtractor()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # FORCE HIGH-QUALITY HARDWARE STREAM UPGRADE
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Sentence Construction Engine State Variables
    sentence = ""              # Holds the string being built live
    prediction_history = []    # Queue tracking the last N frames for stability
    BUFFER_SIZE = 15           # Lookback window (~0.5 seconds of consistent gesture)
    
    last_confirmed_char = None # Tracks the last action to prevent duplicate loops
    cooldown_frames = 0        # Dynamic debounce window after an action is logged
    
    print("\n🖥️ Live Translation Dashboard Online.")
    print("👉 Hold an alphabet gesture steady to append a letter.")
    print("👉 Hold SINGLE DELETE steady to remove the previous letter.")
    print("👉 Hold WORD DELETE steady to wipe out the entire previous word.")
    print("👉 Press 'c' inside the window canvas to wipe the sentence and start fresh.")
    print("👉 Press 'q' to shut down safely.")

    # 🖥️ FORCE FULLSCREEN WINDOW ENGINE PROPERTIES
    window_name = "Alphanumeric Sentence Constructor v3.2"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run MediaPipe hand tracking pipeline
        frame, left_hand, right_hand = detector.detect(frame)
        features = extractor.extract(left_hand, right_hand)

        hands_visible = sum(features) != 0
        predicted_label = "Scanning..."

        if hands_visible:
            # Predict current frame gesture
            features_arr = np.array(features).reshape(1, -1)
            predicted_label = model.predict(features_arr)[0]
            
            # Decrease typing delay threshold down if active
            if cooldown_frames > 0:
                cooldown_frames -= 1
            
            # Append predictions into our stabilization buffer
            if cooldown_frames == 0:
                prediction_history.append(predicted_label)
                if len(prediction_history) > BUFFER_SIZE:
                    prediction_history.pop(0)

                # Find the most frequent, stable prediction in the queue
                counter = Counter(prediction_history)
                most_common_pred, count = counter.most_common(1)[0]

                # STABILITY CHECK: If the same gesture is held for 85% of the buffer size
                if count >= int(BUFFER_SIZE * 0.85):
                    # Check if it's a completely new character/action string
                    if most_common_pred != last_confirmed_char:
                        
                        if most_common_pred == "SPACE":
                            sentence += " "
                            print("📝 Captured Action: [Added Space]")
                        
                        elif most_common_pred == "BACKSPACE":
                            if len(sentence) > 0:
                                sentence = sentence[:-1]  # Strip previous letter
                                print("📝 Captured Action: [Deleted Last Character]")
                            else:
                                print("⚠️ System: Canvas already empty.")
                        
                        elif most_common_pred == "CLEAR_WORD":
                            if len(sentence) > 0:
                                trimmed_sentence = sentence.rstrip()
                                last_space_idx = trimmed_sentence.rfind(" ")
                                
                                if last_space_idx == -1:
                                    sentence = "" # Only one word existed
                                else:
                                    sentence = trimmed_sentence[:last_space_idx + 1]
                                print("📝 Captured Action: [Deleted Entire Word]")
                            else:
                                print("⚠️ System: Canvas already empty.")
                        
                        else:
                            sentence += most_common_pred
                            print(f"📝 Captured Character: [{most_common_pred}]")
                        
                        last_confirmed_char = most_common_pred
                        prediction_history.clear()
                        cooldown_frames = 25  
        else:
            # If no hands are visible, reset stability checkpoints smoothly
            prediction_history.clear()
            last_confirmed_char = None
            if cooldown_frames > 0:
                cooldown_frames -= 1

        # -----------------------------------------------------------------
        # 🎨 RENDER MODERN HEADS-UP DISPLAY LAYOUT
        # -----------------------------------------------------------------
        
        # 1. Top Bar: Live Current Gesture Reading
        cv2.rectangle(frame, (0, 0), (w, 60), (45, 45, 45), cv2.FILLED)
        cv2.putText(frame, f"LIVE GESTURE: {predicted_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if hands_visible else (200, 200, 200), 2)

        # 2. Bottom Bar: Real-time Sentence Construction Monitor
        cv2.rectangle(frame, (0, h - 80), (w, h), (25, 25, 25), cv2.FILLED)
        cv2.putText(frame, "CONSTRUCTED TEXT:", (20, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Display the live built sentence string
        cv2.putText(frame, sentence if sentence else "[Waiting for signs...]", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)

        # Show the display using our fullscreen window frame variable
        cv2.imshow(window_name, frame)
        
        # Check keystroke options within window scope
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):   # Reset sentence manually
            sentence = ""
            last_confirmed_char = None
            print("🧹 Text canvas wiped clean.")
        elif key == ord('q'): # Safe quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()