import cv2
import time
import os
from utils.detector import HandDetector
from utils.extractor import FeatureExtractor
from utils.dataset import DatasetManager
from utils.visualization import draw_status
from utils.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DATASET_PATH,
    LABELS,
    SAMPLES_PER_LABEL,
)

def main():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    detector = HandDetector()
    extractor = FeatureExtractor()
    dataset = DatasetManager(DATASET_PATH)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    label_index = 0
    current_label = LABELS[label_index]
    prev_time = time.time()
    
    # Automated states
    is_collecting = False
    countdown_start = 0
    countdown_duration = 3  # Seconds to prepare your hand
    
    print("🤖 Automated Data Collector Engine Online.")
    print("Press 'SPACE' to start the auto-collection sequence for the current label.")
    print("Press 'Q' to Quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        frame, left_hand, right_hand = detector.detect(frame)
        features = extractor.extract(left_hand, right_hand)
        
        current = time.time()
        fps = int(1 / (current - prev_time)) if (current - prev_time) > 0 else 0
        prev_time = current
        
        count = dataset.count_label(current_label)
        hands_visible = sum(features) != 0
        
        # --- Automated Logic Loop ---
        status_text = f"Label: {current_label} ({count}/{SAMPLES_PER_LABEL})"
        
        if not is_collecting:
            cv2.putText(frame, "Press SPACE to Start", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            time_elapsed = time.time() - countdown_start
            if time_elapsed < countdown_duration:
                # Still in countdown phase
                countdown_left = int(countdown_duration - time_elapsed) + 1
                cv2.putText(frame, f"GET READY: {countdown_left}", (20, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # Active streaming data collection phase
                if hands_visible:
                    dataset.save(current_label, features)
                    # Small delay (approx 30-40ms) happens naturally per frame, 
                    # gathering smooth real-time tracking variants
                    
                    if dataset.count_label(current_label) >= SAMPLES_PER_LABEL:
                        is_collecting = False  # Pause collection
                        if label_index < len(LABELS) - 1:
                            label_index += 1
                            current_label = LABELS[label_index]
                            print(f"\n📦 Shifting target to: [{current_label}]. Press SPACE when ready.")
                        else:
                            print("\n🏆 Entire dataset gathered successfully!")
                            break
                else:
                    cv2.putText(frame, "SHOW HAND TO RECORD", (20, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw default UI elements
        draw_status(frame, current_label, count, SAMPLES_PER_LABEL, fps, hands_visible)
        cv2.imshow("Automated Dataset Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Trigger Countdown/Start
        if key == ord(" "):
            if not is_collecting:
                is_collecting = True
                countdown_start = time.time()
                
        elif key == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()