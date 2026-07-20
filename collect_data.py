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
    # 1. Setup clean directory paths automatically
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    # 2. Initialize our clean, reusable core components
    detector = HandDetector()
    extractor = FeatureExtractor()
    dataset = DatasetManager(DATASET_PATH)
    
    # 3. Configure Camera via Central Config definitions
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    label_index = 0
    
    # Fast-forward past labels that are already completely collected in your CSV
    while label_index < len(LABELS):
        temp_label = LABELS[label_index]
        if dataset.count_label(temp_label) >= SAMPLES_PER_LABEL:
            print(f"⏩ Class [{temp_label}] is already fully collected. Fast-forwarding...")
            label_index += 1
        else:
            break

    if label_index >= len(LABELS):
        print("🏆 Milestone Complete: All configured labels are already fully collected!")
        cap.release()
        return

    current_label = LABELS[label_index]
    prev_time = time.time()
    
    # Dynamic Automation States
    is_recording = False
    countdown_active = False
    countdown_start_time = 0
    countdown_duration = 3.0 # 3 seconds preparation delay

    print(f"🚀 Data Collection Engine Online.")
    print(f"🎯 Target: {SAMPLES_PER_LABEL} samples per class.")
    print(f"📂 Collecting data into: {DATASET_PATH}\n")
    print(f"👉 Current Target Label: [{current_label}]")
    print("👉 Click the camera window and press 'SPACE' to trigger automatic rapid-fire collection.")

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Error: Failed to grab frame from webcam hardware.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Core Pipeline Processing
        frame, left_hand, right_hand = detector.detect(frame)
        features = extractor.extract(left_hand, right_hand)
        
        # Calculate HUD Metrics
        current = time.time()
        fps = int(1 / (current - prev_time)) if (current - prev_time) > 0 else 0
        prev_time = current
        
        count = dataset.count_label(current_label)
        hands_visible = sum(features) != 0
        
        # State Machine Automation
        if countdown_active:
            elapsed = time.time() - countdown_start_time
            remaining = countdown_duration - elapsed
            if remaining > 0:
                # Flash a highly visible preparation banner over the frame
                cv2.rectangle(frame, (0, h//2 - 40), (w, h//2 + 40), (0, 140, 255), cv2.FILLED)
                cv2.putText(frame, f"GET READY: {remaining:.1f}s", (w//2 - 140, h//2 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            else:
                countdown_active = False
                is_recording = True
                print(f"🔴 Streaming frames for [{current_label}] now...")

        elif is_recording:
            if hands_visible:
                dataset.save(current_label, features)
                # Re-fetch count dynamically
                count = dataset.count_label(current_label)
            else:
                # Visual warning if your hand slips out of view mid-collection
                cv2.putText(frame, "⚠️ HOLD HAND IN FRAME!", (w//2 - 180, h - 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Auto-Advance to next character once threshold matches
            if count >= SAMPLES_PER_LABEL:
                is_recording = False
                print(f"✅ Completed 300 frames for class [{current_label}]!")
                
                if label_index < len(LABELS) - 1:
                    label_index += 1
                    current_label = LABELS[label_index]
                    print(f"🔮 Automated shift! Next Target Context: [{current_label}]")
                else:
                    print("\n🏆 Milestone Complete: Entire feature array dataset gathered successfully!")
                    break

        # Render your modular visual HUD display layer
        draw_status(
            frame,
            current_label,
            count,
            SAMPLES_PER_LABEL,
            fps,
            hands_visible,
        )
        
        # Add a visual indicator to show if the script is idle, counting down, or recording
        if not is_recording and not countdown_active:
            cv2.putText(frame, "PRESS SPACE TO START", (20, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif is_recording:
            cv2.putText(frame, "● RECORDING LIVE DATA", (20, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Dataset Collector v2.0", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Key Controls
        if key == ord(" "):  # Listen for Spacebar
            if not is_recording and not countdown_active:
                countdown_active = True
                countdown_start_time = time.time()
                
        elif key == ord("q"):
            print("\n🛑 Collection sequence terminated by developer context.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()