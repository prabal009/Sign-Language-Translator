import sys
import cv2
import joblib
import numpy as np
from collections import Counter

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

from utils.detector import HandDetector
from utils.extractor import FeatureExtractor
from utils.config import CAMERA_INDEX, LABELS

# -----------------------------------------------------------------
# ⚙️ BACKGROUND CAMERA PROCESSING WORKER
# -----------------------------------------------------------------
class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, str) # Emits (live_gesture, current_sentence)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.sentence = ""
        self.prediction_history = []
        self.BUFFER_SIZE = 15
        self.last_confirmed_char = None
        self.cooldown_frames = 0

    def run(self):
        # Load the ML architecture safely inside the thread
        try:
            model_package = joblib.load("models/best_sign_model.pkl")
            model = model_package["model"]
        except Exception as e:
            print(f"Error loading model in background thread: {e}")
            return

        detector = HandDetector()
        extractor = FeatureExtractor()
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        while self._run_flag:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame, left_hand, right_hand = detector.detect(frame)
            features = extractor.extract(left_hand, right_hand)

            hands_visible = sum(features) != 0
            predicted_label = "Scanning..."

            if hands_visible:
                features_arr = np.array(features).reshape(1, -1)
                predicted_label = model.predict(features_arr)[0]
                
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1
                
                if self.cooldown_frames == 0:
                    self.prediction_history.append(predicted_label)
                    if len(self.prediction_history) > self.BUFFER_SIZE:
                        self.prediction_history.pop(0)

                    counter = Counter(self.prediction_history)
                    most_common_pred, count = counter.most_common(1)[0]

                    if count >= int(self.BUFFER_SIZE * 0.85):
                        if most_common_pred != self.last_confirmed_char:
                            if most_common_pred == "SPACE":
                                self.sentence += " "
                            elif most_common_pred == "BACKSPACE":
                                if len(self.sentence) > 0:
                                    self.sentence = self.sentence[:-1]
                            elif most_common_pred == "CLEAR_WORD":
                                if len(self.sentence) > 0:
                                    trimmed = self.sentence.rstrip()
                                    idx = trimmed.rfind(" ")
                                    self.sentence = "" if idx == -1 else trimmed[:idx + 1]
                            else:
                                self.sentence += most_common_pred
                            
                            self.last_confirmed_char = most_common_pred
                            self.prediction_history.clear()
                            self.cooldown_frames = 25
            else:
                self.prediction_history.clear()
                self.last_confirmed_char = None
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1

            # Broadcast the camera frames and the textual readouts to the UI
            self.change_pixmap_signal.emit(frame)
            self.prediction_signal.emit(predicted_label, self.sentence)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def clear_canvas(self):
        self.sentence = ""
        self.last_confirmed_char = None


# -----------------------------------------------------------------
# 🎨 HIGH-END USER INTERFACE DISPLAY WINDOW
# -----------------------------------------------------------------
class ModernSignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translation System v4.0")
        self.showFullScreen() # Force full-screen application coverage
        
        # Core Dark UI Theme Configuration Profile
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        # Main Layout Structure
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Bar Control Ribbon Component
        self.top_bar = QWidget()
        self.top_bar.setStyleSheet("background-color: #1E1E1E; border-bottom: 2px solid #2D2D2D;")
        self.top_bar.setFixedHeight(80)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(40, 0, 40, 0)

        self.live_gesture_label = QLabel("LIVE GESTURE: Scanning...")
        self.live_gesture_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.live_gesture_label.setStyleSheet("color: #00FF7F;")
        top_layout.addWidget(self.live_gesture_label)

        self.info_label = QLabel("Press 'C' to clear | Press 'Esc' to exit")
        self.info_label.setFont(QFont("Segoe UI", 12))
        self.info_label.setStyleSheet("color: #888888;")
        top_layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.top_bar)

        # 2. Central Video Widget Area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000;")
        main_layout.addWidget(self.video_label, stretch=1)

        # 3. Bottom Output Console Layer Component
        self.bottom_bar = QWidget()
        self.bottom_bar.setStyleSheet("background-color: #1A1A1A; border-top: 2px solid #2D2D2D;")
        self.bottom_bar.setFixedHeight(140)
        bottom_layout = QVBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(40, 20, 40, 20)

        title_text = QLabel("CONSTRUCTED TRANSLATION LAYER:")
        title_text.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        title_text.setStyleSheet("color: #757575; text-transform: uppercase; letter-spacing: 1px;")
        bottom_layout.addWidget(title_text)

        self.sentence_display = QLabel("[Waiting for signs...]")
        self.sentence_display.setFont(QFont("Segoe UI", 26, QFont.Weight.DemiBold))
        self.sentence_display.setStyleSheet("color: #FFD700;")
        bottom_layout.addWidget(self.sentence_display)
        main_layout.addWidget(self.bottom_bar)

        # Mount processing worker loop threads
        self.thread = VideoWorker()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.prediction_signal.connect(self.update_text)
        self.thread.start()

    def update_image(self, cv_img):
        # Transform the OpenCV layout metrics directly into high-res Qt Pixmaps
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale smoothly to fill the workspace while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_format).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_text(self, gesture, sentence):
        self.live_gesture_label.setText(f"LIVE GESTURE: {gesture}")
        if gesture == "Scanning...":
            self.live_gesture_label.setStyleSheet("color: #B0B0B0;")
        else:
            self.live_gesture_label.setStyleSheet("color: #00FF7F;")

        if sentence:
            self.sentence_display.setText(sentence)
        else:
            self.sentence_display.setText("[Waiting for signs...]")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.thread.clear_canvas()
        elif event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernSignLanguageApp()
    window.show()
    sys.exit(app.exec())