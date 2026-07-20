"""
Project Configuration
Change values here instead of editing multiple files.
"""
import os

# ===========================
# Camera
# ===========================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ===========================
# MediaPipe
# ===========================

MAX_HANDS = 2

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# ===========================
# Dataset
# ===========================

# Setup dynamic path resolution so it never loses track of the root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "raw", "sign_data.csv")

SAMPLES_PER_LABEL = 300

# ===========================
# Labels (Master Dictionary)
# ===========================

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUMBERS = [str(i) for i in range(10)]

WORDS = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "YES",
    "NO",
    "HELP",
    "STOP",
    "GOOD",
    "BAD"
]

# =====================================================================
# ACTIVE ACTIVE TRAINING TARGETS
# Change this line depending on what you want to build/test right now!
# =====================================================================

# Phase 1: Test multi-class tracking with just the first 4 letters (A, B, C, D)
# =====================================================================
# ACTIVE TRAINING TARGETS
# =====================================================================

# Append both SPACE and BACKSPACE to your 26 alphabets 
# =====================================================================
# ACTIVE TRAINING TARGETS
# =====================================================================

# Add CLEAR_WORD to your active functional targets
LABELS = ALPHABETS + ["SPACE", "BACKSPACE", "CLEAR_WORD"]

# Or mix-and-match letters and phrases like this when you are ready:
# LABELS = ["A", "B", "HELLO", "THANK_YOU"]

# Or go full production mode to train all 42 targets at once:
# LABELS = ALPHABETS + NUMBERS + WORDS

# ===========================
# Display
# ===========================

FONT_SCALE = 0.8

TEXT_COLOR = (0, 255, 0)

WARNING_COLOR = (0, 0, 255)

INFO_COLOR = (255, 255, 0) 