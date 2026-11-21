"""
Feature extraction for the AI Sign Language Translator.

This module converts MediaPipe landmarks into a fixed-size
126-element feature vector:
- Left hand: 21 landmarks × (x, y, z) = 63 values
- Right hand: 21 landmarks × (x, y, z) = 63 values
"""

import numpy as np

from utils.normalize import normalize_hand


def legacy_features(hands):
    """Raw MediaPipe observations → mirrored Left/Right wrist/farthest features."""
    vector = np.zeros(126)
    for hand in hands:
        mirrored = hand.points.astype(float, copy=True)
        mirrored[:, 0] = 1 - mirrored[:, 0]
        offset = 63 if hand.label == "Left" else 0
        vector[offset:offset + 63] = normalize_hand(mirrored.ravel())
    return vector