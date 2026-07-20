"""
Feature extraction for the AI Sign Language Translator.

This module converts MediaPipe landmarks into a fixed-size
126-element feature vector:
- Left hand: 21 landmarks × (x, y, z) = 63 values
- Right hand: 21 landmarks × (x, y, z) = 63 values
"""

from utils.normalize import normalize_hand


class FeatureExtractor:
    """Extract and normalize landmark features."""

    def __init__(self):
        pass

    def extract(self, left_hand, right_hand):
        """
        Parameters
        ----------
        left_hand : list
            63 values (or all zeros if no left hand)

        right_hand : list
            63 values (or all zeros if no right hand)

        Returns
        -------
        list
            Normalized feature vector of length 126
        """

        left_hand = normalize_hand(left_hand)
        right_hand = normalize_hand(right_hand)

        return left_hand + right_hand