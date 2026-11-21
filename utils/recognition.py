"""Confidence and safety policy for two distinct classifiers."""

from dataclasses import dataclass
from math import isfinite
from string import ascii_uppercase
from typing import Literal

from utils.custom_gestures import DESTRUCTIVE, custom_eligible, sample_detection
from utils.observations import Detection, Prediction, validate_hands


@dataclass(frozen=True, slots=True)
class Thresholds:
    alphabet: float = 0.50
    custom: float = 0.85
    destructive: float = 0.95

    def __post_init__(self):
        if not all(isfinite(x) and 0 <= x <= 1 for x in (self.alphabet, self.custom, self.destructive)):
            raise ValueError("Confidence thresholds must be between 0 and 1")
        if self.destructive < max(self.alphabet, self.custom):
            raise ValueError("Destructive threshold cannot be lower than alphabet or custom thresholds")


@dataclass(frozen=True, slots=True)
class Candidate:
    source: Literal["alphabet", "custom"]
    label: str

    @property
    def destructive(self):
        return self.source == "custom" and self.label in DESTRUCTIVE


@dataclass(frozen=True, slots=True)
class Decision:
    kind: Literal["no-hands", "uncertain", "usable", "ambiguous"]
    reason: str = ""
    candidate: Candidate | None = None
    alphabet: Prediction | None = None
    custom: Prediction | None = None


class Arbiter:
    def __init__(self, alphabet, custom, thresholds=Thresholds()):
        self.alphabet = alphabet
        self.custom = custom
        self.thresholds = thresholds

    def observe(self, detection: Detection, aspect: float) -> Decision:
        if not detection.valid or not validate_hands(detection.hands):
            return Decision("uncertain", "Invalid landmarks or handedness")
        if not detection.hands:
            return Decision("no-hands", "No hands")
        alphabet = self.alphabet.predict(detection) if self.alphabet is not None else None
        alphabet_eligible = (alphabet is not None and len(alphabet.label) == 1
                             and alphabet.label in ascii_uppercase and alphabet.confidence >= self.thresholds.alphabet)
        sample = sample_detection(detection, aspect) if self.custom is not None else None
        custom = self.custom.predict(sample) if sample is not None else None
        if custom is not None and custom_eligible(custom, sample, self.thresholds.custom, self.thresholds.destructive):
            if custom.label in DESTRUCTIVE:
                if alphabet_eligible:
                    return Decision("ambiguous", "Alphabet/command conflict — release both hands", None, alphabet, custom)
                if alphabet is None:
                    reason = ("Destructive webcam actions need SignSpeak" if self.alphabet is None else
                              "No valid alphabet score to check against the destructive command")
                    return Decision("uncertain", reason, None, alphabet, custom)
            return Decision("usable", "", Candidate("custom", custom.label), alphabet, custom)
        if alphabet_eligible:
            return Decision("usable", "", Candidate("alphabet", alphabet.label), alphabet, custom)
        return Decision("uncertain", "No eligible prediction", None, alphabet, custom)
