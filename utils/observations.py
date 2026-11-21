"""Camera-independent inputs and predictions shared by the recognizers."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np


@dataclass(frozen=True, slots=True)
class Hand:
    points: np.ndarray
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class Detection:
    hands: tuple[Hand, ...] = ()
    valid: bool = True


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    slot: str = ""


def valid_hand(hand: Hand) -> bool:
    return (
        isinstance(hand, Hand)
        and isinstance(hand.label, str)
        and hand.label in ("Left", "Right")
        and isinstance(hand.score, Real)
        and not isinstance(hand.score, bool)
        and isfinite(hand.score)
        and 0 <= hand.score <= 1
        and isinstance(hand.points, np.ndarray)
        and hand.points.shape == (21, 3)
        and hand.points.dtype.kind in "fiu"
        and bool(np.isfinite(hand.points).all())
    )


def validate_hands(hands: tuple[Hand, ...]) -> bool:
    """Check usable landmark arrays and raw handedness without dropping a hand."""
    return (
        len(hands) <= 2
        and all(valid_hand(hand) for hand in hands)
        and (len(hands) != 2 or hands[0].label != hands[1].label)
    )
