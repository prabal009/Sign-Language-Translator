"""Legacy mirrored features, custom pose gates and pickle-free recognition."""

import json
from dataclasses import dataclass
from math import cos, isfinite, radians
from pathlib import Path

import numpy as np

from utils.extractor import legacy_features
from utils.model_assets import data_dir
from utils.observations import Detection, Prediction, validate_hands

DESCRIPTIONS = {
    "0": "Both index fingers up; other non-thumb fingers curled",
    "1": "Both little fingers up; other non-thumb fingers curled",
    "SPACE": "One hand, palm facing downward (not upward or turning)",
    "BACKSPACE": "One middle finger up; other non-thumb fingers curled",
    "CLEAR_WORD": "Both middle fingers up; other non-thumb fingers curled",
    "OTHER": "Ordinary letters, palms up/neutral, and transitions (no action)",
}
ACTION_LABELS = tuple(DESCRIPTIONS)[:-1]
LABELS = tuple(DESCRIPTIONS)
DESTRUCTIVE = frozenset(("BACKSPACE", "CLEAR_WORD"))
_HAND_COUNT = dict(zip(ACTION_LABELS, (2, 2, 1, 1, 2)))
_FINGER_INDEX = {"0": 0, "1": 3, "BACKSPACE": 1, "CLEAR_WORD": 1}
_MCP = (5, 9, 13, 17)
_PIP = (6, 10, 14, 18)
_TIP = (8, 12, 16, 20)
_STRAIGHT = cos(radians(155))
_CURLED = cos(radians(130))
_MIN_LENGTH = 1e-9
SCHEMA = 1
PREPROCESSING = "legacy-mirrored-wrist-farthest-v1"
MODEL_LIMIT = 256 * 1024
MODEL_NAME = "custom_gestures.json"
RECOVERY = "Collect custom gesture sessions with collect_data.py, then run python train_model.py."


class CustomModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Sample:
    features: np.ndarray
    aspect: float
    hand_score: float

    def hands(self):
        return tuple(hand for hand in self.features.reshape(2, 21, 3) if np.any(hand))


def sample_detection(detection: Detection, aspect: float) -> Sample | None:
    hands = detection.hands
    if not detection.valid or not hands or not validate_hands(hands) or not isfinite(aspect) or aspect <= 0:
        return None
    features = legacy_features(hands)
    if not np.isfinite(features).all():
        return None
    sample = Sample(features, aspect, min(hand.score for hand in hands))
    return sample if len(sample.hands()) == len(hands) else None


def pose_eligible(label: str, sample: Sample) -> bool:
    hands = sample.hands()
    if label == "OTHER":
        return len(hands) in (1, 2)
    if len(hands) != _HAND_COUNT.get(label):
        return False
    if label == "SPACE":
        return True  # Orientation is learned from positive and negative samples, not inferred in 3D.
    primary = _FINGER_INDEX[label]
    for hand in hands:
        xy = hand[:, :2] * (sample.aspect, 1)
        mcp, pip, tip = xy[list(_MCP)], xy[list(_PIP)], xy[list(_TIP)]
        toward_mcp = mcp - pip
        toward_tip = tip - pip
        mcp_lengths = np.linalg.norm(toward_mcp, axis=1)
        tip_lengths = np.linalg.norm(toward_tip, axis=1)
        palm = np.linalg.norm(xy[9] - xy[0])
        if not np.isfinite(palm) or palm <= _MIN_LENGTH or np.any(
            (mcp_lengths <= _MIN_LENGTH) | (tip_lengths <= _MIN_LENGTH)
        ):
            return False
        angles = np.einsum("ij,ij->i", toward_mcp, toward_tip) / (mcp_lengths * tip_lengths)
        if not np.isfinite(angles).all() or angles[primary] > _STRAIGHT:
            return False
        if pip[primary, 1] - tip[primary, 1] < 0.12 * palm:
            return False
        if any(angle < _CURLED for index, angle in enumerate(angles) if index != primary):
            return False
    return True


def custom_eligible(prediction: Prediction, sample: Sample, threshold=0.85, destructive_threshold=0.95) -> bool:
    label = prediction.label
    destructive = label in DESTRUCTIVE
    return (
        label in ACTION_LABELS
        and sample.hand_score >= 0.70
        and prediction.confidence >= (destructive_threshold if destructive else threshold)
        and prediction.margin >= (0.30 if destructive else 0.20)
        and pose_eligible(label, sample)
    )


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CustomModelError(f"Duplicate field: {key}")
        result[key] = value
    return result


def _array(value, shape):
    def validate(item, dimensions):
        if not dimensions:
            return type(item) in (int, float)
        return (isinstance(item, list) and len(item) == dimensions[0]
                and all(validate(child, dimensions[1:]) for child in item))

    if not validate(value, shape):
        raise CustomModelError(f"Expected numeric parameter shape {shape}")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise CustomModelError("Parameters must be finite")
    return result


class CustomGestureRecognizer:
    def __init__(self, path: Path | None = None):
        path = path if path is not None else data_dir() / "models" / MODEL_NAME
        try:
            with path.open("rb") as file:
                content = file.read(MODEL_LIMIT + 1)
            if len(content) > MODEL_LIMIT:
                raise CustomModelError("Custom model exceeds 256 KiB")
            self._initialize(json.loads(content, object_pairs_hook=_unique_fields))
        except FileNotFoundError as exc:
            raise CustomModelError(f"Custom gestures unavailable: no local custom model. {RECOVERY}") from exc
        except (OSError, ValueError, OverflowError, RecursionError) as exc:
            raise CustomModelError(f"Custom gestures unavailable: {exc}. {RECOVERY}") from exc

    @classmethod
    def from_parameters(cls, parameters):
        model = cls.__new__(cls)
        model._initialize(parameters)
        return model

    def _initialize(self, doc):
        fields = {"schema", "preprocessing", "classes", "mean", "scale", "coef", "intercept"}
        if not isinstance(doc, dict) or set(doc) != fields or type(doc["schema"]) is not int or doc["schema"] != SCHEMA:
            raise CustomModelError("Unknown custom model schema")
        if doc["preprocessing"] != PREPROCESSING:
            raise CustomModelError("Unknown custom feature preprocessing")
        labels = doc["classes"]
        if (not isinstance(labels, list) or len(labels) != len(LABELS)
                or not all(type(label) is str for label in labels) or set(labels) != set(LABELS)):
            raise CustomModelError("Custom classes must be exactly 0, 1, SPACE, BACKSPACE, CLEAR_WORD, OTHER")
        self.classes = tuple(labels)
        self.mean = _array(doc["mean"], (126,))
        self.scale = _array(doc["scale"], (126,))
        self.coef = _array(doc["coef"], (6, 126))
        self.intercept = _array(doc["intercept"], (6,))
        if np.any(self.scale <= 0):
            raise CustomModelError("Scaler divisors must be positive")

    def probabilities(self, features):
        if features.shape != (126,) or not np.isfinite(features).all():
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            logits = self.coef @ ((features - self.mean) / self.scale) + self.intercept
            if not np.isfinite(logits).all():
                return None
            weights = np.exp(logits - logits.max())
            return weights / weights.sum()

    def predict(self, sample: Sample) -> Prediction | None:
        probabilities = self.probabilities(sample.features)
        if probabilities is None:
            return None
        best = int(np.argmax(probabilities))
        runner_up = np.partition(probabilities, -2)[-2]
        return Prediction(self.classes[best], float(probabilities[best]), float(probabilities[best] - runner_up))
