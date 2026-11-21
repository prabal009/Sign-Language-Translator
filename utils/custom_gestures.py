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
    min_hand_score = 0.50 if prediction.slot == "geometric" else 0.70
    return (
        label in ACTION_LABELS
        and sample.hand_score >= min_hand_score
        and prediction.confidence >= (destructive_threshold if destructive else threshold)
        and prediction.margin >= (0.30 if destructive else 0.20)
        and pose_eligible(label, sample)
    )


def _separated_zero(detection: Detection, sample: Sample) -> bool:
    """Two vertical index fingers pointing upward and clearly separated (not touching like E/F/J/X)."""
    if not pose_eligible("0", sample) or len(detection.hands) != 2:
        return False
    p0, p1 = detection.hands[0].points, detection.hands[1].points
    palm = max(float(np.linalg.norm(p0[9, :2] - p0[0, :2])), float(np.linalg.norm(p1[9, :2] - p1[0, :2])), _MIN_LENGTH)
    tip_dist = float(np.linalg.norm(p0[8, :2] - p1[8, :2]))
    mcp_dist = float(np.linalg.norm(p0[5, :2] - p1[5, :2]))
    for points in (p0, p1):
        dy = float(points[6, 1] - points[8, 1])
        dx = abs(float(points[8, 0] - points[6, 0]))
        if dy <= 0 or dx > 0.45 * dy:
            return False
    return tip_dist > 0.85 * palm and mcp_dist > 0.70 * palm


def _palm_down_space(detection: Detection) -> bool:
    """Single flat horizontal palm facing downward (all 4 non-thumb fingers extended, narrow vertical span)."""
    if len(detection.hands) != 1:
        return False
    points = detection.hands[0].points
    palm_len = float(np.linalg.norm(points[9] - points[0]))
    if not isfinite(palm_len) or palm_len <= 1e-4:
        return False
    straight_3d = cos(radians(145))
    for mcp, pip, tip in zip(_MCP, _PIP, _TIP):
        v1 = points[mcp] - points[pip]
        v2 = points[tip] - points[pip]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 <= _MIN_LENGTH or n2 <= _MIN_LENGTH:
            return False
        if float(np.dot(v1, v2) / (n1 * n2)) > straight_3d:
            return False
    key = [0, 5, 8, 9, 12, 13, 16, 17, 20]
    y_span = float(points[key, 1].max() - points[key, 1].min())
    x_span = float(points[key, 0].max() - points[key, 0].min())
    avg_tip_y = float(points[list(_TIP), 1].mean())
    avg_mcp_y = float(points[list(_MCP), 1].mean())
    return x_span > 1.8 * y_span and abs(avg_tip_y - avg_mcp_y) < 0.08 and avg_tip_y >= float(points[0, 1]) - 0.06


class GeometricCustomGestureRecognizer:
    """Deterministic zero-FP landmark pose recognizer for 0, 1, SPACE, BACKSPACE, and CLEAR_WORD."""

    def predict(self, sample: Sample) -> Prediction | None:
        for label in ("CLEAR_WORD", "BACKSPACE", "1"):
            if pose_eligible(label, sample):
                return Prediction(label, 0.98, 0.90, slot="geometric")
        return None

    def predict_detection(self, detection: Detection, sample: Sample) -> Prediction | None:
        if not detection.valid or not detection.hands:
            return None
        if len(detection.hands) == 2:
            if pose_eligible("CLEAR_WORD", sample):
                return Prediction("CLEAR_WORD", 0.98, 0.90, slot="geometric")
            if pose_eligible("1", sample):
                return Prediction("1", 0.98, 0.90, slot="geometric")
            if _separated_zero(detection, sample):
                return Prediction("0", 0.96, 0.85, slot="geometric")
            return None
        if pose_eligible("BACKSPACE", sample):
            return Prediction("BACKSPACE", 0.98, 0.90, slot="geometric")
        if _palm_down_space(detection):
            return Prediction("SPACE", 0.95, 0.85, slot="geometric")
        return None


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
