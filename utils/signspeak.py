"""Raw-hand SignSpeak inference using the pinned RealSign release."""

import json
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase

import numpy as np

from .model_assets import INSTALL_HINT, AssetError, verified_pair
from .observations import Detection, Prediction, validate_hands

MODEL_VERSION = "1.0.0-realsign"
SKLEARN_VERSION = "1.6.1"
HAND_FEATURES = 63
FEATURE_DIMENSION = 2 * HAND_FEATURES
ALPHABET = frozenset(ascii_uppercase)


def build_features(detection: Detection) -> np.ndarray | None:
    """Create left/right wrist-centered slots, or None for invalid/empty input."""
    if not detection.valid or not detection.hands or not validate_hands(detection.hands):
        return None
    features = np.zeros(FEATURE_DIMENSION)
    with np.errstate(over="ignore", invalid="ignore"):
        for hand in detection.hands:
            offset = 0 if hand.label == "Left" else HAND_FEATURES
            slot = features[offset:offset + HAND_FEATURES].reshape(21, 3)
            np.subtract(hand.points, hand.points[0], out=slot, dtype=np.float64)
            scale = np.linalg.norm(slot[12])
            if not np.isfinite(scale):
                return None
            if scale >= 1e-6:
                slot /= scale
    return features if np.isfinite(features).all() else None


def _alphabet_labels(labels) -> bool:
    return (
        len(labels) == len(ALPHABET)
        and all(isinstance(label, str) for label in labels)
        and set(labels) == ALPHABET
    )


def _validate_metadata(metadata_bytes: bytes) -> None:
    try:
        metadata = json.loads(metadata_bytes)
    except ValueError as exc:
        raise AssetError(f"Invalid SignSpeak metadata. {INSTALL_HINT}") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != MODEL_VERSION
        or metadata.get("scikit_learn_version") != SKLEARN_VERSION
        or metadata.get("feature_dimension") != FEATURE_DIMENSION
        or metadata.get("classifier") != "SVC"
        or not isinstance(metadata.get("supported_labels"), list)
        or not _alphabet_labels(metadata["supported_labels"])
    ):
        raise AssetError(f"Incompatible SignSpeak metadata. {INSTALL_HINT}")


def _validate_pipeline(pipeline) -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != [
        "scaler", "classifier",
    ]:
        raise AssetError("Expected the pinned SignSpeak scaler/classifier pipeline.")
    scaler, classifier = (step for _, step in pipeline.steps)
    labels = getattr(classifier, "classes_", None)
    if (
        not isinstance(scaler, StandardScaler)
        or not isinstance(classifier, SVC)
        or classifier.probability is not True
        or any(getattr(step, "n_features_in_", None) != FEATURE_DIMENSION for step in (scaler, classifier))
        or not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or not _alphabet_labels(labels)
    ):
        raise AssetError("SignSpeak pipeline must be a 126-feature StandardScaler and probability SVC for A–Z.")


class SignSpeakAlphabetRecognizer:
    """Offline A–Z inference; confidence eligibility belongs to the arbiter."""

    def __init__(self, directory: Path | None = None):
        model_bytes, metadata_bytes = verified_pair(directory, bundled=directory is None)
        _validate_metadata(metadata_bytes)

        import joblib
        import sklearn

        if sklearn.__version__ != SKLEARN_VERSION:
            raise AssetError(
                f"SignSpeak requires scikit-learn {SKLEARN_VERSION}; found {sklearn.__version__}. "
                "Install the pinned runtime dependencies from requirements.txt."
            )
        try:
            pipeline = joblib.load(BytesIO(model_bytes))
        except Exception as exc:
            raise AssetError(f"Cannot load the verified SignSpeak model. {INSTALL_HINT}") from exc
        _validate_pipeline(pipeline)
        self.pipeline = pipeline

    def predict(self, detection: Detection) -> Prediction | None:
        vector = build_features(detection)
        return None if vector is None else self.predict_features(vector)

    def predict_proba(self, detection: Detection) -> tuple[np.ndarray, str] | None:
        """Return the winning 26-class probability vector and slot for temporal smoothing."""
        vector = build_features(detection)
        if vector is None:
            return None
        batch = np.empty((2, FEATURE_DIMENSION))
        with np.errstate(over="ignore", invalid="ignore"):
            batch[0] = vector
        if not np.isfinite(batch[0]).all():
            return None
        batch[1, :HAND_FEATURES] = batch[0, HAND_FEATURES:]
        batch[1, HAND_FEATURES:] = batch[0, :HAND_FEATURES]
        probabilities = self.pipeline.predict_proba(batch)
        winner = int(np.argmax(probabilities.max(axis=1)))
        return probabilities[winner], ("swapped" if winner else "as-detected")

    def predict_features(self, vector: np.ndarray) -> Prediction | None:
        """Score as-detected and swapped slots together, without reflecting any coordinate."""
        if (
            not isinstance(vector, np.ndarray)
            or vector.shape != (FEATURE_DIMENSION,)
            or vector.dtype.kind not in "fiu"
        ):
            return None
        batch = np.empty((2, FEATURE_DIMENSION))
        with np.errstate(over="ignore", invalid="ignore"):
            batch[0] = vector
        if not np.isfinite(batch[0]).all():
            return None
        batch[1, :HAND_FEATURES] = batch[0, HAND_FEATURES:]
        batch[1, HAND_FEATURES:] = batch[0, :HAND_FEATURES]
        probabilities = self.pipeline.predict_proba(batch)
        winner = int(np.argmax(probabilities.max(axis=1)))
        row = probabilities[winner]
        # Use the pinned service's per-row ordering, including class ties.
        runner_up, top = np.argsort(row)[-2:]
        return Prediction(
            label=str(self.pipeline.classes_[top]),
            confidence=float(row[top]),
            margin=float(row[top] - row[runner_up]),
            slot="swapped" if winner else "as-detected",
        )
