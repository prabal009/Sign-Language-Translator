"""Strict feature CSVs and session-isolated data for custom gesture training."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from io import SEEK_END, TextIOWrapper
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

from utils.custom_gestures import LABELS, PREPROCESSING, Sample, pose_eligible
from utils.extractor import legacy_features
from utils.observations import Hand

FEATURE_HEADER = tuple(f"{hand}{index}_{axis}" for hand in "LR" for index in range(21) for axis in "xyz")
SESSION_HEADER = (*FEATURE_HEADER, "aspect", "hand_score", "label")
LEGACY_HEADER = (*FEATURE_HEADER, "label")
MIN_SESSIONS = 3
MIN_ROWS = 30
_FEATURE_COUNT = len(FEATURE_HEADER)
_SESSION_NAME = re.compile(r"[A-Za-z0-9_-]{1,48}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARITY_LIMIT = 256 * 1024
_PARITY_TOLERANCE = 0.025  # Unit hand distances; allow small differences between mirrored detector passes.


class DatasetError(ValueError):
    pass


def valid_session(name: str) -> bool:
    return isinstance(name, str) and _SESSION_NAME.fullmatch(name) is not None


def _validate_sample(sample: Sample, label: str) -> None:
    features = sample.features
    if (
        not isinstance(features, np.ndarray)
        or features.shape != (_FEATURE_COUNT,)
        or features.dtype.kind not in "fiu"
        or not np.isfinite(features).all()
        or not isinstance(sample.aspect, Real)
        or isinstance(sample.aspect, bool)
        or not isfinite(sample.aspect)
        or sample.aspect <= 0
        or not isinstance(sample.hand_score, Real)
        or isinstance(sample.hand_score, bool)
        or not isfinite(sample.hand_score)
        or not 0 <= sample.hand_score <= 1
    ):
        raise DatasetError("expected 126 finite features, positive aspect and hand_score in [0, 1]")
    if not pose_eligible(label, sample):
        raise DatasetError(f"sample does not meet the hand-count or pose requirements for {label}")


def _feature_key(features: np.ndarray) -> bytes:
    # The classifier sees features only; metadata cannot make a repeated vector independent.
    return (np.asarray(features, dtype=np.float64) + 0.0).tobytes()  # Canonicalize negative zeros.


def _csv_rows(file, path: Path, header: tuple[str, ...]):
    reader = csv.reader(file, strict=True)
    try:
        if next(reader, None) != list(header):
            raise DatasetError(f"{path}: invalid CSV header")
        for row in reader:
            if len(row) != len(header):
                raise DatasetError(f"{path}:{reader.line_num}: expected {len(header)} columns")
            yield reader.line_num, row
    except (csv.Error, UnicodeError) as exc:
        raise DatasetError(f"{path}:{reader.line_num}: invalid CSV: {exc}") from exc


def _session_rows(file, path: Path, label: str):
    for line, row in _csv_rows(file, path, SESSION_HEADER):
        try:
            if row[-1] != label:
                raise DatasetError(f"expected the exact label {label}")
            features = np.asarray(row[:_FEATURE_COUNT], dtype=np.float64)
            sample = Sample(features, float(row[-3]), float(row[-2]))
            _validate_sample(sample, label)
        except ValueError as exc:
            raise DatasetError(f"{path}:{line}: {exc}") from exc
        yield sample


class DatasetManager:
    """Append context for one session and exact label, storing no images."""

    def __init__(self, path: Path, label: str):
        if label not in LABELS:
            raise DatasetError(f"unknown custom label: {label!r}")
        self.path = path
        self.label = label
        self._keys: set[bytes] = set()
        self._file = None
        self._writer = None
        self._needs_separator = False

    @property
    def count(self) -> int:
        return len(self._keys)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = self.path.open("x+b")
        except FileExistsError:
            stream = self.path.open("a+b")
            created = False
        else:
            created = True
        file = TextIOWrapper(stream, encoding="utf-8", newline="")

        try:
            writer = csv.writer(file)
            if created:
                writer.writerow(SESSION_HEADER)
                file.flush()
                self._keys.clear()
                self._needs_separator = False
            else:
                file.seek(0)
                self._keys = {_feature_key(sample.features) for sample in _session_rows(file, self.path, self.label)}
                # A complete CSV need not have a final record terminator.
                stream.seek(-1, SEEK_END)
                self._needs_separator = stream.read(1) not in (b"\n", b"\r")
                file.seek(0, SEEK_END)
            self._file = file
            self._writer = writer
        except BaseException:
            file.close()
            raise
        return self

    def __exit__(self, *_):
        try:
            self._file.close()
        finally:
            self._file = None
            self._writer = None

    def append(self, sample: Sample) -> bool:
        if self._writer is None:
            raise DatasetError("open the dataset with a context manager before appending")
        _validate_sample(sample, self.label)
        key = _feature_key(sample.features)
        if key in self._keys:
            return False
        if self._needs_separator:
            self._file.write("\r\n")
            self._needs_separator = False
        self._writer.writerow((*sample.features, sample.aspect, sample.hand_score, self.label))
        self._file.flush()
        self._keys.add(key)
        return True


@dataclass(slots=True)
class TrainingData:
    training: list[tuple[str, np.ndarray]]
    validation: list[tuple[str, Sample]]
    holdout: str
    duplicates: int
    legacy_count: int


def load_training_data(root: Path, legacy_csv: Path | None = None) -> TrainingData:
    try:
        sessions = sorted(
            (path for path in root.iterdir() if valid_session(path.name) and path.is_dir()), key=lambda path: path.name
        )
    except FileNotFoundError:
        sessions = []
    if len(sessions) < MIN_SESSIONS:
        found = ", ".join(path.name for path in sessions) or "none"
        raise DatasetError(f"{root}: need at least {MIN_SESSIONS} independent sessions; found {len(sessions)} ({found})")

    training = []
    validation = []
    seen: dict[bytes, tuple[str, Path]] = {}
    duplicates = 0
    holdout = sessions[-1]

    def new_row(label, features, path):
        nonlocal duplicates
        key = _feature_key(features)
        previous = seen.get(key)
        if previous is None:
            seen[key] = label, path
            return True
        old_label, old_path = previous
        if old_label != label:
            raise DatasetError(f"conflicting labels on identical features: {old_label} in {old_path}; {label} in {path}")
        duplicates += 1
        return False

    # Reserved full session takes precedence over every training source.
    shortages = []
    for session in (holdout, *sessions[:-1]):
        insufficient = []
        for label in LABELS:
            path = session / f"{label}.csv"
            count = 0
            try:
                file = path.open(encoding="utf-8", newline="")
            except FileNotFoundError:
                pass
            else:
                with file:
                    for sample in _session_rows(file, path, label):
                        if new_row(label, sample.features, path):
                            count += 1
                            if session == holdout:
                                validation.append((label, sample))
                            else:
                                training.append((label, sample.features))
            if count < MIN_ROWS:
                insufficient.append(f"{label}={count}/{MIN_ROWS}")
        if insufficient:
            shortages.append(f"{session.name}: {', '.join(insufficient)}")
    if shortages:
        raise DatasetError("need globally nonduplicate samples per session and label: " + "; ".join(shortages))

    legacy_count = 0
    if legacy_csv is not None:
        for label, features in _legacy_rows(legacy_csv):
            if new_row(label, features, legacy_csv):
                training.append((label, features))
                legacy_count += 1
    return TrainingData(training, validation, holdout.name, duplicates, legacy_count)


def _unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise DatasetError("duplicate JSON fields")
    return fields


def _json_numbers(value, shape):
    if not shape:
        if type(value) not in (float, int) or not isfinite(value):
            raise DatasetError("expected finite numeric landmarks/features")
    elif not isinstance(value, list) or len(value) != shape[0]:
        raise DatasetError("invalid landmarks/features shape")
    else:
        for item in value:
            _json_numbers(item, shape[1:])


def _digest(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _check_parity(path: Path, csv_digest: str):
    receipt = path.with_name(path.name + ".parity.json")
    try:
        with receipt.open("rb") as file:
            contents = file.read(_PARITY_LIMIT + 1)
        if len(contents) > _PARITY_LIMIT:
            raise DatasetError("receipt exceeds 256 KiB")
        doc = json.loads(contents, object_pairs_hook=_unique_fields)
        if (
            not isinstance(doc, dict)
            or set(doc) != {"schema", "preprocessing", "csv_sha256", "references"}
            or type(doc["schema"]) is not int
            or doc["schema"] != 1
            or doc["preprocessing"] != PREPROCESSING
            or doc["csv_sha256"] != csv_digest
        ):
            raise DatasetError("incorrect receipt schema, preprocessing ID or SHA-256 of CSV")
        references = doc["references"]
        if not isinstance(references, list) or not 3 <= len(references) <= 12:
            raise DatasetError("provide 3–12 distinct reference images")
        images = set()
        slots = set()
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"image_sha256", "raw_hands", "legacy_features"}:
                raise DatasetError("reference needs image_sha256, raw_hands and legacy_features")
            image, raw_hands = reference["image_sha256"], reference["raw_hands"]
            if not _digest(image) or image in images:
                raise DatasetError("invalid or duplicate reference image SHA-256")
            if not isinstance(raw_hands, dict) or not raw_hands or set(raw_hands) - {"Left", "Right"}:
                raise DatasetError("reference raw_hands must use raw Left/Right handedness")
            images.add(image)
            slots.add(tuple(sorted(raw_hands)))
            hands = []
            for label, points in raw_hands.items():
                _json_numbers(points, (21, 3))
                hands.append(Hand(np.asarray(points, dtype=np.float64), label, 1.0))
            values = reference["legacy_features"]
            _json_numbers(values, (_FEATURE_COUNT,))
            original = np.asarray(values, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                current = legacy_features(hands)
            original_count = _normalized_hand_count(original)
            if (
                original_count != len(hands)
                or not np.isfinite(current).all()
                or not np.array_equal(np.any(original.reshape(2, 63), axis=1), np.any(current.reshape(2, 63), axis=1))
                or not np.allclose(original, current, rtol=0, atol=_PARITY_TOLERANCE)
            ):
                raise DatasetError(f"reference {image} differs from current mirrored wrist/farthest features")
        if slots != {("Left",), ("Right",), ("Left", "Right")}:
            raise DatasetError("references must cover both single-hand slots and two hands")
    except (OSError, ValueError, OverflowError, RecursionError) as exc:
        raise DatasetError(f"{receipt}: legacy normalization parity not established: {exc}. Recollect if unverified.") from exc


def _normalized_hand_count(features):
    """Count present wrist-centered slots at unit farthest distance, or zero if malformed."""
    if np.max(np.abs(features)) > 1.000001:
        return 0
    count = 0
    for hand in features.reshape(2, 21, 3):
        if np.any(hand):
            if np.any(hand[0]) or not np.isclose(np.linalg.norm(hand, axis=1).max(), 1, atol=1e-6, rtol=0):
                return 0
            count += 1
    return count


def _legacy_rows(path: Path):
    with path.open("rb") as stream:
        _check_parity(path, hashlib.file_digest(stream, "sha256").hexdigest())
        stream.seek(0)
        with TextIOWrapper(stream, encoding="utf-8", newline="") as file:
            for line, row in _csv_rows(file, path, LEGACY_HEADER):
                try:
                    label = row[-1]
                    features = np.asarray(row[:-1], dtype=np.float64)
                    if not label or not np.isfinite(features).all():
                        raise DatasetError("expected a label and 126 finite features")
                    if label in LABELS:
                        if not _normalized_hand_count(features):
                            raise DatasetError("legacy custom features are not wrist-centered and farthest-normalized")
                        yield label, features
                except ValueError as exc:
                    raise DatasetError(f"{path}:{line}: {exc}") from exc