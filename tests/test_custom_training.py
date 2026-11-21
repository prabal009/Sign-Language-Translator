"""Synthetic landmark tests only; no user data, camera, or installed model access."""

import csv
import hashlib
import json

import numpy as np
import pytest

from train_model import evaluate_holdout, train_pipeline
from utils.custom_gestures import (
    LABELS,
    MODEL_LIMIT,
    MODEL_NAME,
    PREPROCESSING,
    CustomGestureRecognizer,
    Sample,
    sample_detection,
)
from utils.dataset import LEGACY_HEADER, SESSION_HEADER, DatasetError, DatasetManager, load_training_data, valid_session
from utils.observations import Detection, Hand

_SESSIONS = {"z_last": 200, "a_first": 0, "m_second": 100}
_FINGERS = {"0": 0, "1": 3, "BACKSPACE": 1, "CLEAR_WORD": 1}


def _raw_hand(label, row, handedness):
    points = np.tile([0.4, 0.75, 0.05], (21, 1))
    for index, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        x = 0.38 + 0.025 * index
        points[mcp] = (x, 0.62, 0.03)
        points[pip] = (x, 0.54, 0.04)
        points[tip] = (x, 0.4 if index == _FINGERS.get(label) else 0.60, 0.06)
    # Artificially distinct class markers in joints unused by hard gates.
    marker = (1, 2, 3, 4, 7, 11)[LABELS.index(label)]
    points[marker] = (0.48, 0.70, 0.24)
    points[19, 2] += row * 1e-6 + (0.01 if handedness == "Right" else 0)
    return Hand(points, handedness, 0.99)


def _sample(label, row):
    hands = (_raw_hand(label, row, "Left"),)
    if label in ("0", "1", "CLEAR_WORD"):
        hands += (_raw_hand(label, row, "Right"),)
    sample = sample_detection(Detection(hands), 4 / 3)
    assert sample is not None
    return sample


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _write_session_label(root, session, label, samples):
    _write_csv(
        root / session / f"{label}.csv",
        SESSION_HEADER,
        ((*sample.features, sample.aspect, sample.hand_score, label) for sample in samples),
    )


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "custom-data"
    for session, start in _SESSIONS.items():
        (root / session).mkdir(parents=True)
        for label in LABELS:
            _write_session_label(root, session, label, (_sample(label, n) for n in range(start, start + 30)))
    return root


@pytest.mark.parametrize(
    "name,expected",
    [("A-b_0", True), ("a" * 48, True), ("", False), ("a" * 49, False), ("../a", False), ("a\n", False)],
)
def test_session_name_literal(name, expected):
    assert valid_session(name) == expected


def test_collector_counts_unique_rows_and_validates_before_append(tmp_path):
    path = tmp_path / "a" / "SPACE.csv"
    sample = _sample("SPACE", 0)
    changed = sample.features.copy()
    changed[0] = -0.0
    with DatasetManager(path, "SPACE") as dataset:
        assert dataset.count == 0
        assert dataset.append(sample)
        assert not dataset.append(Sample(changed, 1, 0.9))
        with pytest.raises(DatasetError, match="hand-count or pose"):
            dataset.append(_sample("0", 0))
        with pytest.raises(DatasetError, match="finite features"):
            dataset.append(Sample(np.full(126, np.nan), 1, 0.99))
        assert dataset.count == 1

    path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
    with DatasetManager(path, "SPACE") as dataset:
        assert dataset.count == 1
        assert dataset.append(_sample("SPACE", 1))
    with path.open(encoding="utf-8", newline="") as file:
        header, *rows = csv.reader(file)
    assert tuple(header) == SESSION_HEADER
    assert len(rows) == 2
    assert all(row[-1] == "SPACE" for row in rows)


def test_collector_refuses_existing_wrong_header(tmp_path):
    path = tmp_path / "SPACE.csv"
    payload = b"label\nSPACE\n"
    path.write_bytes(payload)
    with pytest.raises(DatasetError, match="invalid CSV header"), DatasetManager(path, "SPACE"):
        pytest.fail("collector accepted an incompatible CSV")
    assert path.read_bytes() == payload


def test_training_uses_full_last_session_and_portable_matches_sklearn(dataset_root, tmp_path, capsys):
    path = tmp_path / "models" / MODEL_NAME
    pipeline = train_pipeline(dataset_root, path)
    data = load_training_data(dataset_root)
    assert data.holdout == "z_last"
    assert len(data.validation) == 180
    assert len(data.training) == 360
    portable = CustomGestureRecognizer(path)
    assert portable.classes == tuple(pipeline.classes_)
    expected = pipeline.predict_proba(np.stack([sample.features for _, sample in data.validation]))
    actual = np.stack([portable.probabilities(sample.features) for _, sample in data.validation])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    confusion = evaluate_holdout(portable, data.validation)
    assert all(confusion[label, label] > 0 for label in LABELS if label != "OTHER")
    assert confusion["OTHER", "REJECT"] == 30
    assert path.stat().st_size <= MODEL_LIMIT
    assert set(json.loads(path.read_bytes())) == {"schema", "preprocessing", "classes", "mean", "scale", "coef", "intercept"}
    assert sorted(path.parent.iterdir()) == [path]
    output = capsys.readouterr().out
    assert "Held-out session: z_last" in output and "Support" in output and "REJECT" in output


def test_missing_label_or_missing_session_preserves_model(dataset_root, tmp_path):
    path = tmp_path / MODEL_NAME
    path.write_bytes(b"existing test model")
    (dataset_root / "a_first" / "CLEAR_WORD.csv").unlink()
    with pytest.raises(DatasetError, match=r"a_first: CLEAR_WORD=0/30"):
        train_pipeline(dataset_root, path)
    (dataset_root / "m_second").rename(tmp_path / "unused")
    with pytest.raises(DatasetError, match=r"at least 3 independent sessions; found 2"):
        train_pipeline(dataset_root, path)
    assert path.read_bytes() == b"existing test model"


def test_global_duplicate_keeps_held_out_row_first(dataset_root):
    repeated = _sample("0", 200)
    with DatasetManager(dataset_root / "a_first" / "0.csv", "0") as dataset:
        assert dataset.append(Sample(repeated.features, 1, 0.9))
    data = load_training_data(dataset_root)
    assert data.duplicates == 1
    assert not any(np.array_equal(features, repeated.features) for _, features in data.training)
    assert np.array_equal(data.validation[0][1].features, repeated.features)

    _write_session_label(dataset_root, "a_first", "0", (_sample("0", n) for n in [*range(29), 200]))
    with pytest.raises(DatasetError, match=r"a_first: 0=29/30"):
        load_training_data(dataset_root)


def test_cross_label_duplicate_is_an_error(dataset_root):
    with DatasetManager(dataset_root / "a_first" / "OTHER.csv", "OTHER") as dataset:
        assert dataset.append(_sample("SPACE", 200))
    with pytest.raises(DatasetError, match="conflicting labels on identical features: SPACE.*OTHER"):
        load_training_data(dataset_root)


def test_affected_other_blocks_installation_using_real_custom_gates(dataset_root, tmp_path):
    path = tmp_path / MODEL_NAME
    path.write_bytes(b"existing test model")
    # A held-out OTHER example has a qualifying middle-finger pose and lies close
    # to the BACKSPACE training cluster; the runtime gate must count it unsafe.
    examples = [_sample("BACKSPACE", 30), *(_sample("OTHER", n) for n in range(201, 230))]
    _write_session_label(dataset_root, "z_last", "OTHER", examples)
    with pytest.raises(DatasetError, match=r"1 held-out OTHER samples produced destructive decisions"):
        train_pipeline(dataset_root, path)
    assert path.read_bytes() == b"existing test model"


def test_zero_eligible_holdout_decisions_preserve_model(dataset_root, tmp_path):
    path = tmp_path / MODEL_NAME
    path.write_bytes(b"existing test model")
    examples = (
        Sample(sample.features, sample.aspect, 0.69)
        for sample in (_sample("BACKSPACE", n) for n in range(200, 230))
    )
    _write_session_label(dataset_root, "z_last", "BACKSPACE", examples)
    with pytest.raises(DatasetError, match=r"no correct held-out decisions for BACKSPACE"):
        train_pipeline(dataset_root, path)
    assert path.read_bytes() == b"existing test model"


def _reference(hands, index):
    slots = [np.zeros(63), np.zeros(63)]
    for hand in hands:
        # Independent baseline formula on the fixed synthetic landmarks.
        mirrored = np.column_stack((1 - hand.points[:, 0], hand.points[:, 1:]))
        shifted = mirrored - mirrored[0]
        longest = max(sum(float(coordinate) ** 2 for coordinate in point) ** 0.5 for point in shifted)
        slots[0 if hand.label == "Right" else 1] = (shifted / longest).ravel()
    return {
        "image_sha256": hashlib.sha256(f"synthetic reference {index}".encode()).hexdigest(),
        "raw_hands": {hand.label: hand.points.tolist() for hand in hands},
        "legacy_features": np.concatenate(slots).tolist(),
    }


def test_legacy_requires_csv_hash_and_verified_reference_vectors(dataset_root, tmp_path):
    path = tmp_path / "legacy.csv"
    training = _sample("SPACE", 990)
    heldout = _sample("SPACE", 200)
    _write_csv(path, LEGACY_HEADER, [(*training.features, "SPACE"), (*heldout.features, "SPACE"), (*training.features, "D")])
    original = path.read_bytes()
    with pytest.raises(DatasetError, match="parity not established"):
        load_training_data(dataset_root, path)

    receipt = path.with_name(path.name + ".parity.json")
    left, right = _raw_hand("BACKSPACE", 0, "Left"), _raw_hand("1", 1, "Right")
    doc = {
        "schema": 1,
        "preprocessing": PREPROCESSING,
        "csv_sha256": hashlib.sha256(original).hexdigest(),
        "references": [_reference((left,), 1), _reference((right,), 2), _reference((left, right), 3)],
    }
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    data = load_training_data(dataset_root, path)
    assert data.legacy_count == 1
    assert data.duplicates == 1
    assert len(data.training) == 361 and len(data.validation) == 180
    assert data.training[-1][0] == "SPACE"
    np.testing.assert_array_equal(data.training[-1][1], training.features)
    assert path.read_bytes() == original

    path.write_bytes(original + b"\n")
    with pytest.raises(DatasetError, match="SHA-256 of CSV"):
        load_training_data(dataset_root, path)
    path.write_bytes(original)
    doc["references"][0]["legacy_features"] = np.roll(doc["references"][0]["legacy_features"], 63).tolist()
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DatasetError, match="differs from current mirrored wrist/farthest features"):
        load_training_data(dataset_root, path)
