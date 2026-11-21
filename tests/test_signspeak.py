import json
import sys
from hashlib import sha256
from io import BytesIO
from math import sqrt
from string import ascii_uppercase
from types import SimpleNamespace

import numpy as np
import pytest

from utils import model_assets, signspeak
from utils.detector import HandDetector, render_preview
from utils.observations import Detection, Hand, validate_hands
from utils.signspeak import SignSpeakAlphabetRecognizer, build_features


def _points(shift):
    # Asymmetric synthetic coordinates; index 12 is deliberately not farthest from the wrist.
    index = np.arange(21, dtype=float)
    points = np.column_stack((0.17 + 0.024 * index, 0.49 - 0.013 * index, -0.13 + 0.008 * index))
    points += shift
    points[12] = points[0] + (0.08, -0.04, 0.05)
    return points


def _reference_slot(points):
    """Independent transcription of the pinned extract_realsign.py preprocessing."""
    wrist = points[0].tolist()
    middle_tip = points[12].tolist()
    scale = sqrt(sum((current - origin) ** 2 for current, origin in zip(middle_tip, wrist)))
    divisor = 1 if scale < 1e-6 else scale
    return [(current - origin) / divisor for point in points for current, origin in zip(point, wrist)]


def test_feature_order_raw_coordinates_and_wrist_to_middle_tip():
    right = Hand(_points((-0.07, 0.21, -0.04)) * (0.7, 0.9, -1.7), "Right", 0.99)
    left = Hand(_points((0.13, 0.02, 0.14)), "Left", 0.62)
    saved = left.points.copy()
    vector = build_features(Detection((right, left)))
    np.testing.assert_allclose(vector, _reference_slot(left.points) + _reference_slot(right.points))
    np.testing.assert_array_equal(left.points, saved)
    assert np.linalg.norm(vector[60:63]) > 1
    assert np.linalg.norm(vector[36:39]) == pytest.approx(1)


@pytest.mark.parametrize("hand_label,offset", [("Left", 0), ("Right", 63)])
def test_near_zero_scale_uses_one_and_missing_slot_is_exactly_zero(hand_label, offset):
    points = _points((0.1, 0.1, 0.1))
    points[12] = points[0] + (0, 0, 0.0000005)
    vector = build_features(Detection((Hand(points, hand_label, 0.8),)))
    np.testing.assert_array_equal(vector[offset:offset + 63], (points - points[0]).ravel())
    absent = 63 - offset
    np.testing.assert_array_equal(vector[absent:absent + 63], np.zeros(63))


@pytest.mark.parametrize("hands", [
    (Hand(np.zeros(63), "Left", 0.9),),
    (Hand(np.full((21, 3), np.inf), "Left", 0.9),),
    (Hand(np.zeros((21, 3), dtype=complex), "Left", 0.9),),
    (Hand(np.zeros((21, 3)), "Unknown", 0.9),),
    (Hand(np.zeros((21, 3)), "Left", -0.1),),
    (Hand(np.zeros((21, 3)), "Left", float("nan")),),
    (Hand(np.zeros((21, 3)), "Right", 0.9), Hand(np.ones((21, 3)), "Right", 0.8)),
    (Hand(np.zeros((21, 3)), "Left", 0.9),) * 3,
])
def test_invalid_hand_data_is_not_inferred(hands):
    assert not validate_hands(hands)
    recognizer = SignSpeakAlphabetRecognizer.__new__(SignSpeakAlphabetRecognizer)
    assert recognizer.predict(Detection(hands)) is None


def test_no_hands_invalid_detection_and_bad_feature_vectors_bypass_inference():
    # No pipeline attribute: any accidental inference raises instead of returning None.
    recognizer = SignSpeakAlphabetRecognizer.__new__(SignSpeakAlphabetRecognizer)
    assert validate_hands(())
    assert recognizer.predict(Detection()) is None
    assert recognizer.predict(Detection((Hand(_points(0), "Left", 0.9),), valid=False)) is None
    for vector in (np.zeros((2, 63)), np.zeros(125), np.full(126, np.nan), np.full(126, "x")):
        assert recognizer.predict_features(vector) is None


@pytest.mark.parametrize("swapped_top,winner,letter", [(0.91, "swapped", "H"), (0.82, "as-detected", "W")])
def test_one_probability_batch_swaps_slots_without_reflecting_and_uses_actual_classes(swapped_top, winner, letter):
    vector = np.arange(126) / 99 - 0.2
    calls = []
    probabilities = np.zeros((2, 26))
    probabilities[0, [3, 2]] = [0.82, 0.18]
    probabilities[1, [18, 17]] = [swapped_top, 1 - swapped_top]

    def predict_proba(batch):
        calls.append(batch.copy())
        return probabilities

    recognizer = SignSpeakAlphabetRecognizer.__new__(SignSpeakAlphabetRecognizer)
    recognizer.pipeline = SimpleNamespace(classes_=list(ascii_uppercase[::-1]), predict_proba=predict_proba)
    result = recognizer.predict_features(vector)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [vector, np.r_[vector[63:], vector[:63]]])
    assert result.label == letter
    assert result.slot == winner
    confidence = swapped_top if winner == "swapped" else 0.82
    assert result.confidence == pytest.approx(confidence)
    assert result.margin == pytest.approx(2 * confidence - 1)


def _metadata(**changes):
    return {
        "version": "1.0.0-realsign",
        "scikit_learn_version": "1.6.1",
        "feature_dimension": 126,
        "classifier": "SVC",
        "supported_labels": list(ascii_uppercase),
    } | changes


@pytest.fixture
def local_asset_pair(tmp_path, monkeypatch):
    """A test-only tiny pair exercises real size/hash verification without offline pickle use."""
    sources = {"MODEL": b"example artifact bytes", "METADATA": json.dumps(_metadata()).encode()}
    paths = {}
    for key, content in sources.items():
        original_asset = getattr(model_assets, key)
        asset = model_assets.Asset(original_asset.name, len(content), sha256(content).hexdigest())
        monkeypatch.setattr(model_assets, key, asset)
        path = tmp_path / asset.name
        path.write_bytes(content)
        paths[key] = path
    return paths, sources


@pytest.mark.parametrize("key", ["MODEL", "METADATA"])
@pytest.mark.parametrize("damage", ["missing", "truncated", "wrong-digest"])
def test_either_unverified_asset_prevents_deserialization(local_asset_pair, tmp_path, monkeypatch, key, damage):
    import joblib

    paths, sources = local_asset_pair
    path = paths[key]
    content = sources[key]
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(content[:-1])
    else:
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("unverified artifact was deserialized"))
    with pytest.raises(model_assets.AssetError) as exc:
        SignSpeakAlphabetRecognizer(tmp_path)
    assert path.name in str(exc.value)


@pytest.mark.parametrize("changes", [
    {"version": "1.0.1"},
    {"scikit_learn_version": "1.7.0"},
    {"feature_dimension": 63},
    {"classifier": "RandomForestClassifier"},
    {"supported_labels": list(ascii_uppercase[:-1]) + ["0"]},
])
def test_incompatible_metadata_prevents_deserialization(monkeypatch, changes):
    import joblib

    monkeypatch.setattr(signspeak, "verified_pair", lambda *args, **kwargs: (b"example", json.dumps(_metadata(**changes))))
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("incompatible artifact was deserialized"))
    with pytest.raises(model_assets.AssetError, match="Incompatible SignSpeak metadata"):
        SignSpeakAlphabetRecognizer()


def test_wrong_sklearn_runtime_prevents_deserialization(local_asset_pair, tmp_path, monkeypatch):
    import joblib
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", "1.7.0")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("version-incompatible artifact was deserialized"))
    with pytest.raises(model_assets.AssetError, match="requires scikit-learn 1.6.1"):
        SignSpeakAlphabetRecognizer(tmp_path)


def test_only_verified_bytes_are_deserialized_after_path_replacement(local_asset_pair, tmp_path, monkeypatch):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    paths, sources = local_asset_pair
    labels = np.repeat(list(ascii_uppercase), 3)
    synthetic_fixture = np.outer(np.arange(len(labels)), np.linspace(0.1, 0.4, 126))
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))])
    pipeline.fit(synthetic_fixture, labels)

    def load(stream):
        paths["MODEL"].write_bytes(b"replaced after verification")
        assert isinstance(stream, BytesIO)
        assert stream.read() == sources["MODEL"]
        return pipeline

    monkeypatch.setattr(joblib, "load", load)
    assert SignSpeakAlphabetRecognizer(tmp_path).pipeline is pipeline
    pipeline.named_steps["classifier"].probability = False
    paths["MODEL"].write_bytes(sources["MODEL"])
    with pytest.raises(model_assets.AssetError, match="probability SVC"):
        SignSpeakAlphabetRecognizer(tmp_path)


def _fake_vision(monkeypatch, results):
    trace = SimpleNamespace(processed=[], closed=[], lines=[], text=[])
    replies = iter(results)

    def process(rgb):
        assert not rgb.flags.writeable
        trace.processed.append(rgb)
        return next(replies)

    def hands(**settings):
        trace.settings = settings
        return SimpleNamespace(process=process, close=lambda: trace.closed.append(True))

    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        solutions=SimpleNamespace(hands=SimpleNamespace(Hands=hands, HAND_CONNECTIONS=[(0, 1)])),
    ))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_BGR2RGB=1,
        FONT_HERSHEY_SIMPLEX=1,
        LINE_AA=1,
        cvtColor=lambda frame, _mode: frame[..., ::-1].copy(),
        flip=lambda frame, _axis: frame[:, ::-1].copy(),
        line=lambda _frame, start, end, *_args: trace.lines.append((start, end)),
        circle=lambda *_args: None,
        putText=lambda _frame, name, *_args: trace.text.append(name),
    ))
    return trace


def _raw_result(points, classifications):
    return SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in hand])
                              for hand in points],
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label=label, score=score)])
                          for label, score in classifications],
    )


def test_detector_calculates_once_and_only_preview_is_mirrored(monkeypatch):
    points = np.tile([0.25, 0.25, -0.125], (21, 1))
    points[1] = (0.5, 0.5, 0.125)
    trace = _fake_vision(monkeypatch, [_raw_result([points], [("Right", 0.71)])])
    frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape((4, 8, 3))
    with HandDetector(static_image_mode=True) as detector:
        detection = detector.detect(frame)
        preview = render_preview(frame, detection, indices=True)
    detector.close()
    assert trace.settings == {
        "static_image_mode": True, "max_num_hands": 2,
        "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5,
    }
    assert len(trace.processed) == len(trace.closed) == 1
    np.testing.assert_array_equal(trace.processed[0], frame[..., ::-1])
    np.testing.assert_array_equal(preview, frame[:, ::-1])
    assert trace.lines == [((5, 1), (3, 2))]
    assert trace.text == [str(i) for i in range(21)] + ["Right (raw)"]
    assert detection.valid
    assert len(detection.hands) == 1
    hand = detection.hands[0]
    np.testing.assert_array_equal(hand.points, points)
    assert (hand.label, hand.score) == ("Right", 0.71)


def test_detector_distinguishes_no_hands_from_unusable_metadata(monkeypatch):
    points = _points(0)
    cases = [
        (_raw_result([], []), True, 0),
        (_raw_result([points], []), False, 0),
        (_raw_result([points], [("unknown", 0.9)]), False, 0),
        (_raw_result([points[:-1]], [("Left", 0.9)]), False, 0),
        (_raw_result([points, points], [("Left", 0.9), ("Left", 0.9)]), False, 2),
    ]
    trace = _fake_vision(monkeypatch, [result for result, _, _ in cases])
    with HandDetector() as detector:
        for _, valid, expected_count in cases:
            detection = detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))
            assert detection.valid is valid
            assert len(detection.hands) == expected_count
    assert len(trace.processed) == len(cases)
    assert trace.settings["static_image_mode"] is False


def test_detect_rgb_and_render_preview_rgb_avoid_second_conversion(monkeypatch):
    from utils.detector import render_preview_rgb

    points = np.tile([0.25, 0.25, -0.125], (21, 1))
    points[1] = (0.5, 0.5, 0.125)
    trace = _fake_vision(monkeypatch, [_raw_result([points], [("Left", 0.85)])])
    rgb = np.arange(4 * 8 * 3, dtype=np.uint8).reshape((4, 8, 3))
    with HandDetector() as detector:
        detection = detector.detect_rgb(rgb)
        preview = render_preview_rgb(rgb, detection)
    assert len(trace.processed) == 1
    np.testing.assert_array_equal(trace.processed[0], rgb)
    np.testing.assert_array_equal(preview, rgb[:, ::-1])
    assert trace.lines == [((5, 1), (3, 2))]
    assert detection.valid and len(detection.hands) == 1


@pytest.mark.real_model
def test_pinned_pipeline_properties_and_probability_parity(request):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    directory = model_assets.data_dir() / "models"
    if not request.config.getoption("--require-real-model") and not any(
        model_assets.present(directory / asset.name) for asset in model_assets.ASSETS
    ):
        pytest.skip(model_assets.INSTALL_HINT)
    recognizer = SignSpeakAlphabetRecognizer(directory)
    pipeline = recognizer.pipeline
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], SVC)
    assert pipeline.named_steps["classifier"].probability
    assert pipeline.n_features_in_ == 126
    assert set(pipeline.classes_) == set(ascii_uppercase)

    left = Hand(_points(0), "Left", 0.97)
    right = Hand(_points((-0.05, 0.1, 0.05)) * (0.6, 1.3, -1.8), "Right", 0.88)
    for hands, reference in (
        ((right, left), _reference_slot(left.points) + _reference_slot(right.points)),
        ((right,), [0.0] * 63 + _reference_slot(right.points)),
    ):
        first = pipeline.predict_proba([reference])[0]
        swapped = pipeline.predict_proba([reference[63:] + reference[:63]])[0]
        is_swapped = max(swapped) > max(first)
        row = swapped if is_swapped else first
        top, runner_up = np.argsort(row)[::-1][:2]
        actual = recognizer.predict(Detection(hands))
        assert actual.label == pipeline.classes_[top]
        assert actual.slot == ("swapped" if is_swapped else "as-detected")
        assert actual.confidence == pytest.approx(row[top], abs=1e-12)
        assert actual.margin == pytest.approx(row[top] - row[runner_up], abs=1e-12)
