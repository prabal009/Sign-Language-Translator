import hashlib
import json
from string import ascii_uppercase
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pytest

from utils import evaluate_signspeak as evaluator
from utils.observations import Detection, Hand
from utils.signspeak import SignSpeakAlphabetRecognizer


def _png(left, right):
    frame = np.empty((4, 6, 3), dtype=np.uint8)
    frame[:, :3] = left
    frame[:, 3:] = right
    success, encoded = cv2.imencode(".png", frame)
    assert success
    return encoded.tobytes()


def _points(marker):
    points = np.repeat([[0.4, 0.5, -0.3]], 21, axis=0)
    points[5, 0] += marker
    points[12, 2] += 0.25
    points[20, 1] += 0.8  # Deliberately not the normalization reference.
    return points


class _Pipeline:
    classes_ = np.array(list(reversed(ascii_uppercase)))

    def predict_proba(self, batch):
        rows = []
        for features in batch:
            right_present = features[63 + 15] != 0
            marker = features[63 + 15] if right_present else features[15]
            label, confidence = ("I", 0.49) if marker > 0.3 else ("Y", 0.83)
            if not right_present:
                confidence -= 0.10
            row = np.full(26, (1 - confidence) / 25)
            row[np.where(self.classes_ == label)[0][0]] = confidence
            rows.append(row)
        return np.array(rows)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    first = _png(10, 11)
    entries = {
        "Dataset/Testing/I/03.png": b"not an image",
        "Dataset/Testing/A/04.png": _png(50, 51),  # Not a replacement for the duplicate.
        "Dataset/Testing/A/03.png": _png(10, 12),
        "Dataset/Testing/Q/02.png": first,  # Duplicate across labels.
        "Dataset/Testing/Q/01.png": _png(20, 21),
        "Dataset/Testing/I/02.png": _png(30, 31),
        "Dataset/Testing/I/01.png": _png(10, 13),
        "Dataset/Testing/A/02.png": first,
        "Dataset/Testing/A/01.PNG": first,
        "Dataset/Testing/A/00.txt": b"not an image",
        "Dataset/Training/B/01.png": first,
        "Dataset/Testing/AA/01.png": first,
        "Dataset/Testing/B/nested/01.png": first,
    }
    path = tmp_path / "Sample Dataset.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as bundle:
        for name, data in entries.items():
            bundle.writestr(name, data)
    pin = path.read_bytes()
    monkeypatch.setattr(evaluator, "ARCHIVE_SIZE", len(pin))
    monkeypatch.setattr(evaluator, "ARCHIVE_SHA256", hashlib.sha256(pin).hexdigest())
    return path, entries


@pytest.fixture
def synthetic_inference(monkeypatch):
    calls = []

    class Detector:
        def __init__(self, *, static_image_mode):
            assert static_image_mode is True

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            calls.append("closed")

        def detect(self, frame):
            ends = tuple(int(value) for value in frame[0, [0, -1], 0])
            calls.append(ends)
            if ends == (20, 21):
                return Detection()
            if ends == (21, 20):
                hand = Hand(_points(0.05), "Left", 0.82)
                return Detection((hand, hand), valid=False)
            hand = Hand(_points(0.1 if 30 in ends else 0.05), "Right" if ends[0] < ends[1] else "Left", 0.92)
            return Detection((hand,))

    model = object.__new__(SignSpeakAlphabetRecognizer)
    model.pipeline = _Pipeline()
    monkeypatch.setattr(evaluator, "HandDetector", Detector)
    monkeypatch.setattr(evaluator, "SignSpeakAlphabetRecognizer", lambda: model)
    return calls


def test_sorted_selection_accepts_root_and_enclosed_images_without_extracting(tmp_path):
    path = tmp_path / "layout.zip"
    names = (
        "Testing/B/03.WebP", "Testing/B/02.jpeg", "Testing/B/01.JPG", "Testing/B/04.png",
        "Dataset/Testing/A/01.png", "Dataset/Testing/A/00/", "Outside/Dataset/Testing/A/00.jpg",
        "/Testing/C/00.jpg", "../Testing/C/01.jpg", "Dataset\\Testing\\C\\00.jpg",
    )
    with ZipFile(path, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"")
    with ZipFile(path) as bundle:
        selection, counts = evaluator._select(bundle)
    assert [(label, member.filename) for label, member in selection] == [
        ("A", "Dataset/Testing/A/01.png"),
        ("B", "Testing/B/01.JPG"), ("B", "Testing/B/02.jpeg"), ("B", "Testing/B/03.WebP"),
    ]
    assert sum(counts.values()) == 5
    assert counts["B"] == 4


def test_cli_report_preserves_misses_duplicates_and_flip_evidence(tmp_path, archive, synthetic_inference, monkeypatch):
    path, entries = archive
    report_path = tmp_path / "Evidence Report.json"
    report_path.write_text("old contents", encoding="utf-8")

    def no_extract(*_args, **_kwargs):
        pytest.fail("the evaluator must not extract archive members")

    monkeypatch.setattr(ZipFile, "extract", no_extract)
    monkeypatch.setattr(ZipFile, "extractall", no_extract)
    assert evaluator.main(["--archive", str(path), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["archive"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["model"]["alphabet_threshold"] == 0.5
    assert "not an independent benchmark" in report["scope"]
    selection = report["selection"]
    assert (selection["eligible_paths"], selection["selected"], selection["unique"], selection["duplicates"]) == (9, 8, 6, 2)
    assert selection["eligible_by_letter"]["A"] == 4
    assert selection["selected_by_letter"]["A"] == 3
    samples = {row["filename"]: row for row in report["samples"]}
    assert list(samples) == sorted(samples)
    assert "Dataset/Testing/A/04.png" not in samples
    for name, row in samples.items():
        assert row["sha256"] == hashlib.sha256(entries[name]).hexdigest()
    for name in ("Dataset/Testing/A/02.png", "Dataset/Testing/Q/02.png"):
        assert samples[name]["duplicate_of"] == "Dataset/Testing/A/01.PNG"
        assert "passes" not in samples[name]

    wrong = samples["Dataset/Testing/I/01.png"]
    assert wrong["expected"] == "I"
    for orientation, slot, side in (("raw", "as-detected", "Right"), ("flipped", "swapped", "Left")):
        row = wrong["passes"][orientation]
        assert row["raw_top"] == "Y" and row["confidence"] == pytest.approx(0.83)
        assert row["reason"] == "accepted" and row["accepted"] is True
        assert row["hand_count"] == 1 and row["winning_slot"] == slot
        assert row["raw_handedness"] == [{"label": side, "score": 0.92}]
        summary = report["orientations"][orientation]
        assert summary["totals"] == dict(evaluated=6, scored=4, accepted=3, correct_top=1,
                                         accepted_correct=0, wrong_top=3, accepted_wrong=3, rejected=3)
        assert summary["coverage_by_letter"]["I"] == dict(evaluated=3, scored=2, accepted=1)
        assert summary["coverage_by_letter"]["Z"] == dict(evaluated=0, scored=0, accepted=0)
        assert summary["top_confusion"]["I"] == {"I": 1, "Y": 1}
        assert summary["accepted_confusion"]["I"] == {"Y": 1}
        assert summary["targeted_confusions"]["I->Y"] == {"raw_top": 1, "accepted": 1}
        assert summary["targeted_confusions"]["Y->I"] == {"raw_top": 0, "accepted": 0}
        assert set(summary["targeted_confusions"]) == {
            "I->Y", "Y->I", "Q->Y", "Y->Q", "S->E", "E->S", "V->N", "N->V", "Z->B", "B->Z",
        }
        low = samples["Dataset/Testing/I/02.png"]["passes"][orientation]
        assert low["raw_top"] == "I" and low["confidence"] == pytest.approx(0.49)
        assert low["reason"] == "below-threshold" and low["accepted"] is False
        unreadable = samples["Dataset/Testing/I/03.png"]["passes"][orientation]
        assert unreadable["reason"] == "decode-failed" and unreadable["hand_count"] is None

    passes = samples["Dataset/Testing/Q/01.png"]["passes"]
    assert passes["raw"]["reason"] == "no-hands" and passes["raw"]["hand_count"] == 0
    assert passes["flipped"]["reason"] == "invalid-detection" and passes["flipped"]["hand_count"] == 2
    assert report["raw_to_flipped_single_hand"] == {"Left": {"Left": 0, "Right": 0}, "Right": {"Left": 4, "Right": 0}}
    parity = report["reference_parity"]
    assert parity["passed"] and parity["checked"] == 6
    test = parity["results"]["flipped"][0]
    assert test["features_match"] and test["prediction_matches"]
    assert test["reference_winning_slot"] == "swapped"
    assert test["reference_scores"]["swapped"]["confidence"] == pytest.approx(0.83)
    assert synthetic_inference == [
        (10, 11), (11, 10), (10, 12), (12, 10), (10, 13), (13, 10), (30, 31), (31, 30),
        (20, 21), (21, 20), "closed",
    ]
    assert set(tmp_path.iterdir()) == {path, report_path}


@pytest.mark.parametrize(("pin", "value", "message"), [
    ("ARCHIVE_SHA256", "0" * 64, "SHA-256 mismatch"),
    ("ARCHIVE_SIZE", 3, "archive size mismatch"),
])
def test_archive_mismatch_fails_before_zip_open_and_keeps_existing_report(
    tmp_path, archive, monkeypatch, capsys, pin, value, message,
):
    path, _ = archive
    report = tmp_path / "report.json"
    report.write_text("preserve", encoding="utf-8")

    def unexpected_zip(*_args):
        pytest.fail("archive integrity checks must happen before opening as a ZIP")

    monkeypatch.setattr(evaluator, pin, value)
    monkeypatch.setattr(evaluator, "ZipFile", unexpected_zip)
    assert evaluator.main(["--archive", str(path), "--report", str(report)]) == 1
    assert message in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == "preserve"


def test_archive_cannot_be_overwritten_by_report(archive, capsys):
    path, _ = archive
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert evaluator.main(["--archive", str(path), "--report", str(path)]) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_image_size_limits_apply_before_member_read_and_pixel_decode(archive, monkeypatch):
    path, entries = archive
    with ZipFile(path) as bundle:
        member = bundle.getinfo("Dataset/Testing/A/01.PNG")
        monkeypatch.setattr(evaluator, "MAX_IMAGE_BYTES", member.file_size - 1)

        def unexpected(*_args):
            pytest.fail("must reject size limit before reading/decoding")

        monkeypatch.setattr(bundle, "open", unexpected)
        with pytest.raises(ValueError, match="exceeds the image byte limit"):
            evaluator._read_image(bundle, member)
        monkeypatch.setattr(evaluator, "MAX_IMAGE_PIXELS", 23)
        monkeypatch.setattr(cv2, "imdecode", unexpected)
        assert evaluator._decode(entries[member.filename]) == (None, "image-too-large")


def test_reference_mismatch_keeps_report_but_fails_cli(tmp_path, archive, synthetic_inference, monkeypatch, capsys):
    path, _ = archive
    report = tmp_path / "report.json"
    runtime_features = evaluator.build_features
    monkeypatch.setattr(evaluator, "build_features", lambda detection: np.roll(runtime_features(detection), 1))
    assert evaluator.main(["--archive", str(path), "--report", str(report)]) == 1
    assert "reference mismatch" in capsys.readouterr().err
    parity = json.loads(report.read_text(encoding="utf-8"))["reference_parity"]
    assert parity["checked"] == 6 and parity["passed"] is False
    assert not parity["results"]["raw"][0]["features_match"]
