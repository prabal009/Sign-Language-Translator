"""Evaluate the pinned RealSign test sample locally without extracting images."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from io import BytesIO
from math import dist, isclose
from pathlib import Path, PurePosixPath
from string import ascii_uppercase
from zipfile import BadZipFile, ZipFile

import numpy as np

from .detector import HandDetector
from .model_assets import COMMIT
from .recognition import Thresholds
from .signspeak import MODEL_VERSION, SignSpeakAlphabetRecognizer, build_features

ARCHIVE_COMMIT = "17c51dcc158b7b6359b7c6667edc51c7f148f3f4"
ARCHIVE_URL = (
    "https://media.githubusercontent.com/media/RealSign62/RealSign-Indian-Sign-Language-Dataset/"
    f"{ARCHIVE_COMMIT}/Dataset.zip"
)
ARCHIVE_SIZE = 656_689_688
ARCHIVE_SHA256 = "008cae248e346b8c31fbbea057fcc3f69c6909d29a88e6bb1fb0369f528de2b5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
PER_CLASS = 3
REFERENCE_PER_PASS = 3
THRESHOLD = Thresholds().alphabet
ORIENTATIONS = ("raw", "flipped")
FOCUS_PAIRS = ("IY", "QY", "SE", "VN", "ZB")


def _verify_archive(source):
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(min(1024 * 1024, ARCHIVE_SIZE + 1 - size)):
        size += len(chunk)
        if size > ARCHIVE_SIZE:
            break
        digest.update(chunk)
    if size != ARCHIVE_SIZE:
        raise ValueError(f"RealSign archive size mismatch (expected {ARCHIVE_SIZE:,} bytes)")
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("RealSign archive SHA-256 mismatch; use the pinned Dataset.zip")
    source.seek(0)


def _select(archive):
    candidates = {label: [] for label in ascii_uppercase}
    for member in archive.infolist():
        parts = member.filename.split("/")
        if (member.is_dir() or len(parts) not in (3, 4) or "\\" in member.filename
                or any(part in ("", ".", "..") for part in parts)):
            continue
        split, label, filename = parts[-3:]
        if split == "Testing" and label in candidates and PurePosixPath(filename).suffix.lower() in IMAGE_SUFFIXES:
            candidates[label].append(member)
    counts = {label: len(members) for label, members in candidates.items()}
    if not any(counts.values()):
        raise ValueError("RealSign archive contains no Testing/<LETTER> images")
    selected = [
        (label, member)
        for label, members in candidates.items()
        for member in sorted(members, key=lambda item: item.filename)[:PER_CLASS]
    ]
    return selected, counts


def _read_image(archive, member):
    if member.file_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Selected RealSign member exceeds the image byte limit: {member.filename}")
    with archive.open(member) as image:
        data = image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Selected RealSign member exceeds the image byte limit: {member.filename}")
    return data


def _decode(data):
    import cv2
    from PIL import Image  # Locked with the MediaPipe runtime via matplotlib.

    # Read dimensions before asking OpenCV to allocate the decoded BGR frame.
    try:
        with Image.open(BytesIO(data), formats=("JPEG", "PNG", "WEBP")) as header:
            if header.width * header.height > MAX_IMAGE_PIXELS:
                return None, "image-too-large"
    except Image.DecompressionBombError:
        return None, "image-too-large"
    except (OSError, ValueError):
        return None, "decode-failed"
    try:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None, "decode-failed"
    return (None, "decode-failed") if frame is None or not frame.size else (frame, None)


def _unscored(reason):
    return {
        "raw_top": None, "confidence": None, "winning_slot": None, "accepted": False,
        "reason": reason, "hand_count": None, "valid_detection": None, "raw_handedness": [],
    }


def _score(detector, model, frame):
    detection = detector.detect(frame)
    prediction = None
    if not detection.valid:
        reason = "invalid-detection"
    elif not detection.hands:
        reason = "no-hands"
    else:
        prediction = model.predict(detection)
        reason = "unusable-features" if prediction is None else (
            "accepted" if prediction.confidence >= THRESHOLD else "below-threshold"
        )
    result = _unscored(reason)
    result.update(
        hand_count=len(detection.hands), valid_detection=detection.valid,
        raw_handedness=[{"label": hand.label, "score": float(hand.score)} for hand in detection.hands],
    )
    if prediction is not None:
        result.update(raw_top=prediction.label, confidence=prediction.confidence,
                      winning_slot=prediction.slot, accepted=reason == "accepted")
    return result, detection, prediction


def _reference_check(detection, prediction, pipeline):
    # Independent transcription of the pinned MediaPipeService and ModelService:
    # index-interleaved coordinates, wrist-to-12 scale, Left then Right, separate prediction calls.
    def hand_vector(points):
        wrist = points[0]
        scale = dist(wrist, points[12])
        if scale < 1e-6:
            scale = 1.0
        return [(float(component) - origin) / scale for point in points
                for component, origin in zip(point, wrist)]

    hands = {hand.label: hand_vector(hand.points) for hand in detection.hands}
    empty = [0.0] * 63
    left, right = hands.get("Left", empty), hands.get("Right", empty)
    expected = np.array(left + right)
    actual_features = build_features(detection)
    max_error = None if actual_features is None else float(np.max(np.abs(actual_features - expected)))
    features_match = actual_features is not None and bool(np.allclose(actual_features, expected, atol=1e-12, rtol=1e-12))

    scores = {}
    for slot, vector in (("as-detected", expected), ("swapped", np.array(right + left))):
        row = pipeline.predict_proba(vector.reshape(1, -1))[0]
        top, runner_up = np.argsort(row)[::-1][:2]
        scores[slot] = {"label": str(pipeline.classes_[top]), "confidence": float(row[top]),
                        "margin": float(row[top] - row[runner_up])}
    expected_slot = ("swapped" if scores["swapped"]["confidence"] > scores["as-detected"]["confidence"]
                     else "as-detected")
    winner = scores[expected_slot]
    prediction_match = (
        prediction.slot == expected_slot and prediction.label == winner["label"]
        and isclose(prediction.confidence, winner["confidence"], rel_tol=1e-12, abs_tol=1e-12)
        and isclose(prediction.margin, winner["margin"], rel_tol=1e-12, abs_tol=1e-12)
    )
    return {
        "matches": features_match and prediction_match,
        "features_match": features_match, "max_feature_absolute_error": max_error,
        "prediction_matches": prediction_match, "reference_winning_slot": expected_slot,
        "reference_scores": scores,
    }


def _summarize(samples, orientation):
    reasons = Counter()
    slot_winners = Counter()
    coverage = {label: dict(evaluated=0, scored=0, accepted=0) for label in ascii_uppercase}
    top = {label: Counter() for label in ascii_uppercase}
    accepted = {label: Counter() for label in ascii_uppercase}
    for sample in samples:
        if "duplicate_of" in sample:
            continue
        expected, row = sample["expected"], sample["passes"][orientation]
        counts = coverage[expected]
        reasons[row["reason"]] += 1
        counts["evaluated"] += 1
        label = row["raw_top"]
        if label is None:
            continue
        counts["scored"] += 1
        top[expected][label] += 1
        slot_winners[row["winning_slot"]] += 1
        if row["accepted"]:
            counts["accepted"] += 1
            accepted[expected][label] += 1
    totals = {key: sum(counts[key] for counts in coverage.values()) for key in ("evaluated", "scored", "accepted")}
    totals.update(correct_top=sum(top[label][label] for label in ascii_uppercase),
                  accepted_correct=sum(accepted[label][label] for label in ascii_uppercase))
    totals.update(wrong_top=totals["scored"] - totals["correct_top"],
                  accepted_wrong=totals["accepted"] - totals["accepted_correct"],
                  rejected=totals["evaluated"] - totals["accepted"])
    return {
        "totals": totals, "coverage_by_letter": coverage, "reasons": dict(reasons),
        "winning_slots": dict(slot_winners), "top_confusion": top, "accepted_confusion": accepted,
        "targeted_confusions": {
            f"{actual}->{predicted}": {"raw_top": top[actual][predicted], "accepted": accepted[actual][predicted]}
            for pair in FOCUS_PAIRS for actual, predicted in (tuple(pair), tuple(reversed(pair)))
        },
    }


def _evaluate(archive, selected, detector, model):
    import cv2

    samples, first_occurrence = [], {}
    checks = {orientation: [] for orientation in ORIENTATIONS}
    single_hand_flip = {label: {"Left": 0, "Right": 0} for label in ("Left", "Right")}
    for label, member in selected:
        data = _read_image(archive, member)
        digest = hashlib.sha256(data).hexdigest()
        sample = {"filename": member.filename, "sha256": digest, "expected": label}
        samples.append(sample)
        if digest in first_occurrence:
            sample["duplicate_of"] = first_occurrence[digest]
            continue
        first_occurrence[digest] = member.filename
        frame, error = _decode(data)
        passes = sample["passes"] = {}
        if frame is None:
            for orientation in ORIENTATIONS:
                passes[orientation] = _unscored(error)
            continue
        for orientation in ORIENTATIONS:
            oriented = frame if orientation == "raw" else cv2.flip(frame, 1)
            passes[orientation], detection, prediction = _score(detector, model, oriented)
            if prediction is not None and len(checks[orientation]) < REFERENCE_PER_PASS:
                checks[orientation].append({
                    "filename": member.filename,
                    **_reference_check(detection, prediction, model.pipeline),
                })
        raw, flipped = (passes[orientation] for orientation in ORIENTATIONS)
        if all(row["valid_detection"] and row["hand_count"] == 1 for row in (raw, flipped)):
            single_hand_flip[raw["raw_handedness"][0]["label"]][flipped["raw_handedness"][0]["label"]] += 1
    return samples, checks, single_hand_flip


def evaluate(archive_path: Path) -> dict:
    with archive_path.open("rb") as source:
        _verify_archive(source)
        # Parse the very same open file, not the path again. Never extract members.
        with ZipFile(source) as archive:
            selected, counts = _select(archive)
            model = SignSpeakAlphabetRecognizer()
            with HandDetector(static_image_mode=True) as detector:
                samples, checks, single_hand_flip = _evaluate(archive, selected, detector, model)
    reference_results = [result for orientation in ORIENTATIONS for result in checks[orientation]]
    duplicates = sum("duplicate_of" in row for row in samples)
    return {
        "schema_version": 1,
        "scope": "Publisher test images; not an independent benchmark or a live signing trial.",
        "archive": {"source": ARCHIVE_URL, "bytes": ARCHIVE_SIZE, "sha256": ARCHIVE_SHA256},
        "model": {"version": MODEL_VERSION, "commit": COMMIT, "alphabet_threshold": THRESHOLD},
        "selection": {
            "method": "First three sorted image paths per A–Z; global SHA-256 deduplication, no refills.",
            "eligible_paths": sum(counts.values()), "eligible_by_letter": counts,
            "selected": len(samples), "duplicates": duplicates, "unique": len(samples) - duplicates,
            "selected_by_letter": {label: min(count, PER_CLASS) for label, count in counts.items()},
        },
        "orientations": {orientation: _summarize(samples, orientation) for orientation in ORIENTATIONS},
        "raw_to_flipped_single_hand": single_hand_flip,
        "handedness_note": "Reported MediaPipe labels, not anatomical ground truth; only valid single-hand pairs compared.",
        "reference_parity": {
            "method": "First three scored frames per orientation; independent features and separate direct slot scores.",
            "checked": len(reference_results), "passed": bool(reference_results) and all(
                row["matches"] for row in reference_results
            ),
            "results": checks,
        },
        "samples": samples,
    }


def _write_report(path, report):
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".realsign-report-", delete=False) as output:
            staged = Path(output.name)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="pinned RealSign Dataset.zip already on disk")
    parser.add_argument("--report", type=Path, required=True, help="destination JSON file")
    args = parser.parse_args(argv)
    try:
        if (args.archive.resolve() == args.report.resolve()
                or (args.report.exists() and args.archive.samefile(args.report))):
            raise ValueError("Report must not overwrite the input archive")
        report = evaluate(args.archive)
        _write_report(args.report, report)
    except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
        print(f"RealSign evaluation: {exc}", file=sys.stderr)
        return 1
    parity = report["reference_parity"]
    if not parity["passed"]:
        why = "reference mismatch" if parity["checked"] else "no usable frame for reference comparison"
        print(f"RealSign evaluation: {why}; evidence written to {args.report}", file=sys.stderr)
        return 1
    print(f"RealSign evaluation: wrote {report['selection']['unique']} unique samples to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
