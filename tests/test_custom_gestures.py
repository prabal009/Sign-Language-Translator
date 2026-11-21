import numpy as np
import pytest

from utils.custom_gestures import custom_eligible, pose_eligible, sample_detection
from utils.normalize import normalize_hand
from utils.observations import Detection, Hand, Prediction


def _hand(primary, label="Left", score=0.99):
    points = np.tile([0.4, 0.75, 0.05], (21, 1))
    for i, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        x = 0.38 + i * 0.025
        points[mcp] = (x, 0.62, 0.03)
        points[pip] = (x, 0.54, 0.04)
        points[tip] = (x, 0.4 if i == primary else 0.60, 0.06)
    return Hand(points, label, score)


@pytest.mark.parametrize("label,finger,count", [("0", 0, 2), ("1", 3, 2), ("BACKSPACE", 1, 1), ("CLEAR_WORD", 1, 2)])
def test_each_finger_mapping_and_rejection(label, finger, count):
    first = _hand(finger)
    second = _hand(finger, "Right")
    hands = (first,) if count == 1 else (first, second)
    assert pose_eligible(label, sample_detection(Detection(hands), 4 / 3))
    wrong_count = (first, second) if count == 1 else (first,)
    assert not pose_eligible(label, sample_detection(Detection(wrong_count), 4 / 3))
    wrong_finger = _hand((finger + 1) % 4)
    assert not pose_eligible(label, sample_detection(Detection((wrong_finger, *hands[1:])), 4 / 3))


def test_mirror_slot_order_no_mutation_and_signed_sum():
    hand = _hand(0)
    original = hand.points.copy()
    sample = sample_detection(Detection((hand,)), 1)
    assert np.all(sample.features[:63] == 0)
    np.testing.assert_array_equal(original, hand.points)
    centered = original - original[0]
    centered[:, 0] *= -1
    np.testing.assert_allclose(sample.features[63:], (centered / np.linalg.norm(centered, axis=1).max()).ravel())
    hand_data = [1, 2, -3] + [0] * 60
    assert normalize_hand(hand_data) != hand_data


def test_degenerate_or_wrong_direction_rejects_destructive_pose():
    for tip in ((0.405, 0.54, 0.04), (0.405, 0.68, 0.06)):
        hand = _hand(1)
        hand.points[12] = tip
        sample = sample_detection(Detection((hand,)), 4 / 3)
        assert not pose_eligible("BACKSPACE", sample)


def test_other_low_hand_score_and_weak_destructive_margin_do_not_act():
    sample = sample_detection(Detection((_hand(1, score=0.69),)), 1)
    assert pose_eligible("OTHER", sample)
    assert not custom_eligible(Prediction("OTHER", 0.99, 0.99), sample)
    assert not custom_eligible(Prediction("BACKSPACE", 0.99, 0.99), sample)
    sample = sample_detection(Detection((_hand(1),)), 1)
    assert not custom_eligible(Prediction("BACKSPACE", 0.97, 0.29), sample)
    assert custom_eligible(Prediction("BACKSPACE", 0.95, 0.30), sample)
    assert pose_eligible("SPACE", sample)


def test_geometric_custom_recognizer_handles_middle_fingers_and_separated_zero():
    from utils.custom_gestures import GeometricCustomGestureRecognizer

    geo = GeometricCustomGestureRecognizer()
    one_mid = Detection((_hand(1, "Left", 0.60),))
    s1 = sample_detection(one_mid, 4 / 3)
    p1 = geo.predict_detection(one_mid, s1)
    assert p1 is not None and p1.label == "BACKSPACE" and p1.slot == "geometric"
    assert custom_eligible(p1, s1)

    two_mid = Detection((_hand(1, "Left", 0.65), _hand(1, "Right", 0.65)))
    s2 = sample_detection(two_mid, 4 / 3)
    p2 = geo.predict_detection(two_mid, s2)
    assert p2 is not None and p2.label == "CLEAR_WORD" and custom_eligible(p2, s2)

    two_pinky = Detection((_hand(3, "Left", 0.80), _hand(3, "Right", 0.80)))
    s3 = sample_detection(two_pinky, 4 / 3)
    p3 = geo.predict_detection(two_pinky, s3)
    assert p3 is not None and p3.label == "1" and custom_eligible(p3, s3)

    # Touching index fingers (like E/F/X) must NOT trigger 0, while separated vertical index fingers DO.
    h_left = _hand(0, "Left", 0.90)
    h_right_touching = _hand(0, "Right", 0.90)
    s_touch = sample_detection(Detection((h_left, h_right_touching)), 4 / 3)
    assert geo.predict_detection(Detection((h_left, h_right_touching)), s_touch) is None

    shifted = _hand(0, "Right", 0.90)
    shifted_pts = shifted.points.copy()
    shifted_pts[:, 0] += 0.30
    h_right_sep = Hand(shifted_pts, "Right", 0.90)
    det_sep = Detection((h_left, h_right_sep))
    s_sep = sample_detection(det_sep, 4 / 3)
    p0 = geo.predict_detection(det_sep, s_sep)
    assert p0 is not None and p0.label == "0" and custom_eligible(p0, s_sep)
