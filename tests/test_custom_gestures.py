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
