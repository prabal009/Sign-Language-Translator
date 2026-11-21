from types import SimpleNamespace

import numpy as np
import pytest

from utils.observations import Detection, Hand, Prediction
from utils.recognition import Arbiter, Candidate, Thresholds


def _middle(label="Left", score=0.95):
    points = np.zeros((21, 3))
    points[:, 1] = 0.8
    for i, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        points[[mcp, pip, tip], 0] = i * 0.05
        points[[mcp, pip, tip], 1] = [0.6, 0.50, 0.3 if i == 1 else 0.57]
    return Hand(points, label, score)


def _model(label, confidence, margin):
    return SimpleNamespace(predict=lambda _: Prediction(label, confidence, margin))


def test_invalid_is_not_release_and_no_hands_never_infers():
    arbiter = Arbiter(object(), object())
    assert arbiter.observe(Detection(), 1).kind == "no-hands"
    assert arbiter.observe(Detection(valid=False), 1).kind == "uncertain"
    assert arbiter.observe(Detection((_middle(), _middle())), 1).kind == "uncertain"


def test_destructive_requires_valid_below_threshold_alphabet_score():
    detection = Detection((_middle(),))
    delete = _model("BACKSPACE", 0.99, 0.98)
    unavailable = Arbiter(None, delete).observe(detection, 1)
    assert unavailable.kind == "uncertain" and unavailable.candidate is None
    assert "need SignSpeak" in unavailable.reason
    assert Arbiter(SimpleNamespace(predict=lambda _: None), delete).observe(detection, 1).candidate is None
    below = Arbiter(_model("A", 0.49, 0.1), delete).observe(detection, 1)
    assert below.candidate == Candidate("custom", "BACKSPACE") and below.alphabet.label == "A"
    at_floor = Arbiter(_model("A", 0.50, 0.1), delete).observe(detection, 1)
    assert at_floor.kind == "ambiguous" and at_floor.candidate is None
    assert at_floor.alphabet.label == "A" and at_floor.custom.label == "BACKSPACE"

    # Geometric middle-finger pose is not blocked by moderate closed-set alphabet guesses (<0.90),
    # but still yields ambiguity if alphabet confidence is >=0.90.
    geo_delete = SimpleNamespace(predict=lambda _: Prediction("BACKSPACE", 0.98, 0.90, slot="geometric"))
    mid_alpha = Arbiter(_model("I", 0.72, 0.2), geo_delete).observe(detection, 1)
    assert mid_alpha.kind == "usable" and mid_alpha.candidate == Candidate("custom", "BACKSPACE")
    high_alpha = Arbiter(_model("I", 0.92, 0.6), geo_delete).observe(detection, 1)
    assert high_alpha.kind == "ambiguous" and high_alpha.candidate is None


def test_nondestructive_custom_priority_and_guarded_fallback():
    detection = Detection((_middle(),))
    alphabet = _model("I", 0.99, 0.98)
    assert Arbiter(alphabet, _model("SPACE", 0.86, 0.4)).observe(detection, 1).candidate == (
        Candidate("custom", "SPACE")
    )
    for custom in (_model("OTHER", 1, 1), _model("BACKSPACE", 0.99, 0.29), _model("0", 1, 1)):
        assert Arbiter(alphabet, custom).observe(detection, 1).candidate == Candidate("alphabet", "I")
    assert Arbiter(alphabet, None).observe(detection, 1).custom is None
    assert Arbiter(alphabet, _model("SPACE", 0.99, 0.98)).observe(
        Detection((_middle(score=0.65),)), 1
    ).candidate == Candidate("alphabet", "I")


@pytest.mark.parametrize("params", [(1.1, 0.85, 0.95), (0.5, 0.8, 0.7), (0.5, 0.8, float("nan"))])
def test_invalid_thresholds_rejected(params):
    with pytest.raises(ValueError):
        Thresholds(*params)
