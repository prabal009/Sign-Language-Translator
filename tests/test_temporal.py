import pytest

from utils.recognition import Candidate, Decision
from utils.temporal import Stabilizer, Timing
from utils.text_actions import apply_action


RELEASE = Decision("no-hands")
UNCERTAIN = Decision("uncertain")
AMBIGUOUS = Decision("ambiguous")


def choice(label, source="alphabet"):
    return Decision("usable", candidate=Candidate(source, label))


def send(state, decision, *times):
    return [commit.label for when in times if (commit := state.observe(decision, when))]


def test_hello_needs_release_between_two_ls():
    stabilizer = Stabilizer()
    committed = []
    for kind, times in (
        (choice("H"), (0, 0.15, 0.3, 0.45)),
        (choice("E"), (0.50, 0.65, 0.80, 0.95)),
        (choice("L"), (1.00, 1.15, 1.30, 1.45)),
        (UNCERTAIN, (1.50,)),
        (choice("L"), (1.60, 1.75, 1.90, 2.05, 2.15)),
        (RELEASE, (2.20, 2.35, 2.50)),
        (choice("L"), (2.55, 2.70, 2.85, 3.00)),
        (choice("O"), (3.05, 3.20, 3.35, 3.50)),
    ):
        committed.extend(send(stabilizer, kind, *times))
    text = ""
    for action in committed:
        text = apply_action(text, action)
    assert text == "HELLO"


def test_text_actions_and_empty_or_unknown_no_ops():
    for label in ("BACKSPACE", "CLEAR_WORD", "CLEAR", "SPACE"):
        assert apply_action("", label) == ""
    text = apply_action("HELLO", "SPACE")
    assert apply_action(text, "SPACE") == "HELLO "
    text = apply_action("HELLO WORLD ", "CLEAR_WORD")
    assert text == "HELLO"
    assert apply_action(text, "BACKSPACE") == "HELL"
    assert apply_action(text, "CLEAR") == ""
    assert apply_action(apply_action(text, "0"), "1") == "HELLO01"
    assert apply_action(text, "OTHER") == text


def test_brief_interruption_freezes_hold_and_contradiction_restarts():
    state = Stabilizer()
    a, b = choice("A"), choice("B")
    assert not send(state, a, 0, 0.15, 0.30)
    assert not send(state, UNCERTAIN, 0.35)
    assert not send(state, a, 0.43, 0.55)
    assert send(state, a, 0.58) == ["A"]
    assert not send(state, b, 0.65, 0.80, 0.95)
    assert not send(state, a, 1.0)
    assert not send(state, b, 1.1, 1.25, 1.4)
    assert send(state, b, 1.55) == ["B"]


def test_no_commit_on_late_timestamp_gap_or_too_few_observations():
    state = Stabilizer(Timing(alphabet_hold=0.1))
    a = choice("A")
    assert not send(state, a, 0, 0.12, 0.12, 0.11)
    assert not send(state, a, 0.40, 0.55)
    assert send(state, a, 0.56) == ["A"]


def test_numeric_source_is_stabilized_in_its_own_lane():
    state = Stabilizer()
    assert not send(state, choice("I"), 0, 0.2, 0.4)
    one = choice("1", "custom")
    assert not send(state, one, 0.45, 0.60, 0.75, 0.90, 1.05)
    assert send(state, one, 1.15) == ["1"]


def test_destructive_fresh_rearm_ambiguity_and_read_failure():
    state = Stabilizer()
    delete = choice("BACKSPACE", "custom")
    assert not send(state, delete, *[i / 10 for i in range(13)])
    assert not send(state, RELEASE, 1.30, 1.44, 1.59)  # 0.29 s is not release.
    assert not send(state, delete, *[i / 10 for i in range(16, 29)])
    assert not send(state, RELEASE, 2.90, 3.05, 3.20)
    assert state.destructive_armed
    assert not send(state, delete, 3.3, 3.5, 3.7, 3.9, 4.1)
    assert not send(state, UNCERTAIN, 4.15)
    assert not send(state, delete, 4.2, 4.4, 4.6, 4.8, 5.0)
    assert send(state, delete, 5.2) == ["BACKSPACE"]
    assert not send(state, delete, *[i / 10 for i in range(53, 253)])
    assert not send(state, RELEASE, 25.3, 25.45, 25.6)
    assert not send(state, AMBIGUOUS, 25.7)
    assert not send(state, delete, *[i / 10 for i in range(258, 280)])
    assert not send(state, RELEASE, 28.0, 28.15, 28.3)
    assert send(state, delete, 28.4, 28.6, 28.8, 29.0, 29.2, 29.4) == ["BACKSPACE"]
    state.reset()
    assert not send(state, delete, *[i / 10 for i in range(13)])


def test_global_and_destructive_cooldowns_and_release_interruption():
    state = Stabilizer(Timing(0.1, 0.1, 0.1, 0.35, 1.25))
    delete = choice("BACKSPACE", "custom")
    assert not send(state, RELEASE, 0, 0.15)
    assert not send(state, UNCERTAIN, 0.2)
    assert not send(state, RELEASE, 0.25, 0.40, 0.55)
    assert send(state, delete, 0.60, 0.65, 0.70) == ["BACKSPACE"]
    assert not send(state, choice("H"), 0.75, 0.80, 0.85, 1.00)
    assert send(state, choice("H"), 1.05) == ["H"]
    assert not send(state, RELEASE, 1.1, 1.25, 1.4)
    clear_word = choice("CLEAR_WORD", "custom")
    assert not send(state, clear_word, 1.45, 1.55, 1.65, 1.85)
    assert send(state, clear_word, 1.95) == ["CLEAR_WORD"]


@pytest.mark.parametrize("values", [(0, 1, 1, 1, 1), (1, 2, 1, 1, 1), (1, 1, 1, 2, 1)])
def test_timing_rejects_invalid_order_and_nonpositive_values(values):
    with pytest.raises(ValueError):
        Timing(*values)


def test_progress_and_waiting_release_track_hold_state():
    state = Stabilizer(Timing(alphabet_hold=0.40, custom_hold=0.60, destructive_hold=0.80, cooldown=0.20, destructive_cooldown=0.50))
    assert state.progress == 0.0 and not state.waiting_release
    assert state.observe(choice("A"), 0.0) is None
    assert state.progress == 0.0 and not state.waiting_release
    assert state.observe(choice("A"), 0.20) is None
    assert state.progress == pytest.approx(0.5) and not state.waiting_release
    assert state.observe(choice("A"), 0.40).label == "A"
    assert state.progress == 0.0 and not state.waiting_release
    # Holding the same label after commit sets waiting_release until hands leave.
    assert state.observe(choice("A"), 0.50) is None
    assert state.waiting_release and state.progress == 0.0
    assert state.observe(RELEASE, 0.60) is None
    assert not state.waiting_release
    assert state.observe(RELEASE, 0.75) is None
    assert state.observe(RELEASE, 0.90) is None
    assert state.observe(choice("A"), 1.00) is None
    assert not state.waiting_release


def test_arm_destructive_on_start_allows_first_delete_then_requires_release():
    state = Stabilizer(
        Timing(0.45, 0.70, 0.75, 0.35, 0.90),
        arm_destructive_on_start=True,
    )
    delete = choice("CLEAR_WORD", "custom")
    assert state.destructive_armed
    assert send(state, delete, 0.0, 0.20, 0.40, 0.60, 0.75) == ["CLEAR_WORD"]
    assert not state.destructive_armed
    # Holding again without a no-hands release must NOT delete a second word.
    assert not send(state, delete, 1.70, 1.90, 2.10, 2.30, 2.50)
    assert not send(state, RELEASE, 2.60, 2.75, 2.90)
    assert state.destructive_armed
    assert send(state, delete, 3.00, 3.20, 3.40, 3.60, 3.75) == ["CLEAR_WORD"]
