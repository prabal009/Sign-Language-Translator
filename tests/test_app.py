import os
import signal
from time import monotonic, sleep

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from app import ModernSignLanguageApp
from utils.video_session import VideoSession
from utils.video_worker import Preview


class FakeVideoSession:
    """A camera-free implementation of the session contract, polled by the real Qt timer."""

    state = "idle"
    active = False
    camera_status = "Test camera stopped"
    alphabet_status = custom_status = "Not loaded"
    model_statuses = ("Test alphabet available", "Test custom gestures available")

    def __init__(self):
        self.starts = []
        self.events = []
        self.preview = None
        self.sequence = 0

    def start(self, settings, _thresholds, _timing):
        self.starts.append(settings.index)
        self.events.clear()
        self.preview = None
        self.state, self.active = "running", True
        self.camera_status = f"Test camera {settings.index}"
        self.alphabet_status, self.custom_status = self.model_statuses

    def stop(self):
        self.events.clear()
        self.preview = None
        self.state, self.active = "idle", False
        self.camera_status = "Test camera stopped"
        self.alphabet_status = self.custom_status = "Not loaded"

    def tick(self):
        events, self.events = self.events, []
        preview, self.preview = self.preview, None
        return events, preview

    def action(self, label, timestamp=None):
        assert self.active
        self.events.append({"kind": "action", "label": label,
                            "timestamp": monotonic() if timestamp is None else timestamp})

    def frame(self, color, width=16, height=8):
        assert self.active
        self.sequence += 1
        data = bytes((color.red(), color.green(), color.blue())) * (width * height)
        self.preview = Preview(data, width, height, self.sequence)


def _show(qtbot, worker):
    window = ModernSignLanguageApp(worker=worker)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    return window


@pytest.fixture
def desktop(qtbot):
    worker = FakeVideoSession()
    return _show(qtbot, worker), worker


def _seed(qtbot, window, worker, text):
    window.start_button.click()
    for char in text:
        worker.action("SPACE" if char == " " else char)
    qtbot.waitUntil(lambda: window.output.toPlainText() == text)


def _focus(qtbot, widget):
    widget.setFocus()
    qtbot.waitUntil(widget.hasFocus)


def test_default_window_is_idle_and_manual_editing_does_not_open_camera(qtbot, desktop):
    window, worker = desktop
    assert window.isVisible() and not window.isFullScreen()
    assert window.preview.image is None
    assert window.output.toPlainText() == ""
    _focus(qtbot, window.output)
    for key in (Qt.Key.Key_Space, Qt.Key.Key_Backspace, Qt.Key.Key_C):
        qtbot.keyClick(window.output, key)
    qtbot.keyClick(window.output, Qt.Key.Key_Backspace, Qt.KeyboardModifier.ControlModifier)
    assert window.output.toPlainText() == ""
    assert worker.starts == [] and not worker.active

    _seed(qtbot, window, worker, "AB")
    window.stop_button.click()
    _focus(qtbot, window.output)
    qtbot.keyClick(window.output, Qt.Key.Key_Backspace)
    assert window.output.toPlainText() == "A"
    assert len(worker.starts) == 1 and not worker.active


@pytest.mark.parametrize("model_problem", ["missing", "corrupt"])
def test_unavailable_models_show_recovery_and_leave_manual_editing_enabled(qtbot, desktop, model_problem):
    window, worker = desktop
    _seed(qtbot, window, worker, "HELLO WORLD")
    window.stop_button.click()
    worker.model_statuses = (
        f"SignSpeak {model_problem}; run python -m utils.model_assets install",
        f"Custom model {model_problem}; run python train_model.py",
    )
    window.start_button.click()
    qtbot.waitUntil(lambda: "utils.model_assets install" in window.alphabet_status.text()
                    and "train_model.py" in window.custom_status.text())

    _focus(qtbot, window.output)
    qtbot.keyClick(window.output, Qt.Key.Key_Space)
    qtbot.keyClick(window.output, Qt.Key.Key_Space)
    assert window.output.toPlainText() == "HELLO WORLD "
    qtbot.keyClick(window.output, Qt.Key.Key_Backspace, Qt.KeyboardModifier.ControlModifier)
    assert window.output.toPlainText() == "HELLO"
    qtbot.keyClick(window.output, Qt.Key.Key_Backspace)
    assert window.output.toPlainText() == "HELL"
    qtbot.keyClick(window.output, Qt.Key.Key_C)
    assert window.output.toPlainText() == ""
    assert len(worker.starts) == 2 and worker.active


def _rendered_extents(widget, color):
    image = widget.grab().toImage()
    mid_x, mid_y = image.width() // 2, image.height() // 2
    width = sum(image.pixelColor(x, mid_y) == color for x in range(image.width()))
    height = sum(image.pixelColor(mid_x, y) == color for y in range(image.height()))
    return width, height


def test_pause_keeps_original_preview_and_resize_preserves_aspect(qtbot, desktop):
    window, worker = desktop
    window.resize(640, 480)
    small_size = window.size()
    window.start_button.click()
    color = QColor(18, 222, 171)
    worker.frame(color)
    qtbot.waitUntil(lambda: window.preview.image is not None)
    window.pause_button.click()
    assert not worker.active
    before = _rendered_extents(window.preview, color)
    assert before[1] > 0 and abs(before[0] - 2 * before[1]) <= 2

    window.resize(window.width() + 160, window.height() + 120)
    qtbot.waitUntil(lambda: _rendered_extents(window.preview, color)[0] > before[0])
    width, height = _rendered_extents(window.preview, color)
    assert height > before[1] and abs(width - 2 * height) <= 2
    original = window.preview.image
    assert (original.width(), original.height(), original.pixelColor(0, 0)) == (16, 8, color)

    window.resize(small_size)
    qtbot.waitUntil(lambda: _rendered_extents(window.preview, color) == before)
    assert not worker.active and len(worker.starts) == 1


def test_camera_switch_while_running_and_stop_preserve_text(qtbot, desktop):
    window, worker = desktop
    _seed(qtbot, window, worker, "A")
    initial = window.camera_index.value()
    other = 31 if initial != 31 else 32
    editor = window.camera_index.lineEdit()
    _focus(qtbot, editor)
    editor.selectAll()
    qtbot.keyClicks(editor, str(other))
    assert worker.starts == [initial]
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: worker.starts == [initial, other])
    assert window.output.toPlainText() == "A"

    worker.frame(QColor(18, 222, 171))
    qtbot.waitUntil(lambda: window.preview.image is not None)
    window.stop_button.click()
    assert not worker.active and window.preview.image is None
    assert window.output.toPlainText() == "A"
    window.camera_index.setValue(other + 1)
    assert worker.starts == [initial, other]


def test_clear_does_not_apply_pre_clear_queued_commit(qtbot, desktop):
    window, worker = desktop
    _seed(qtbot, window, worker, "A")
    before_clear = monotonic()
    qtbot.waitUntil(lambda: monotonic() > before_clear)
    worker.action("Z", before_clear)
    window.clear_button.click()  # No event-loop turn between queuing Z and clearing.
    assert window.output.toPlainText() == ""

    after_clear = monotonic()
    qtbot.waitUntil(lambda: monotonic() > after_clear)
    worker.action("Y", before_clear)  # Also reject delayed delivery of an earlier commit.
    worker.action("B")
    qtbot.waitUntil(lambda: window.output.toPlainText() == "B")


def test_copy_button_and_shortcut_work_without_a_camera(qtbot, qapp, desktop):
    window, worker = desktop
    _seed(qtbot, window, worker, "AB")
    window.stop_button.click()
    clipboard = qapp.clipboard()
    saved = clipboard.text()
    try:
        clipboard.setText("before copy")
        window.copy_button.click()
        assert clipboard.text() == "AB"
        clipboard.setText("before keyboard copy")
        _focus(qtbot, window.copy_button)
        qtbot.keyClick(window.copy_button, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        assert clipboard.text() == "AB"
        assert window.output.toPlainText() == "AB" and not worker.active
    finally:
        clipboard.setText(saved)


def test_application_shortcuts_are_suppressed_in_camera_editor(qtbot, qapp, desktop):
    window, worker = desktop
    _seed(qtbot, window, worker, "AB CD")
    window.stop_button.click()
    window.camera_index.setValue(12)
    editor = window.camera_index.lineEdit()
    _focus(qtbot, editor)
    editor.deselect()
    editor.setCursorPosition(len(editor.text()))

    clipboard = qapp.clipboard()
    saved = clipboard.text()
    try:
        clipboard.setText("camera editor sentinel")
        qtbot.keyClick(editor, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        assert clipboard.text() == "camera editor sentinel"
    finally:
        clipboard.setText(saved)

    qtbot.keyClick(editor, Qt.Key.Key_Backspace)
    assert len(editor.text()) == 1
    for key, modifier in (
        (Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier),
        (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
        (Qt.Key.Key_Backspace, Qt.KeyboardModifier.ControlModifier),
        (Qt.Key.Key_Q, Qt.KeyboardModifier.NoModifier),
        (Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    ):
        qtbot.keyClick(editor, key, modifier)
        assert window.output.toPlainText() == "AB CD"
        assert window.isVisible()
    assert not worker.active and len(worker.starts) == 1


@pytest.mark.parametrize("key", [Qt.Key.Key_Q, Qt.Key.Key_Escape])
def test_quit_shortcut_closes_idle_window(qtbot, desktop, key):
    window, worker = desktop
    _focus(qtbot, window.output)
    qtbot.keyClick(window.output, key)
    assert not window.isVisible() and worker.starts == []


def test_manual_buttons_live_readout_settings_and_guide(qtbot, desktop):
    from app import SettingsDialog, SignGuideDialog

    window, worker = desktop
    _seed(qtbot, window, worker, "HI")
    window.space_button.click()
    assert window.output.toPlainText() == "HI "
    worker.action("Y")
    worker.action("O")
    qtbot.waitUntil(lambda: window.output.toPlainText() == "HI YO")
    window.backspace_button.click()
    assert window.output.toPlainText() == "HI Y"
    window.clear_word_button.click()
    assert window.output.toPlainText() == "HI"

    worker.events.append({
        "kind": "prediction", "state": "usable", "reason": "", "source": "alphabet", "label": "K",
        "alphabet": ["K", 0.91], "custom": None, "progress": 75, "waiting_release": False,
    })
    qtbot.waitUntil(lambda: window.sign_badge.text() == "K")
    assert window.confidence_bar.value() == 91
    assert window.hold_bar.value() == 75
    assert window.hold_value.text() == "75%"

    worker.events.append({
        "kind": "prediction", "state": "usable", "reason": "Lower hands briefly to repeat this sign",
        "source": "alphabet", "label": "K", "alphabet": ["K", 0.91], "custom": None,
        "progress": 0, "waiting_release": True,
    })
    qtbot.waitUntil(lambda: window.hold_value.text() == "Release")

    dialog = SettingsDialog(window.thresholds, window.timing, "dark", window)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.destructive_threshold.setValue(0.40)
    dialog.apply_button.click()
    assert dialog.error_label.isVisible()
    dialog.reset_button.click()
    dialog.mirror_box.setChecked(True)
    dialog.theme_box.setCurrentIndex(dialog.theme_box.findData("light"))
    dialog.apply_button.click()
    assert dialog.result() == SettingsDialog.DialogCode.Accepted
    assert dialog.theme == "light" and dialog.mirror_input is True
    window._apply_theme("light")
    assert window._theme == "light"

    guide = SignGuideDialog(window)
    qtbot.addWidget(guide)
    guide.show()
    assert guide.isVisible()
    guide.close()


def _stuck_camera(_settings, _backend, _thresholds, _timing, channels):
    if os.name == "posix":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    channels.publish({"kind": "opened", "message": "Test camera entered blocked native read"})
    sleep(20)  # Simulate an uninterruptible capture: do not observe channels.stop.


def test_qt_heartbeat_and_close_while_spawned_camera_is_stuck(qtbot):
    worker = VideoSession(target=_stuck_camera, backends=("ANY",))
    window = _show(qtbot, worker)
    heartbeat = QTimer(window)
    heartbeat.setInterval(20)
    beats = []
    heartbeat.timeout.connect(lambda: beats.append(worker.active and window.isVisible()))
    try:
        window.start_button.click()
        qtbot.waitUntil(lambda: worker.state == "initializing", timeout=8000)
        heartbeat.start()
        window.close()
        assert window.isVisible() and worker.active
        qtbot.waitUntil(lambda: beats.count(True) >= 3, timeout=600)
        qtbot.waitUntil(lambda: not worker.active and not window.isVisible(), timeout=5000)
    finally:
        heartbeat.stop()
        if worker.active:
            worker.stop()

            def reaped():
                worker.tick()
                return not worker.active

            qtbot.waitUntil(reaped, timeout=5000)
