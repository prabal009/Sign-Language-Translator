"""Native desktop sign-language translator. Cameras open only on Start."""

import argparse
import json
import multiprocessing as mp
import signal
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import QEvent, QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from utils.camera import CameraSettings, camera_choices, default_camera
from utils.config import MAX_INDEX
from utils.custom_gestures import ACTION_LABELS, DESCRIPTIONS
from utils.model_assets import configure_qt_identity
from utils.recognition import Thresholds
from utils.temporal import Timing
from utils.text_actions import apply_action
from utils.video_session import VideoSession

_THEMES = ("dark", "light", "system")

_DARK_QSS = """
QMainWindow, QDialog { background: #16181c; color: #eceef2; }
QLabel { color: #eceef2; background: transparent; }
QWidget#toolbar, QWidget#panel, QWidget#bottomBar {
    background: #1d2026;
    border: 1px solid #2b303a;
    border-radius: 6px;
}
QFrame#divider {
    background: #2b303a;
    max-height: 1px;
}
QFrame#signCard {
    background: #15181d;
    border: 1px solid #2b303a;
    border-radius: 6px;
}
QFrame#previewFrame {
    background: #0b0d10;
    border: 1px solid #2b303a;
    border-radius: 6px;
}
QLabel#sectionTitle {
    color: #9ba3b2;
    font-size: 11px;
    font-weight: 600;
}
QLabel#signBadge {
    color: #f3f5f8;
    font-size: 32px;
    font-weight: 700;
}
QLabel#signSource {
    color: #a6b0c0;
    font-size: 12px;
}
QLabel#statusLine {
    color: #cfd4de;
    font-size: 12px;
}
QLabel#muted {
    color: #9ba3b2;
    font-size: 11px;
}
QPushButton {
    background: #282c35;
    color: #eceef2;
    border: 1px solid #383e4b;
    border-radius: 5px;
    padding: 5px 11px;
    font-weight: 500;
}
QPushButton:hover { background: #323744; border-color: #4b5365; }
QPushButton:pressed { background: #22262e; }
QPushButton:disabled { background: #20232a; color: #6c7382; border-color: #2b303a; }
QPushButton#primaryButton {
    background: #2b6cb0;
    border-color: #367ec9;
    color: #ffffff;
}
QPushButton#primaryButton:hover { background: #3379c4; }
QPushButton#primaryButton:disabled { background: #22384f; color: #8aa3bd; border-color: #28425c; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #13161b;
    color: #eceef2;
    border: 1px solid #343a46;
    border-radius: 5px;
    padding: 4px 8px;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border-color: #5b9bd5;
}
QPlainTextEdit {
    background: #111318;
    color: #f2f4f8;
    border: 1px solid #2e3440;
    border-radius: 5px;
    padding: 6px 8px;
}
QProgressBar {
    background: #12151a;
    border: 1px solid #2c313c;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #4e9a78;
    border-radius: 3px;
}
QProgressBar#holdBar::chunk {
    background: #4f8dd6;
}
"""

_LIGHT_QSS = """
QMainWindow, QDialog { background: #f3f5f8; color: #1b2028; }
QLabel { color: #1b2028; background: transparent; }
QWidget#toolbar, QWidget#panel, QWidget#bottomBar {
    background: #ffffff;
    border: 1px solid #d7dde6;
    border-radius: 6px;
}
QFrame#divider {
    background: #d7dde6;
    max-height: 1px;
}
QFrame#signCard {
    background: #f7f9fc;
    border: 1px solid #d7dde6;
    border-radius: 6px;
}
QFrame#previewFrame {
    background: #111418;
    border: 1px solid #c9d1dc;
    border-radius: 6px;
}
QLabel#sectionTitle {
    color: #5a6475;
    font-size: 11px;
    font-weight: 600;
}
QLabel#signBadge {
    color: #141922;
    font-size: 32px;
    font-weight: 700;
}
QLabel#signSource {
    color: #525c6d;
    font-size: 12px;
}
QLabel#statusLine {
    color: #2c3442;
    font-size: 12px;
}
QLabel#muted {
    color: #5c6677;
    font-size: 11px;
}
QPushButton {
    background: #f0f3f7;
    color: #1b2028;
    border: 1px solid #c7d0dc;
    border-radius: 5px;
    padding: 5px 11px;
    font-weight: 500;
}
QPushButton:hover { background: #e5ebf2; border-color: #aeb9c8; }
QPushButton:pressed { background: #dce3ec; }
QPushButton:disabled { background: #f5f7fa; color: #96a0b0; border-color: #dfe5ed; }
QPushButton#primaryButton {
    background: #2563a8;
    border-color: #1f5591;
    color: #ffffff;
}
QPushButton#primaryButton:hover { background: #2c70bc; }
QPushButton#primaryButton:disabled { background: #b8cee6; color: #f5f8fc; border-color: #a7c2df; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    color: #1b2028;
    border: 1px solid #c5cedb;
    border-radius: 5px;
    padding: 4px 8px;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border-color: #2563a8;
}
QPlainTextEdit {
    background: #fafbfd;
    color: #141922;
    border: 1px solid #cfd7e2;
    border-radius: 5px;
    padding: 6px 8px;
}
QProgressBar {
    background: #edf1f6;
    border: 1px solid #cfd7e2;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2f855a;
    border-radius: 3px;
}
QProgressBar#holdBar::chunk {
    background: #2563a8;
}
"""


def _label(text="", *, object_name=None):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if object_name:
        label.setObjectName(object_name)
    return label


def _dark_palette():
    palette = QPalette()
    colors = {
        "Window": "#16181c", "WindowText": "#eceef2", "Base": "#111318", "Text": "#eceef2",
        "AlternateBase": "#232730", "Button": "#282c35", "ButtonText": "#eceef2",
        "Highlight": "#4f8dd6", "HighlightedText": "#ffffff", "PlaceholderText": "#8f97a6",
    }
    for role, color in colors.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    for role in ("WindowText", "Text", "ButtonText"):
        palette.setColor(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, role), QColor("#7c8392"))
    return palette


def _light_palette():
    palette = QPalette()
    colors = {
        "Window": "#f3f5f8", "WindowText": "#1b2028", "Base": "#ffffff", "Text": "#1b2028",
        "AlternateBase": "#edf1f6", "Button": "#f0f3f7", "ButtonText": "#1b2028",
        "Highlight": "#2563a8", "HighlightedText": "#ffffff", "PlaceholderText": "#707a8a",
    }
    for role, color in colors.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    for role in ("WindowText", "Text", "ButtonText"):
        palette.setColor(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, role), QColor("#96a0b0"))
    return palette


def _reference_chart_path() -> Path | None:
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "docs" / "realsign-alphabet-reference.png")
    candidates.append(Path(__file__).resolve().parent / "docs" / "realsign-alphabet-reference.png")
    return next((path for path in candidates if path.is_file()), None)


class PreviewLabel(QLabel):
    def __init__(self, theme):
        super().__init__()
        self.image = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAccessibleName("Mirrored camera preview")
        self.apply_theme(theme)
        self.reset_frame()

    def apply_theme(self, theme):
        if theme == "system":
            self.setStyleSheet("")
        else:
            self.setStyleSheet("background: #0b0d10; color: #cfd4de; border: none;")

    def reset_frame(self):
        self.image = None
        self.setPixmap(QPixmap())
        self.setText("Camera preview — press Start")

    def set_frame(self, frame):
        self.image = QImage(frame.data, frame.width, frame.height, frame.width * 3, QImage.Format.Format_RGB888).copy()
        self._scale_image(smooth=False)

    def _scale_image(self, *, smooth=True):
        if self.image is not None and not self.size().isEmpty():
            mode = Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
            self.setPixmap(QPixmap.fromImage(self.image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, mode,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_image(smooth=True)


class SettingsDialog(QDialog):
    """Adjust recognition thresholds, hold/cooldown timing, mirroring, and theme without restarting."""

    def __init__(self, thresholds: Thresholds, timing: Timing, theme: str, parent=None, *, mirror_input: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Recognition Settings")
        self.setMinimumWidth(400)
        self.thresholds = thresholds
        self.timing = timing
        self.theme = theme
        self.mirror_input = mirror_input

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        self.theme_box = QComboBox()
        for option in _THEMES:
            self.theme_box.addItem(option.title(), option)
        self.theme_box.setCurrentIndex(max(0, self.theme_box.findData(theme)))
        form.addRow("Theme", self.theme_box)

        self.mirror_box = QCheckBox("Flip input horizontally (left-handed / mirrored webcam)")
        self.mirror_box.setChecked(mirror_input)
        form.addRow("Input orientation", self.mirror_box)

        self.alphabet_threshold = self._spin(0.10, 0.99, 0.05, thresholds.alphabet, 2)
        self.custom_threshold = self._spin(0.10, 0.99, 0.05, thresholds.custom, 2)
        self.destructive_threshold = self._spin(0.10, 0.99, 0.05, thresholds.destructive, 2)
        self.alphabet_hold = self._spin(0.10, 3.00, 0.05, timing.alphabet_hold, 2, " s")
        self.custom_hold = self._spin(0.10, 3.00, 0.05, timing.custom_hold, 2, " s")
        self.destructive_hold = self._spin(0.10, 3.00, 0.05, timing.destructive_hold, 2, " s")
        self.cooldown = self._spin(0.05, 3.00, 0.05, timing.cooldown, 2, " s")
        self.destructive_cooldown = self._spin(0.05, 5.00, 0.05, timing.destructive_cooldown, 2, " s")

        form.addRow("Alphabet confidence floor", self.alphabet_threshold)
        form.addRow("Custom gesture confidence floor", self.custom_threshold)
        form.addRow("Destructive command confidence floor", self.destructive_threshold)
        form.addRow("Alphabet hold duration", self.alphabet_hold)
        form.addRow("Custom gesture hold duration", self.custom_hold)
        form.addRow("Destructive command hold duration", self.destructive_hold)
        form.addRow("Global commit cooldown", self.cooldown)
        form.addRow("Destructive command cooldown", self.destructive_cooldown)
        layout.addLayout(form)

        self.error_label = _label("", object_name="muted")
        self.error_label.setStyleSheet("color: #e05d5d;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Reset Defaults")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        self.reset_button.clicked.connect(self._reset_defaults)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self._submit)

    @staticmethod
    def _spin(minimum, maximum, step, value, decimals, suffix=""):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        if suffix:
            box.setSuffix(suffix)
        return box

    def _reset_defaults(self):
        defaults_t, defaults_timing = Thresholds(), Timing()
        self.mirror_box.setChecked(False)
        self.alphabet_threshold.setValue(defaults_t.alphabet)
        self.custom_threshold.setValue(defaults_t.custom)
        self.destructive_threshold.setValue(defaults_t.destructive)
        self.alphabet_hold.setValue(defaults_timing.alphabet_hold)
        self.custom_hold.setValue(defaults_timing.custom_hold)
        self.destructive_hold.setValue(defaults_timing.destructive_hold)
        self.cooldown.setValue(defaults_timing.cooldown)
        self.destructive_cooldown.setValue(defaults_timing.destructive_cooldown)
        self.error_label.hide()

    def _submit(self):
        try:
            self.thresholds = Thresholds(
                self.alphabet_threshold.value(),
                self.custom_threshold.value(),
                self.destructive_threshold.value(),
            )
            self.timing = Timing(
                self.alphabet_hold.value(),
                self.custom_hold.value(),
                self.destructive_hold.value(),
                self.cooldown.value(),
                self.destructive_cooldown.value(),
            )
        except ValueError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            return
        self.theme = self.theme_box.currentData() or "dark"
        self.mirror_input = self.mirror_box.isChecked()
        self.accept()


class SignGuideDialog(QDialog):
    """Offline A–Z training reference chart and custom command definitions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ISL Sign Reference Guide")
        self.resize(920, 680)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = _label(
            "A–Z uses the pretrained SignSpeak 1.0.0-realsign model trained on the CC0-1.0 RealSign ISL dataset. "
            "Hold a static sign steadily until the Hold bar completes. Lower both hands briefly to repeat a letter.",
            object_name="statusLine",
        )
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setSpacing(10)

        chart_path = _reference_chart_path()
        self.chart_label = QLabel()
        self.chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if chart_path is not None and not (pixmap := QPixmap(str(chart_path))).isNull():
            self.chart_label.setPixmap(pixmap.scaledToWidth(
                min(860, pixmap.width()), Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            self.chart_label.setTextFormat(Qt.TextFormat.PlainText)
            self.chart_label.setWordWrap(True)
            self.chart_label.setText(
                "Reference chart image not found (docs/realsign-alphabet-reference.png). "
                "See README.md for the RealSign A–Z reference."
            )
        body.addWidget(self.chart_label)

        custom_heading = _label("OPTIONAL CUSTOM WEBCAM COMMANDS (REQUIRE LOCAL TRAINING)", object_name="sectionTitle")
        body.addWidget(custom_heading)
        for label in ACTION_LABELS:
            body.addWidget(_label(f"{label}: {DESCRIPTIONS[label]}", object_name="statusLine"))

        body.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addWidget(_label("Source: RealSign ISL Dataset (CC0-1.0)", object_name="muted"))
        footer.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)


class ModernSignLanguageApp(QMainWindow):
    def __init__(self, camera=None, *, thresholds=Thresholds(), timing=Timing(), theme="dark", worker=None):
        super().__init__()
        self.camera = CameraSettings(default_camera()) if camera is None else camera
        self.thresholds, self.timing = thresholds, timing
        self._theme = theme if theme in _THEMES else "dark"
        self.worker = VideoSession() if worker is None else worker
        self._mode = "idle"
        self._text = ""
        self._clear_at = float("-inf")
        self._smoke = None
        self._guide_dialog = None
        self.setWindowTitle("Sign Language Translator")
        self.setMinimumSize(640, 480)
        self.resize(960, 660)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Top control bar
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(6)

        chooser = QHBoxLayout()
        chooser.setSpacing(6)
        self.camera_choice = QComboBox()
        self.camera_choice.setAccessibleName("Camera choices")
        self.camera_choice.setPlaceholderText("Manual camera index")
        self.camera_choice.setMinimumContentsLength(12)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, MAX_INDEX)
        self.camera_index.setMaximumWidth(78)
        self.camera_index.setKeyboardTracking(False)
        self.camera_index.setValue(self.camera.index)
        self.camera_index.setAccessibleName("Camera index")
        self.camera_index.setToolTip("Enter a nonnegative index and press Enter to select it.")
        self.refresh_button = QPushButton("Refresh")
        self.guide_button = QPushButton("Sign Guide")
        self.guide_button.setToolTip("View exact RealSign A–Z training signs and custom commands")
        self.settings_button = QPushButton("Settings")
        self.settings_button.setToolTip("Adjust confidence thresholds, hold durations, cooldowns, and theme")
        chooser.addWidget(_label("Camera"))
        chooser.addWidget(self.camera_choice, 1)
        chooser.addWidget(_label("Index"))
        chooser.addWidget(self.camera_index)
        chooser.addWidget(self.refresh_button)
        chooser.addWidget(self.guide_button)
        chooser.addWidget(self.settings_button)
        toolbar_layout.addLayout(chooser)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        for button in (self.start_button, self.pause_button, self.stop_button):
            controls.addWidget(button)
        controls.addStretch()
        hint = _label("Shortcuts: Space · Backspace · Ctrl+Backspace · C · Ctrl+C", object_name="muted")
        hint.setWordWrap(False)
        controls.addWidget(hint)
        toolbar_layout.addLayout(controls)
        layout.addWidget(toolbar)

        # Central stage: preview (left) + Live Recognition & System Status inspector (right)
        stage = QHBoxLayout()
        stage.setSpacing(8)

        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = PreviewLabel(self._theme)
        preview_layout.addWidget(self.preview)
        stage.addWidget(preview_frame, 1)

        inspector = QWidget()
        inspector.setObjectName("panel")
        inspector.setMinimumWidth(210)
        inspector.setMaximumWidth(275)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(8)

        inspector_layout.addWidget(_label("LIVE RECOGNITION", object_name="sectionTitle"))
        sign_card = QFrame()
        sign_card.setObjectName("signCard")
        sign_card_layout = QVBoxLayout(sign_card)
        sign_card_layout.setContentsMargins(10, 10, 10, 10)
        sign_card_layout.setSpacing(2)
        self.sign_badge = _label("—", object_name="signBadge")
        self.sign_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_badge.setAccessibleName("Detected sign")
        self.sign_source = _label("Idle", object_name="signSource")
        self.sign_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sign_card_layout.addWidget(self.sign_badge)
        sign_card_layout.addWidget(self.sign_source)
        inspector_layout.addWidget(sign_card)

        conf_header = QHBoxLayout()
        conf_header.addWidget(_label("Confidence", object_name="muted"))
        conf_header.addStretch()
        self.confidence_value = _label("—", object_name="muted")
        conf_header.addWidget(self.confidence_value)
        inspector_layout.addLayout(conf_header)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setAccessibleName("Recognition confidence")
        inspector_layout.addWidget(self.confidence_bar)

        hold_header = QHBoxLayout()
        hold_header.addWidget(_label("Hold to commit", object_name="muted"))
        hold_header.addStretch()
        self.hold_value = _label("0%", object_name="muted")
        hold_header.addWidget(self.hold_value)
        inspector_layout.addLayout(hold_header)
        self.hold_bar = QProgressBar()
        self.hold_bar.setObjectName("holdBar")
        self.hold_bar.setRange(0, 100)
        self.hold_bar.setValue(0)
        self.hold_bar.setTextVisible(False)
        self.hold_bar.setAccessibleName("Hold progress")
        inspector_layout.addWidget(self.hold_bar)

        self.prediction_status = _label("", object_name="statusLine")
        self.prediction_status.setAccessibleName("Prediction")
        inspector_layout.addWidget(self.prediction_status)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        inspector_layout.addWidget(divider)

        inspector_layout.addWidget(_label("SYSTEM STATUS", object_name="sectionTitle"))
        self.camera_status = _label("", object_name="statusLine")
        self.alphabet_status = _label("", object_name="statusLine")
        self.custom_status = _label("", object_name="statusLine")
        for label, name in (
            (self.camera_status, "Camera status"),
            (self.alphabet_status, "Alphabet model status"),
            (self.custom_status, "Custom gesture model status"),
        ):
            label.setAccessibleName(name)
            inspector_layout.addWidget(label)
        inspector_layout.addStretch()
        stage.addWidget(inspector)
        layout.addLayout(stage, 1)

        # Bottom translation area
        bottom = QWidget()
        bottom.setObjectName("bottomBar")
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(10, 8, 10, 8)
        bottom_layout.setSpacing(6)

        text_controls = QHBoxLayout()
        text_controls.setSpacing(6)
        text_controls.addWidget(_label("TRANSLATION", object_name="sectionTitle"))
        text_controls.addStretch()
        self.space_button = QPushButton("Space")
        self.space_button.setToolTip("Insert space (Space)")
        self.backspace_button = QPushButton("Backspace")
        self.backspace_button.setToolTip("Delete last character (Backspace)")
        self.clear_word_button = QPushButton("Clear Word")
        self.clear_word_button.setToolTip("Delete previous word (Ctrl+Backspace)")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear all text (C)")
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy translation to clipboard (Ctrl+C)")
        for button in (
            self.space_button, self.backspace_button, self.clear_word_button,
            self.clear_button, self.copy_button,
        ):
            text_controls.addWidget(button)
        bottom_layout.addLayout(text_controls)

        self.output = QPlainTextEdit()
        self.output.setAccessibleName("Translation text")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Committed signs appear here.")
        output_font = QFont(self.output.font())
        output_font.setPointSize(max(13, output_font.pointSize() + 3))
        self.output.setFont(output_font)
        self.output.setMinimumHeight(72)
        self.output.setMaximumHeight(112)
        bottom_layout.addWidget(self.output)
        layout.addWidget(bottom)

        self._apply_theme(self._theme)
        self.output.setFocus()

        self.refresh_button.clicked.connect(self._refresh)
        self.guide_button.clicked.connect(self._open_guide)
        self.settings_button.clicked.connect(self._open_settings)
        self.camera_choice.currentIndexChanged.connect(self._select_list_item)
        self.camera_index.valueChanged.connect(self._select_index)
        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.stop_button.clicked.connect(self._stop)
        self.space_button.clicked.connect(lambda: self._apply("SPACE"))
        self.backspace_button.clicked.connect(lambda: self._apply("BACKSPACE"))
        self.clear_word_button.clicked.connect(lambda: self._apply("CLEAR_WORD"))
        self.clear_button.clicked.connect(self._clear)
        self.copy_button.clicked.connect(self._copy)
        self._copy_reset = QTimer(self)
        self._copy_reset.setSingleShot(True)
        self._copy_reset.setInterval(1200)
        self._copy_reset.timeout.connect(lambda: self.copy_button.setText("Copy"))

        self._refresh()
        self._reset_status()
        self._update_controls()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        QApplication.instance().installEventFilter(self)

    def _apply_theme(self, theme: str):
        self._theme = theme if theme in _THEMES else "dark"
        if self._theme == "dark":
            self.setPalette(_dark_palette())
            self.setStyleSheet(_DARK_QSS)
        elif self._theme == "light":
            self.setPalette(_light_palette())
            self.setStyleSheet(_LIGHT_QSS)
        else:
            self.setPalette(QApplication.style().standardPalette())
            self.setStyleSheet("")
        self.preview.apply_theme(self._theme)

    def _refresh(self):
        with QSignalBlocker(self.camera_choice):
            self.camera_choice.clear()
            for index, name in camera_choices():
                self.camera_choice.addItem(f"{index}: {name}", index)
            self.camera_choice.setCurrentIndex(self.camera_choice.findData(self.camera.index))

    def _select_list_item(self, row):
        if row >= 0:
            self.camera_index.setValue(self.camera_choice.itemData(row))

    def _select_index(self, index):
        with QSignalBlocker(self.camera_choice):
            self.camera_choice.setCurrentIndex(self.camera_choice.findData(index))
        if self.camera.index == index:
            return
        self.camera = replace(self.camera, index=index)
        if self._mode == "live" and self.worker.state != "failed":
            self.preview.reset_frame()
            self._start()

    def _open_guide(self):
        if self._guide_dialog is None:
            self._guide_dialog = SignGuideDialog(self)
        self._guide_dialog.show()
        self._guide_dialog.raise_()
        self._guide_dialog.activateWindow()

    def _open_settings(self):
        dialog = SettingsDialog(
            self.thresholds, self.timing, self._theme, self, mirror_input=self.camera.mirror_input,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            changed_recognition = (
                dialog.thresholds != self.thresholds
                or dialog.timing != self.timing
                or dialog.mirror_input != self.camera.mirror_input
            )
            self.thresholds = dialog.thresholds
            self.timing = dialog.timing
            self.camera = replace(self.camera, mirror_input=dialog.mirror_input)
            if dialog.theme != self._theme:
                self._apply_theme(dialog.theme)
            if changed_recognition and self._mode == "live" and self.worker.state != "failed":
                self._start()

    def _reset_live_readout(self, source="Idle"):
        self.sign_badge.setText("—")
        self.sign_source.setText(source)
        self.confidence_value.setText("—")
        self.confidence_bar.setValue(0)
        self.hold_value.setText("0%")
        self.hold_bar.setValue(0)

    def _reset_status(self):
        self.camera_status.setText("Camera: Stopped")
        self.alphabet_status.setText("Alphabet: Not loaded — press Start")
        self.custom_status.setText("Custom: Not loaded — press Start")
        self.prediction_status.setText("Prediction: —")
        self._reset_live_readout("Idle")

    def _start(self):
        if self._mode == "closing":
            return
        self._mode = "live"
        self.prediction_status.setText("Prediction: Waiting for camera")
        self._reset_live_readout("Starting…")
        self.worker.start(self.camera, self.thresholds, self.timing)
        self._update_status()
        self._update_controls()

    def _pause(self):
        self._mode = "paused"
        self.worker.stop()
        self.camera_status.setText("Camera: Paused — Resume to reopen")
        self._reset_live_readout("Paused")
        self._update_controls()

    def _stop(self):
        self._mode = "idle"
        self.worker.stop()
        self.preview.reset_frame()
        self._reset_status()
        self._update_controls()

    def _update_status(self):
        for label, prefix, message in ((self.camera_status, "Camera", self.worker.camera_status),
                                       (self.alphabet_status, "Alphabet", self.worker.alphabet_status),
                                       (self.custom_status, "Custom", self.worker.custom_status)):
            text = f"{prefix}: {message}"
            if label.text() != text:
                label.setText(text)

    def _update_controls(self):
        live = self._mode == "live"
        retry = live and self.worker.state == "failed"
        closing = self._mode == "closing"
        self.start_button.setText("Resume" if self._mode == "paused" else "Retry" if retry else "Start")
        self.start_button.setEnabled(not closing and (not live or retry))
        self.pause_button.setEnabled(live and not retry)
        self.stop_button.setEnabled(not closing and self._mode != "idle")

    def _show_prediction(self, event):
        state, source, label = event["state"], event["source"], event["label"]
        progress = int(event.get("progress", 0))
        decision = {
            "no-hands": "No hands", "uncertain": "Uncertain", "ambiguous": "Ambiguous", "usable": "Candidate",
        }.get(state, "—")
        details = [decision]
        for classifier in ("alphabet", "custom"):
            if score := event[classifier]:
                selected = " (selected)" if classifier == source and score[0] == label else ""
                details.append(f"{classifier.title()} {score[0]} {score[1]:.0%}{selected}")
        if event["reason"] and event["reason"] != decision:
            details.append(event["reason"])
        self.prediction_status.setText("Prediction: " + " · ".join(details))

        active_score = None
        if source and event.get(source):
            active_score = event[source]
        elif event.get("alphabet"):
            active_score = event["alphabet"]
        elif event.get("custom"):
            active_score = event["custom"]

        if state == "no-hands" or active_score is None:
            self.sign_badge.setText("—")
            self.sign_source.setText("No hands" if state == "no-hands" else decision)
            self.confidence_value.setText("—")
            self.confidence_bar.setValue(0)
        else:
            shown_label = label or active_score[0]
            conf_pct = max(0, min(100, round(active_score[1] * 100)))
            self.sign_badge.setText(shown_label)
            source_text = (
                "SignSpeak A–Z" if source == "alphabet"
                else "Custom Command" if source == "custom"
                else decision
            )
            self.sign_source.setText(source_text)
            self.confidence_value.setText(f"{conf_pct}%")
            self.confidence_bar.setValue(conf_pct)

        progress = max(0, min(100, progress))
        self.hold_bar.setValue(progress)
        self.hold_value.setText("Release" if event.get("waiting_release") else f"{progress}%")

    def _tick(self):
        events, frame = self.worker.tick()
        if self._mode == "closing":
            if not self.worker.active:
                self.close()
            return
        if self._mode != "live":
            return
        for event in events:
            if event["kind"] == "prediction":
                self._show_prediction(event)
            elif event["kind"] == "action" and event["timestamp"] > self._clear_at:
                self._apply(event["label"])
        if frame is not None:
            self.preview.set_frame(frame)
        self._update_status()
        self._update_controls()

    def _apply(self, action):
        updated = apply_action(self._text, action)
        if updated != self._text:
            self._text = updated
            self.output.setPlainText(updated)
            self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _clear(self):
        self._clear_at = monotonic()
        self._apply("CLEAR")

    def _copy(self):
        QApplication.clipboard().setText(self._text)
        if self._text:
            self.copy_button.setText("Copied")
            self._copy_reset.start()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and isinstance(watched, QWidget) and watched.window() is self:
            focus = QApplication.focusWidget()
            if any(widget is not None and (widget is field or field.isAncestorOf(widget))
                   for field in (self.camera_index, self.camera_choice) for widget in (watched, focus)):
                return False
            key, modifier = event.key(), event.modifiers()
            if modifier == Qt.KeyboardModifier.NoModifier:
                callback = {
                    Qt.Key.Key_C: self._clear, Qt.Key.Key_Escape: self.close, Qt.Key.Key_Q: self.close,
                    Qt.Key.Key_Space: lambda: self._apply("SPACE"),
                    Qt.Key.Key_Backspace: lambda: self._apply("BACKSPACE"),
                }.get(key)
            elif modifier == Qt.KeyboardModifier.ControlModifier:
                callback = {Qt.Key.Key_C: self._copy, Qt.Key.Key_Backspace: lambda: self._apply("CLEAR_WORD")}.get(key)
            else:
                callback = None
            if callback is not None:
                callback()
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if self._smoke is not None and self._smoke.active:
            self._smoke.stop()
            event.ignore()
            return
        if self.worker.active:
            if self._mode != "closing":
                self._mode = "closing"
                self.camera_status.setText("Camera: Closing…")
                self.worker.stop()
                self._update_controls()
            event.ignore()
        else:
            self._timer.stop()
            self._copy_reset.stop()
            QApplication.instance().removeEventFilter(self)
            event.accept()


def _parse_args(argv):
    camera_defaults, thresholds_defaults, timing_defaults = CameraSettings(0), Thresholds(), Timing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="Camera index (default: first listed)")
    parser.add_argument("--width", type=int, default=camera_defaults.width)
    parser.add_argument("--height", type=int, default=camera_defaults.height)
    parser.add_argument("--fps", type=float, default=camera_defaults.fps)
    parser.add_argument("--mirror-input", action="store_true", help="Flip recognition input horizontally (left-handed or mirrored camera)")
    parser.add_argument("--theme", choices=_THEMES, default="dark")
    parser.add_argument("--alphabet-threshold", type=float, default=thresholds_defaults.alphabet)
    parser.add_argument("--custom-threshold", type=float, default=thresholds_defaults.custom)
    parser.add_argument("--destructive-threshold", type=float, default=thresholds_defaults.destructive)
    parser.add_argument("--hold-seconds", type=float, default=timing_defaults.alphabet_hold)
    parser.add_argument("--custom-hold-seconds", type=float, default=timing_defaults.custom_hold)
    parser.add_argument("--destructive-hold-seconds", type=float, default=timing_defaults.destructive_hold)
    parser.add_argument("--cooldown-seconds", type=float, default=timing_defaults.cooldown)
    parser.add_argument("--destructive-cooldown-seconds", type=float, default=timing_defaults.destructive_cooldown)
    parser.add_argument("--smoke-test", action="store_true", help="Verify native window and model in a child without a camera")
    parser.add_argument("--smoke-report", type=Path, help="Write smoke-test JSON to this path")
    args = parser.parse_args(argv)
    if args.smoke_report is not None and not args.smoke_test:
        parser.error("--smoke-report requires --smoke-test")
    try:
        camera = CameraSettings(default_camera() if args.camera is None else args.camera,
                                args.width, args.height, args.fps, mirror_input=args.mirror_input)
        thresholds = Thresholds(args.alphabet_threshold, args.custom_threshold, args.destructive_threshold)
        timing = Timing(args.hold_seconds, args.custom_hold_seconds, args.destructive_hold_seconds,
                        args.cooldown_seconds, args.destructive_cooldown_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    return args, camera, thresholds, timing


def main(argv=None):
    args, camera, thresholds, timing = _parse_args(argv)
    app = QApplication([sys.argv[0]])
    configure_qt_identity()
    window = ModernSignLanguageApp(camera, thresholds=thresholds, timing=timing, theme=args.theme)
    old_sigint = signal.signal(signal.SIGINT, lambda *_: window.close())
    if args.smoke_test:
        window.setEnabled(False)
    window.show()
    try:
        if args.smoke_test:
            from utils.smoke import start_smoke

            window._smoke = smoke = start_smoke(window)
        status = app.exec()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
    if not args.smoke_test:
        return status
    result = smoke.result
    text = json.dumps(result, indent=2) + "\n"
    try:
        if args.smoke_report is not None:
            args.smoke_report.write_text(text, encoding="utf-8")
        if sys.stdout is not None:
            sys.stdout.write(text)
    except OSError as exc:
        if sys.stderr is not None:
            print(f"Cannot write smoke report: {exc}", file=sys.stderr)
        return 1
    return 0 if result.get("ok") and status == 0 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
