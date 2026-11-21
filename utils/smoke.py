"""Run the native window and verified inference smoke test without a camera."""

import json
import multiprocessing as mp
import sys
import traceback
from time import monotonic

from PyQt6.QtCore import QCoreApplication, QObject, QTimer

_SMOKE_TIMEOUT = 30.0
_KILL_GRACE = 0.75
_MAX_REPORT = 2048


def _failure(message: str) -> dict:
    return {"ok": False, "camera_opened": False, "error": message[:250]}


def _run_model() -> dict:
    from math import dist, isclose

    import numpy as np

    from .detector import HandDetector
    from .observations import Detection, Hand
    from .signspeak import ALPHABET, MODEL_VERSION, SignSpeakAlphabetRecognizer

    recognizer = SignSpeakAlphabetRecognizer()
    with HandDetector(static_image_mode=True):
        # Synthetic raw landmarks, not an example of any sign. The right slot
        # is populated; the left slot is empty, and landmark 12 is not farthest.
        index = np.arange(21, dtype=np.float64)
        points = np.column_stack((
            0.35 + 0.011 * index + 0.003 * (index % 4),
            0.72 - 0.012 * index + 0.008 * (index % 3),
            -0.002 * index + 0.0005 * (index % 5),
        ))
        actual = recognizer.predict(Detection((Hand(points, "Right", 0.97),)))

        # Independently construct the pinned reference slots and evaluate them
        # separately through the same verified StandardScaler → SVC pipeline.
        hand = ((points - points[0]) / dist(points[0], points[12])).ravel()
        empty = np.zeros(63)
        pipeline = recognizer.pipeline
        detected = pipeline.predict_proba([np.concatenate((empty, hand))])[0]
        swapped = pipeline.predict_proba([np.concatenate((hand, empty))])[0]
        use_swapped = swapped.max() > detected.max()
        row, slot = (swapped, "swapped") if use_swapped else (detected, "as-detected")
        top = np.argsort(row)[-1]
        if (
            actual is None
            or actual.label not in ALPHABET
            or actual.label != pipeline.classes_[top]
            or actual.slot != slot
            or not 0 <= actual.confidence <= 1
            or not isclose(actual.confidence, row[top], rel_tol=0, abs_tol=1e-12)
        ):
            raise RuntimeError("SignSpeak output did not match direct verified pipeline inference.")

    return {
        "ok": True,
        "label": actual.label,
        "confidence": actual.confidence,
        "slot": actual.slot,
        "model": MODEL_VERSION,
        "fixture": "synthetic-landmarks",
        "mediapipe_initialized": True,
        "camera_opened": False,
    }


def _smoke_child(sender) -> None:
    with sender:
        try:
            result = _run_model()
        except Exception as exc:
            if sys.stderr is not None:
                traceback.print_exc()
            result = _failure(f"{type(exc).__name__}: {exc}")
        # A single bounded packet fits in the pipe before the parent reads it.
        sender.send_bytes(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))


class SmokeRun(QObject):
    """Supervise one spawned child, then close the visible smoke window.

    The window defers close while ``active`` and calls ``stop`` to cancel.
    """

    def __init__(self, window):
        super().__init__(window)
        self.result: dict = {}
        self._window = window
        self._process = None
        self._receiver = None
        self._evidence = None
        self._kill_at = None
        self._deadline = monotonic() + _SMOKE_TIMEOUT

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._start_child()
        self._timer.start()

    def _start_child(self) -> None:
        context = mp.get_context("spawn")
        try:
            self._receiver, sender = context.Pipe(duplex=False)
            with sender:
                process = context.Process(target=_smoke_child, args=(sender,), daemon=True)
                process.start()
        except (OSError, RuntimeError) as exc:
            if self._receiver is not None:
                self._receiver.close()
                self._receiver = None
            self._evidence = _failure(f"Cannot spawn the SignSpeak smoke worker: {exc}")
        else:
            self._process = process

    @property
    def active(self) -> bool:
        return self._timer.isActive()

    def stop(self) -> None:
        """Request cancellation without blocking the window's close handler."""
        if self.active:
            self._cancel("The smoke window was closed before the test completed.")

    def _cancel(self, message: str) -> None:
        if self._evidence is not None:
            return
        self._evidence = _failure(message)
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._kill_at = monotonic() + _KILL_GRACE

    def _read_result(self, exitcode: int) -> dict:
        if exitcode != 0:
            return _failure(f"The SignSpeak smoke worker exited with code {exitcode}.")
        try:
            result = json.loads(self._receiver.recv_bytes(_MAX_REPORT))
        except (EOFError, OSError, ValueError) as exc:
            return _failure(f"No valid result from the SignSpeak smoke worker ({type(exc).__name__}).")
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            return _failure("The SignSpeak smoke worker returned an invalid report.")
        return result

    def _tick(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                now = monotonic()
                if self._evidence is None and now >= self._deadline:
                    self._cancel("Timed out waiting for the SignSpeak smoke worker.")
                elif self._kill_at is not None and now >= self._kill_at:
                    process.kill()
                    self._kill_at = None
                return
            process.join(0)
            # With the only writer gone, this read is immediately ready or EOF.
            if self._evidence is None:
                self._evidence = self._read_result(process.exitcode)
            self._receiver.close()
            process.close()
            self._process = self._receiver = None

        self._timer.stop()
        if self._window.close():
            self.result = self._evidence
        else:
            self.result = _failure("Cannot close the smoke window after reaping the worker.")
            QCoreApplication.exit(1)


def start_smoke(window) -> SmokeRun:
    """Start a smoke run; keep the controller alive until the window closes."""
    return SmokeRun(window)
