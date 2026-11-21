"""Spawn-safe camera target and bounded single-producer/single-consumer IPC."""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from signal import SIG_IGN, SIGINT, signal
from sys import float_info
from time import monotonic

from .camera import STREAM_TIMEOUT, StopFlag, frames, open_camera
from .config import MAX_HEIGHT, MAX_WIDTH

_EVENT_COUNT = 32
_EVENT_BYTES = 1024
_STATUS_CHARS = 240  # At most four UTF-8 bytes per printable character, plus framing.
_STATUS_KINDS = frozenset(("opened", "open-error", "alphabet", "custom", "fatal"))


@dataclass(frozen=True, slots=True)
class Preview:
    data: bytes
    width: int
    height: int
    sequence: int


def _text(value, limit):
    return type(value) is str and len(value) <= limit


def _number(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


def _score(value):
    return value is None or (
        type(value) is list and len(value) == 2
        and _text(value[0], 32) and _number(value[1], 0, 1)
    )


def _encode_event(event):
    if not isinstance(event, dict) or not _text(kind := event.get("kind"), 16):
        raise ValueError("Worker event has no valid kind")
    if kind in _STATUS_KINDS:
        valid = event.keys() == {"kind", "message"} and _text(event["message"], _EVENT_BYTES)
    elif kind == "ready":
        valid = event.keys() == {"kind"}
    elif kind == "action":
        valid = (event.keys() == {"kind", "label", "timestamp"} and _text(event["label"], 32)
                 and _number(event["timestamp"], -float_info.max, float_info.max))
    elif kind == "prediction":
        valid = (
            event.keys() == {
                "kind", "state", "reason", "source", "label", "alphabet", "custom",
                "progress", "waiting_release", "hands",
            }
            and event["state"] in ("no-hands", "uncertain", "usable", "ambiguous")
            and _text(event["reason"], _STATUS_CHARS)
            and event["source"] in ("", "alphabet", "custom")
            and _text(event["label"], 32)
            and _score(event["alphabet"]) and _score(event["custom"])
            and type(event["progress"]) is int and 0 <= event["progress"] <= 100
            and type(event["waiting_release"]) is bool
            and type(event["hands"]) is int and 0 <= event["hands"] <= 2
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"Invalid worker event: {kind}")
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > _EVENT_BYTES:
        raise ValueError("Worker event exceeds 1 KiB")
    return payload


class Channels:
    """Latest RGB image and a separate ordered event ring; one child writes both."""

    def __init__(self, ctx, width, height):
        if type(width) is not int or type(height) is not int or not (1 <= width <= MAX_WIDTH and 1 <= height <= MAX_HEIGHT):
            raise ValueError(f"Preview bounds must be integers within {MAX_WIDTH}×{MAX_HEIGHT}")
        self._width, self._height = width, height
        self.stop = StopFlag(ctx)
        self._saturated = ctx.RawValue("B", 0)
        self._pixels = ctx.RawArray("B", width * height * 3)
        self._dimensions = ctx.RawArray("Q", 3)  # Width, height, sequence; sequence 0 means no image.
        self._preview_lock = ctx.Lock()
        self._events = ctx.RawArray("B", _EVENT_COUNT * _EVENT_BYTES)
        self._sizes = ctx.RawArray("H", _EVENT_COUNT)
        self._free = ctx.Semaphore(_EVENT_COUNT)
        self._used = ctx.Semaphore(0)
        # Each cursor has exactly one owner, so neither needs to be shared.
        self._write_slot = self._read_slot = 0

    @property
    def saturated(self) -> bool:
        return bool(self._saturated.value)

    def publish(self, event: dict) -> bool:
        """Write a bounded event, or mark the session failed when the ring is full."""
        payload = _encode_event(event)
        if self.saturated:
            return False
        if not self._free.acquire(False):
            self._saturated.value = 1
            self.stop.set()
            return False
        slot = self._write_slot
        start = slot * _EVENT_BYTES
        memoryview(self._events).cast("B")[start:start + len(payload)] = payload
        self._sizes[slot] = len(payload)
        self._write_slot = (slot + 1) % _EVENT_COUNT
        self._used.release()
        return True

    def drain(self) -> list[dict]:
        """Copy at most 32 complete events; never wait on a pipe or feeder thread."""
        data = memoryview(self._events).cast("B")
        events = []
        for _ in range(_EVENT_COUNT):
            if not self._used.acquire(False):
                break
            slot = self._read_slot
            start = slot * _EVENT_BYTES
            payload = bytes(data[start:start + self._sizes[slot]])
            self._read_slot = (slot + 1) % _EVENT_COUNT
            self._free.release()
            events.append(json.loads(payload))
        return events

    def write_preview(self, rgb):
        view = memoryview(rgb)
        if view.ndim != 3 or view.format != "B" or not view.c_contiguous:
            raise ValueError("Preview must be a contiguous uint8 RGB image")
        height, width, channels = view.shape
        if channels != 3 or not (1 <= width <= self._width and 1 <= height <= self._height):
            raise ValueError("Preview is not RGB or exceeds the shared image bounds")
        if not self._preview_lock.acquire(False):
            return
        try:
            memoryview(self._pixels).cast("B")[:view.nbytes] = view.cast("B")
            self._dimensions[0] = width
            self._dimensions[1] = height
            self._dimensions[2] += 1
        finally:
            self._preview_lock.release()

    def read_preview(self, last_sequence: int) -> Preview | None:
        if not self._preview_lock.acquire(False):
            return None
        try:
            width, height, sequence = self._dimensions
            if sequence == 0 or sequence <= last_sequence:
                return None
            return Preview(bytes(memoryview(self._pixels).cast("B")[:width * height * 3]), width, height, sequence)
        finally:
            self._preview_lock.release()

    def close(self):
        """Drop the parent's buffers/semaphores after reaping; never touch their locks."""
        self._pixels = self._dimensions = self._preview_lock = None
        self._events = self._sizes = self._free = self._used = None
        self.stop = self._saturated = None


def _status(channels, kind, message, guidance=""):
    message = " ".join(message.split())[:_STATUS_CHARS]
    message = "".join(char if char.isprintable() else "?" for char in message)
    if guidance and guidance not in message:
        message = f"{message[:_STATUS_CHARS - len(guidance) - 1]} {guidance}"
    return channels.publish({"kind": kind, "message": message})


def _prepare_rgb(frame, settings):
    import cv2

    height, width = frame.shape[:2]
    if width > settings.width or height > settings.height:
        if settings.width * height <= settings.height * width:
            size = settings.width, max(1, height * settings.width // width)
        else:
            size = max(1, width * settings.height // height), settings.height
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.flip(rgb, 1) if settings.mirror_input else rgb


def _prediction(
    decision, destructive_armed, *, progress: int = 0, waiting_release: bool = False, hands: int = 0,
):
    candidate = decision.candidate
    reason = decision.reason
    if candidate is not None and candidate.destructive and not destructive_armed:
        reason = "Release your hands to arm destructive commands"
    elif waiting_release and candidate is not None:
        reason = "Lower hands briefly to repeat this sign"
    return {
        "kind": "prediction", "state": decision.kind, "reason": reason,
        "source": candidate.source if candidate else "", "label": candidate.label if candidate else "",
        "alphabet": [decision.alphabet.label, decision.alphabet.confidence] if decision.alphabet else None,
        "custom": [decision.custom.label, decision.custom.confidence] if decision.custom else None,
        "progress": max(0, min(100, int(progress))),
        "waiting_release": bool(waiting_release),
        "hands": max(0, min(2, int(hands))),
    }


def _display_key(event):
    alphabet, custom = event["alphabet"], event["custom"]
    return (
        event["state"], event["reason"], event["source"], event["label"],
        (alphabet[0], round(alphabet[1] * 100)) if alphabet else None,
        (custom[0], round(custom[1] * 100)) if custom else None,
        round(event["progress"] / 5) * 5,
        event["waiting_release"],
        event["hands"],
    )


class _SmoothedAlphabet:
    """Short 3-frame probability smoother that removes single-frame jitter without hiding transitions."""

    def __init__(self, recognizer, window: int = 3):
        from collections import deque

        self._recognizer = recognizer
        self._window = deque(maxlen=window)
        self._hand_count = 0

    def reset(self):
        self._window.clear()
        self._hand_count = 0

    def predict(self, detection):
        import numpy as np

        from .observations import Prediction

        count = len(detection.hands)
        if not detection.valid or count == 0:
            self.reset()
            return None
        if count != self._hand_count:
            self._window.clear()
            self._hand_count = count
        result = self._recognizer.predict_proba(detection)
        if result is None:
            self.reset()
            return None
        probs, slot = result
        self._window.append(probs)
        averaged = np.mean(self._window, axis=0)
        runner_up, top = np.argsort(averaged)[-2:]
        return Prediction(
            label=str(self._recognizer.pipeline.classes_[top]),
            confidence=float(averaged[top]),
            margin=float(averaged[top] - averaged[runner_up]),
            slot=slot,
        )


def _hud_label(event):
    hands = event["hands"]
    hand_str = "No hands" if hands == 0 else "1 hand" if hands == 1 else "2 hands"
    label = event["label"]
    source = event["source"]
    score = event[source] if source and event.get(source) else event.get("alphabet") or event.get("custom")
    if hands == 0 or score is None:
        return hand_str
    shown = label or score[0]
    pct = round(score[1] * 100)
    suffix = " (release)" if event["waiting_release"] else ""
    return f"{hand_str} | {shown} {pct}%{suffix}"


def _stream(settings, capture, first, detector, arbiter, stabilizer, channels, smoother=None):
    from .detector import render_preview_rgb
    from .recognition import Decision

    failed_read = Decision("uncertain", "Camera read failed; waiting for the next frame")
    last_display = None
    period = 1 / settings.fps
    for frame in frames(capture, first, channels.stop):
        observed = monotonic()
        rgb = None
        hand_count = 0
        if frame is None:
            decision = failed_read
            if smoother is not None:
                smoother.reset()
        else:
            rgb = _prepare_rgb(frame, settings)
            height, width = rgb.shape[:2]
            detection = detector.detect_rgb(rgb)
            hand_count = len(detection.hands)
            if not detection.valid or hand_count == 0:
                if smoother is not None:
                    smoother.reset()
            decision = arbiter.observe(detection, width / height)
        action = stabilizer.observe(decision, observed)
        if channels.stop.is_set():
            return
        progress = 100 if action is not None else round(stabilizer.progress * 100)
        event = _prediction(
            decision, stabilizer.destructive_armed,
            progress=progress, waiting_release=stabilizer.waiting_release, hands=hand_count,
        )
        display = _display_key(event)
        if display != last_display:
            if not channels.publish(event):
                return
            last_display = display
        if action is not None and not channels.publish({"kind": "action", "label": action.label, "timestamp": observed}):
            return
        if rgb is not None:
            hud = (_hud_label(event), progress)
            channels.write_preview(render_preview_rgb(rgb, detection, mirror=not settings.mirror_input, hud=hud))
            due = observed + period
            while (delay := due - monotonic()) > 0:
                if channels.stop.wait(min(delay, STREAM_TIMEOUT)):
                    return


def camera_main(settings, backend, thresholds, timing, channels):
    """Own exactly one backend attempt and one recognition session in a spawned child."""
    signal(SIGINT, SIG_IGN)
    opened = False
    try:
        with ExitStack() as cleanup:
            if channels.stop.is_set():
                return
            capture, first, info = open_camera(settings, backend)
            cleanup.callback(capture.release)
            opened = True
            if channels.stop.is_set() or not _status(channels, "opened", info.describe()):
                return

            from .custom_gestures import (
                CustomGestureRecognizer,
                CustomModelError,
                GeometricCustomGestureRecognizer,
            )
            from .detector import HandDetector
            from .model_assets import INSTALL_HINT, AssetError
            from .recognition import Arbiter
            from .signspeak import SignSpeakAlphabetRecognizer
            from .temporal import Stabilizer

            try:
                alphabet = SignSpeakAlphabetRecognizer()
            except (AssetError, ImportError, OSError) as exc:
                alphabet = None
                alphabet_message = f"SignSpeak unavailable: {exc}"
            else:
                alphabet_message = "SignSpeak A–Z ready"
            if channels.stop.is_set() or not _status(
                channels, "alphabet", alphabet_message, INSTALL_HINT if alphabet is None else "",
            ):
                return

            try:
                custom = CustomGestureRecognizer()
            except (AssetError, CustomModelError, OSError):
                custom = GeometricCustomGestureRecognizer()
                custom_message = (
                    "Built-in commands ready (0, 1, SPACE, BACKSPACE, CLEAR_WORD)"
                    if alphabet is not None
                    else "Built-in commands ready; destructive commands require SignSpeak."
                )
            else:
                custom_message = "Custom gestures ready" if alphabet is not None else (
                    "Custom gestures ready; destructive webcam commands require SignSpeak. Manual editing still works."
                )
            if channels.stop.is_set() or not _status(channels, "custom", custom_message):
                return

            detector = cleanup.enter_context(
                HandDetector(
                    min_detection_confidence=0.30,
                    min_tracking_confidence=0.30,
                    resolve_duplicate_handedness=True,
                )
            )
            smoother = _SmoothedAlphabet(alphabet) if alphabet is not None else None
            if not channels.stop.is_set() and channels.publish({"kind": "ready"}):
                _stream(
                    settings, capture, first, detector,
                    Arbiter(smoother, custom, thresholds),
                    Stabilizer(timing, arm_destructive_on_start=True),
                    channels, smoother,
                )
    except Exception as exc:
        if not channels.stop.is_set():
            if opened:
                _status(channels, "fatal", f"Camera processing stopped: {exc}", "Check the camera and choose Start/Retry.")
            else:
                _status(channels, "open-error", str(exc), "Check camera selection, connection and permissions.")
