"""Nonblocking parent-side supervision for one camera worker at a time."""

import multiprocessing as mp
from math import inf
from string import ascii_uppercase
from time import monotonic
from typing import TYPE_CHECKING

from utils.camera import START_TIMEOUT, STREAM_TIMEOUT, CameraSettings, backend_names
from utils.video_worker import Channels, Preview, camera_main

if TYPE_CHECKING:
    from utils.recognition import Thresholds
    from utils.temporal import Timing

_INIT_TIMEOUT = 15.0
_STOP_GRACE = 0.75
_RETRY = "Select Start / Retry to try again."
_SATURATION = "Camera event queue filled. Some pending gesture edits could not be delivered."
_ACTIONS = frozenset(ascii_uppercase + "01") | {"SPACE", "BACKSPACE", "CLEAR_WORD"}


def _valid_action(event: dict) -> bool:
    label, timestamp = event.get("label"), event.get("timestamp")
    return (isinstance(label, str) and label in _ACTIONS
            and type(timestamp) in (int, float) and 0 <= timestamp < inf)


class VideoSession:
    """Call from a single GUI thread; only ``tick`` polls and reaps workers.

    A stopping session is still active until reaped, but its events and preview
    are unavailable immediately. The caller owns any preview it wants to retain
    on Pause and decides whether to clear that preview on Stop or camera switch.
    """

    def __init__(self, target=camera_main, backends=None):
        self.state = "idle"
        self.camera_status = "Camera stopped."
        self.alphabet_status = self.custom_status = "Not loaded."
        self._target = target
        self._backends = tuple(backend_names() if backends is None else backends)
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._channels = None
        self._request = None
        self._open_failures = []
        self._next_backend = 0
        self._backend = ""
        self._deadline = 0.0
        self._last_sequence = 0
        self._stop_step = 0
        self._after_reap = "idle"

    @property
    def active(self) -> bool:
        return self._process is not None

    def start(self, settings: CameraSettings, thresholds: "Thresholds", timing: "Timing") -> None:
        """Open selected camera, replacing any active or queued session."""
        self._request = (settings, thresholds, timing)
        self._next_backend = 0
        self._open_failures.clear()
        self.alphabet_status = self.custom_status = "Waiting for camera."
        now = monotonic()
        if self.active:
            self._retire("idle", "Switching camera…", now, restart=True)
        else:
            self._spawn_next(now)

    def stop(self) -> None:
        """Cancel any replacement and discard data from the current session."""
        self.alphabet_status = self.custom_status = "Not loaded."
        status = "Stopping camera…" if self.active else "Camera stopped."
        self._retire("idle", status, monotonic())

    def tick(self, now: float | None = None) -> tuple[list[dict], Preview | None]:
        if now is None:
            now = monotonic()
        if not self.active:
            return [], None
        events = []
        if self.state != "stopping":
            events, preview = self._poll(now)
            if self.state != "stopping":
                return events, preview
        self._finish_stop(now)
        return events, None

    def _spawn_next(self, now: float) -> None:
        if self._next_backend == len(self._backends):
            self._fail("No camera backend is available.", now)
            return
        settings, thresholds, timing = self._request
        backend = self._backends[self._next_backend]
        self._next_backend += 1
        channels = process = None
        try:
            channels = Channels(self._ctx, settings.width, settings.height)
            process = self._ctx.Process(target=self._target, args=(settings, backend, thresholds, timing, channels))
            process.start()
        except Exception as exc:
            if channels is not None:
                channels.close()
            if process is not None:
                process.close()
            self._fail(f"Could not start camera worker: {exc}.", now)
            return
        self._channels, self._process = channels, process
        self._backend = backend
        self._deadline = now + START_TIMEOUT
        self._last_sequence = 0
        self.state = "opening"
        self.camera_status = f"Opening camera {settings.index} ({backend})…"

    def _poll(self, now: float) -> tuple[list[dict], Preview | None]:
        channels = self._channels
        already_saturated = channels.saturated
        events = []
        for event in channels.drain():
            kind = event.get("kind")
            if kind == "opened" and self.state == "opening":
                self.state = "initializing"
                self.camera_status = event["message"]
                self.alphabet_status = self.custom_status = "Loading…"
                self._deadline = now + _INIT_TIMEOUT
            elif kind == "open-error" and self.state == "opening":
                self._open_failed(event["message"], now)
                return [], None
            elif kind == "alphabet":
                self.alphabet_status = event["message"]
            elif kind == "custom":
                self.custom_status = event["message"]
            elif kind == "ready" and self.state == "initializing":
                self.state = "running"
                self._deadline = now + STREAM_TIMEOUT
            elif kind == "fatal":
                self._fail(event["message"], now)
                return [], None
            elif self.state == "running" and (kind == "prediction" or (kind == "action" and _valid_action(event))):
                events.append(event)

        # Read terminal messages even if the child exited just before this tick,
        # but do not use previews or apply edits from a dead worker.
        if channels.saturated:
            # Saturation during this drain may leave accepted edits for the next tick.
            if already_saturated:
                self._fail(_SATURATION, now)
            return [event for event in events if event["kind"] == "action"], None
        if not self._process.is_alive():
            self._fail(f"Camera worker exited unexpectedly (exit code {self._process.exitcode}).", now)
            return [], None
        if self.state == "opening":
            if now >= self._deadline:
                self._open_failed(
                    "Timed out opening the camera or reading its first frame. "
                    "Check camera access and reconnect or close other camera apps.",
                    now,
                )
            return [], None
        if self.state == "initializing" and now >= self._deadline:
            self._fail(
                "Timed out loading recognition or starting hand tracking. "
                "Check the runtime dependencies and run python -m utils.model_assets verify.",
                now,
            )
            return [], None
        if self.state != "running":
            return [], None
        preview = channels.read_preview(self._last_sequence)
        if preview is not None:
            self._last_sequence = preview.sequence
            self._deadline = now + STREAM_TIMEOUT
        elif now >= self._deadline:
            self._fail("Camera stream stalled: no new preview frames. Check the camera connection.", now)
            return [], None
        return events, preview

    def _open_failed(self, message: str, now: float) -> None:
        self._open_failures.append(f"{self._backend}: {message}")
        if self._next_backend < len(self._backends):
            self._retire("failed", f"{message} Trying {self._backends[self._next_backend]} next…", now, restart=True)
        else:
            self._fail("; ".join(self._open_failures), now)

    def _fail(self, message: str, now: float) -> None:
        if self.alphabet_status in ("Waiting for camera.", "Loading…"):
            self.alphabet_status = "Not loaded."
        if self.custom_status in ("Waiting for camera.", "Loading…"):
            self.custom_status = "Not loaded."
        self._retire("failed", f"{message} {_RETRY}", now)

    def _retire(self, state: str, status: str, now: float, *, restart: bool = False) -> None:
        self.camera_status = status
        if not restart:
            self._request = None
        if not self.active:
            self.state = state
            return
        self._after_reap = state
        if self.state != "stopping":
            self.state = "stopping"
            self._deadline = now + _STOP_GRACE
            self._stop_step = 0
            if self._process.is_alive():
                self._channels.stop.set()

    def _finish_stop(self, now: float) -> None:
        process = self._process
        if not process.is_alive():
            process.join(0)
            self._channels.close()
            process.close()
            self._process = self._channels = None
            if self._request is not None:
                self._spawn_next(now)
            else:
                self.state = self._after_reap
                if self.state == "idle":
                    self.camera_status = "Camera stopped."
        elif now >= self._deadline:
            if self._stop_step == 0:
                process.terminate()
                self._stop_step = 1
                self._deadline = now + _STOP_GRACE
            elif self._stop_step == 1:
                process.kill()
                self._stop_step = 2
