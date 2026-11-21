"""Discover cameras and make explicit, single-backend capture attempts."""

import re
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from stat import S_ISDIR
from time import monotonic, sleep

from . import config

START_TIMEOUT = 8.0
STREAM_TIMEOUT = 3.0
INIT_TIMEOUT = 15.0

_DEV_DIR = Path("/dev")
_VIDEO_NODE = re.compile(r"video(0|[1-9][0-9]*)")
_BACKENDS = ("V4L2", "MSMF", "DSHOW", "ANY")
_FAILED_READ_LIMIT = 5
_FAILED_READ_DELAY = 0.05
_OPEN_HINT = "Check camera selection and permissions, close other camera apps, or reconnect and retry."


class CameraError(RuntimeError):
    pass


class StopFlag:
    """Single-byte one-way stop signal; a suspended child cannot block set()."""

    def __init__(self, ctx):
        self._flag = ctx.RawValue("B", 0)

    def set(self):
        self._flag.value = 1

    def is_set(self):
        return bool(self._flag.value)

    def wait(self, timeout):
        deadline = monotonic() + timeout
        while not self.is_set():
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            sleep(min(remaining, 0.05))
        return True


@dataclass(frozen=True, slots=True)
class CameraSettings:
    index: int
    width: int = config.DEFAULT_WIDTH
    height: int = config.DEFAULT_HEIGHT
    fps: float = config.DEFAULT_FPS
    mirror_input: bool = False

    def __post_init__(self):
        if type(self.index) is not int or not 0 <= self.index <= config.MAX_INDEX:
            raise ValueError(f"Camera index must be an integer from 0 to {config.MAX_INDEX}.")
        for name, value, maximum in (("Width", self.width, config.MAX_WIDTH),
                                     ("Height", self.height, config.MAX_HEIGHT)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}.")
        if type(self.fps) not in (int, float) or not 0 < self.fps <= config.MAX_FPS:
            raise ValueError(f"FPS must be greater than 0 and at most {config.MAX_FPS}.")
        if type(self.mirror_input) is not bool:
            raise ValueError("mirror_input must be a boolean.")


@dataclass(frozen=True, slots=True)
class CameraInfo:
    backend: str
    width: int
    height: int
    fps: float | None

    def describe(self) -> str:
        resolution = f"{self.backend}: {self.width}×{self.height}"
        return f"{resolution} at {self.fps:g} FPS" if self.fps is not None else f"{resolution} (FPS not reported)"


def _linux_camera_name(node_name: str) -> str:
    if _DEV_DIR != Path("/dev"):
        return node_name
    try:
        label = (Path("/sys/class/video4linux") / node_name / "name").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return node_name
    if not label:
        return node_name
    if ":" in label:
        head, tail = (part.strip() for part in label.split(":", 1))
        if head and tail and head.lower().startswith(tail.lower()):
            label = head
    return f"{label} ({node_name})" if label != node_name else node_name


def camera_choices() -> list[tuple[int, str]]:
    """List Linux device nodes or Windows candidate indices without opening cameras."""
    if sys.platform.startswith("linux"):
        choices = []
        for node in _DEV_DIR.glob("video*"):
            match = _VIDEO_NODE.fullmatch(node.name)
            if match is None:
                continue
            index = int(match[1])
            if index > config.MAX_INDEX:
                continue
            try:
                if not S_ISDIR(node.stat().st_mode):
                    choices.append((index, _linux_camera_name(node.name)))
            except FileNotFoundError:
                continue
        return sorted(choices)
    if sys.platform == "win32":
        return [(index, f"Camera {index} (candidate)") for index in range(8)]
    return []


def default_camera() -> int:
    choices = camera_choices()
    return choices[0][0] if choices else 0


def backend_names() -> tuple[str, ...]:
    if sys.platform.startswith("linux"):
        return "V4L2", "ANY"
    if sys.platform == "win32":
        return "MSMF", "DSHOW", "ANY"
    return ("ANY",)


def _read_image(capture):
    ok, frame = capture.read()
    return frame if ok and frame is not None and frame.size else None


def open_camera(settings: CameraSettings, backend: str):
    """Open and read once; transfer ownership of the capture only on success."""
    if backend not in _BACKENDS:
        raise CameraError(f"Unknown camera backend {backend!r}.")
    if backend == "ANY" and settings.index >= 100:
        # OpenCV interprets the hundreds as a backend ID under CAP_ANY and changes the index.
        raise CameraError(f"Cannot use CAP_ANY for camera {settings.index} without changing its index.")
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise CameraError(f"Cannot load OpenCV. Install the project requirements and system libraries: {exc}") from exc

    capture = None
    ready = False
    target = f"Camera {settings.index} ({backend})"
    try:
        capture = cv2.VideoCapture()
        if not capture.open(settings.index, getattr(cv2, f"CAP_{backend}")):
            raise CameraError(f"{target} could not open. {_OPEN_HINT}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        capture.set(cv2.CAP_PROP_FPS, settings.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        first_frame = _read_image(capture)
        if first_frame is None:
            raise CameraError(f"{target} opened but returned no first frame. {_OPEN_HINT}")
        height, width = first_frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        info = CameraInfo(capture.getBackendName(), width, height, fps if isfinite(fps) and fps > 0 else None)
        ready = True
        return capture, first_frame, info
    except cv2.error as exc:
        raise CameraError(f"{target} failed: {exc}. {_OPEN_HINT}") from exc
    finally:
        if capture is not None and not ready:
            capture.release()


def open_first(settings: CameraSettings):
    """Try each platform backend once, always using the selected index."""
    failures = []
    for backend in backend_names():
        try:
            return open_camera(settings, backend)
        except CameraError as exc:
            failures.append(str(exc))
    raise CameraError("; ".join(failures))


def frames(capture, first_frame, stop=None):
    """Yield the first image, then images or transient failures until stopped."""
    from cv2 import error as CaptureError

    if stop is not None and stop.is_set():
        return
    yield first_frame
    consecutive_failures = 0
    while stop is None or not stop.is_set():
        try:
            image = _read_image(capture)
        except CaptureError:
            image = None
        if stop is not None and stop.is_set():
            return
        if image is not None:
            consecutive_failures = 0
            yield image
            continue
        consecutive_failures += 1
        if consecutive_failures >= _FAILED_READ_LIMIT:
            raise CameraError("Camera stopped delivering frames after five consecutive failed reads. "
                              "Check its connection and choose Start/Retry.")
        yield None
        if stop is None:
            sleep(_FAILED_READ_DELAY)
        elif stop.wait(_FAILED_READ_DELAY):
            return
