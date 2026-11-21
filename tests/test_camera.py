import pickle
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from utils import camera, config


class FakeOpenCVError(RuntimeError):
    pass


class FakeCapture:
    def __init__(self, events, *, opens=True, reads=(), backend="FAKE", fps=0.0):
        self.events = events
        self.opens = opens
        self.reads = iter(reads)
        self.backend = backend
        self.fps = fps
        self.backend_id = None

    def open(self, index, backend_id):
        self.backend_id = backend_id
        self.events.append(("open", index, backend_id))
        if isinstance(self.opens, Exception):
            raise self.opens
        return self.opens

    def set(self, prop, value):
        self.events.append(("set", prop, value))
        return False  # Unsupported requests must not prevent reading actual settings.

    def read(self):
        self.events.append(("read", self.backend_id))
        result = next(self.reads)
        if isinstance(result, Exception):
            raise result
        return result

    def get(self, prop):
        assert prop == 5
        return self.fps

    def getBackendName(self):
        return self.backend

    def release(self):
        self.events.append(("release", self.backend_id))


def install_fake_cv2(monkeypatch, *captures):
    attempts = iter(captures)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=lambda: next(attempts), error=FakeOpenCVError,
        CAP_V4L2=200, CAP_MSMF=1400, CAP_DSHOW=700, CAP_ANY=0,
        CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5,
    ))


def test_linux_choices_use_actual_node_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "linux")
    monkeypatch.setattr(camera, "_DEV_DIR", tmp_path)
    for name in ("video10", "video2", "video3-metadata", "videox", "video01", "video2147483648"):
        (tmp_path / name).touch()
    (tmp_path / "video1").mkdir()
    assert camera.camera_choices() == [(2, "video2"), (10, "video10")]
    assert camera.default_camera() == 2
    assert camera.backend_names() == ("V4L2", "ANY")


@pytest.mark.parametrize("platform,backends", [("win32", ("MSMF", "DSHOW", "ANY")),
                                                ("darwin", ("ANY",))])
def test_camera_choices_and_backend_order(platform, backends, monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", platform)
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert camera.backend_names() == backends
    assert camera.default_camera() == 0
    choices = camera.camera_choices()
    if platform == "win32":
        assert [index for index, _ in choices] == list(range(8))
        assert all("candidate" in name for _, name in choices)
    else:
        assert choices == []


def test_camera_settings_frozen_with_defaults_and_picklable():
    settings = camera.CameraSettings(0)
    assert (settings.width, settings.height, settings.fps) == (640, 480, 20)
    assert pickle.loads(pickle.dumps(settings)) == settings
    with pytest.raises(FrozenInstanceError):
        settings.index = 1
    assert camera.CameraSettings(config.MAX_INDEX, 1920, 1080, 60).index == 2**31 - 1
    assert camera.CameraSettings(8, fps=29.97).fps == 29.97


@pytest.mark.parametrize("options", [
    {"index": -1}, {"index": 2**31}, {"index": 1.5}, {"index": True},
    {"width": 0}, {"width": 1921}, {"width": 10.5}, {"width": False},
    {"height": -1}, {"height": 1081},
    {"fps": 0}, {"fps": -1}, {"fps": 61}, {"fps": True}, {"fps": "20"},
    {"fps": float("inf")}, {"fps": float("nan")},
    {"mirror_input": "yes"}, {"mirror_input": 1},
])
def test_camera_settings_reject_invalid_values(options):
    with pytest.raises(ValueError):
        camera.CameraSettings(**({"index": 0} | options))


def test_open_first_releases_failed_frame_then_uses_same_index(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "linux")
    image = np.empty((240, 320, 3), dtype=np.uint8)
    events = []
    first = FakeCapture(events, reads=[(False, None)])
    fallback = FakeCapture(events, reads=[(True, image)], backend="V4L2", fps=29.97)
    install_fake_cv2(monkeypatch, first, fallback)

    capture, frame, info = camera.open_first(camera.CameraSettings(7, 800, 600, 30))
    assert capture is fallback
    assert frame is image
    assert info == camera.CameraInfo("V4L2", 320, 240, 29.97)
    assert info.describe() == "V4L2: 320×240 at 29.97 FPS"
    with pytest.raises(FrozenInstanceError):
        info.width = 800
    request = [("set", 3, 800), ("set", 4, 600), ("set", 5, 30)]
    assert events == [
        ("open", 7, 200), *request, ("read", 200), ("release", 200),
        ("open", 7, 0), *request, ("read", 0),
    ]
    capture.release()
    assert events[-1] == ("release", 0)


@pytest.mark.parametrize("opens,reads,message", [
    (False, (), "could not open"),
    (FakeOpenCVError("native open failed"), (), "native open failed"),
    (True, (FakeOpenCVError("native read failed"),), "native read failed"),
    (True, ((True, np.empty((0, 0, 3))),), "no first frame"),
])
def test_open_camera_releases_handle_on_failure(opens, reads, message, monkeypatch):
    events = []
    install_fake_cv2(monkeypatch, FakeCapture(events, opens=opens, reads=reads))
    with pytest.raises(camera.CameraError, match=message) as error:
        camera.open_camera(camera.CameraSettings(7), "DSHOW")
    assert "permissions" in str(error.value)
    assert events[0] == ("open", 7, 700)
    assert events.count(("release", 700)) == 1
    assert events[-1] == ("release", 700)


def test_open_first_stops_after_all_backends_and_allows_explicit_retry(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "win32")
    events = []
    image = np.empty((120, 160, 3), dtype=np.uint8)
    reconnected = FakeCapture(events, reads=[(True, image)], backend="MSMF", fps=float("nan"))
    install_fake_cv2(monkeypatch, *(FakeCapture(events, opens=False) for _ in range(3)), reconnected)
    settings = camera.CameraSettings(14)
    with pytest.raises(camera.CameraError) as error:
        camera.open_first(settings)
    assert all(name in str(error.value) for name in ("MSMF", "DSHOW", "ANY"))
    assert events == [event for api in (1400, 700, 0) for event in (("open", 14, api), ("release", api))]

    capture, frame, info = camera.open_first(settings)
    assert capture is reconnected and frame is image
    assert info.fps is None
    assert info.describe() == "MSMF: 160×120 (FPS not reported)"
    capture.release()


def test_frames_resets_failure_count_after_success_then_stops_at_five(monkeypatch):
    image = np.empty((48, 64, 3), dtype=np.uint8)
    failure = (False, None)
    events = []
    reads = [failure] * 4 + [(True, image), FakeOpenCVError("read error")] + [failure] * 4
    capture = FakeCapture(events, reads=reads)
    install_fake_cv2(monkeypatch)
    monkeypatch.setattr(camera, "sleep", lambda delay: events.append(("wait", delay)))
    stream = camera.frames(capture, image)

    assert next(stream) is image
    assert [next(stream) for _ in range(4)] == [None] * 4
    assert next(stream) is image
    assert [next(stream) for _ in range(4)] == [None] * 4
    with pytest.raises(camera.CameraError, match="five consecutive failed reads.*Start/Retry"):
        next(stream)
    read = ("read", None)
    wait = ("wait", 0.05)
    assert events == [read, wait] * 4 + [read] + [read, wait] * 4 + [read]


def test_frames_stop_interrupts_failure_wait(monkeypatch):
    image = np.empty((48, 64, 3), dtype=np.uint8)
    events = []
    capture = FakeCapture(events, reads=[(False, None)])
    install_fake_cv2(monkeypatch)
    stopped = False

    def stop_during_wait(delay):
        assert delay >= 0.05
        nonlocal stopped
        stopped = True
        return True

    stop = SimpleNamespace(is_set=lambda: stopped, wait=stop_during_wait)
    stream = camera.frames(capture, image, stop)
    assert next(stream) is image
    assert next(stream) is None
    assert list(stream) == []
    assert list(camera.frames(capture, image, stop)) == []
    assert events == [("read", None)]


def test_frames_does_not_yield_an_image_returned_after_stop(monkeypatch):
    image = np.empty((48, 64, 3), dtype=np.uint8)
    install_fake_cv2(monkeypatch)
    stopped = False

    def read():
        nonlocal stopped
        stopped = True
        return True, image

    capture = SimpleNamespace(read=read)
    stop = SimpleNamespace(is_set=lambda: stopped)
    stream = camera.frames(capture, image, stop)
    assert next(stream) is image
    assert list(stream) == []


def test_large_index_does_not_use_opencv_backend_encoded_index(monkeypatch):
    def unexpected_capture():
        pytest.fail("CAP_ANY silently rewrites large camera indices")

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=unexpected_capture))
    with pytest.raises(camera.CameraError, match="changing its index"):
        camera.open_camera(camera.CameraSettings(200), "ANY")
    with pytest.raises(camera.CameraError, match="changing its index"):
        camera.open_camera(camera.CameraSettings(701), "ANY")
