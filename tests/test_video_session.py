import multiprocessing as mp
import os
import signal
import time

import numpy as np

from utils.camera import CameraSettings
from utils.recognition import Thresholds
from utils.temporal import Timing
from utils.video_session import VideoSession
from utils.video_worker import Channels


def _run_until(predicate, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if value := predicate():
            return value
        time.sleep(0.005)
    raise AssertionError("timed out waiting for worker state")


def _close(worker):
    worker.stop()

    def reaped():
        worker.tick()
        return not worker.active

    _run_until(reaped, 5)


def _fake_video(settings, backend, _thresholds, _timing, channels):
    if backend == "V4L2":
        channels.publish({"kind": "open-error", "message": "test cannot read first frame"})
        return
    channels.publish({"kind": "opened", "message": f"test camera {settings.index}"})
    channels.publish({"kind": "alphabet", "message": "test alphabet ready"})
    channels.publish({"kind": "custom", "message": "test custom unavailable"})
    channels.publish({"kind": "ready"})
    channels.write_preview(np.full((2, 4, 3), settings.index, dtype=np.uint8))
    channels.publish({"kind": "action", "label": "A" if settings.index == 1 else "B", "timestamp": time.monotonic()})
    channels.stop.wait(20)
    channels.publish({"kind": "action", "label": "Z", "timestamp": time.monotonic()})


def _stuck(_settings, _backend, _thresholds, _timing, channels):
    if os.name == "posix":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    channels.publish({"kind": "opened", "message": "test native capture"})
    time.sleep(60)


def test_only_latest_frame_and_fifo_saturation():
    channels = Channels(mp.get_context("spawn"), 4, 2)
    try:
        for value in range(50):
            channels.write_preview(np.full((2, 4, 3), value, dtype=np.uint8))
        image = channels.read_preview(0)
        assert (image.width, image.height, image.data) == (4, 2, bytes([49]) * 24)
        assert channels.read_preview(image.sequence) is None
        assert channels.publish({
            "kind": "prediction", "state": "usable", "reason": "", "source": "alphabet", "label": "A",
            "alphabet": ["A", 0.92], "custom": None, "progress": 65, "waiting_release": False,
        })
        assert channels.drain()[0]["progress"] == 65
        for i in range(32):
            assert channels.publish({"kind": "action", "label": "A", "timestamp": float(i)})
        assert not channels.publish({"kind": "action", "label": "B", "timestamp": 32.0})
        assert channels.saturated
        events = []

        def all_read():
            events.extend(channels.drain())
            return len(events) >= 32

        _run_until(all_read)
        assert [(event["label"], event["timestamp"]) for event in events] == [("A", i) for i in range(32)]
    finally:
        channels.close()


def test_fallback_switch_and_stop_discard_stale_messages():
    worker = VideoSession(target=_fake_video, backends=("V4L2", "ANY"))
    thresholds, timing = Thresholds(), Timing()
    events, previews = [], []

    def update():
        batch, latest = worker.tick()
        events.extend(batch)
        if latest:
            previews.append(latest)
        return worker.state == "running" and bool(events) and bool(previews)

    try:
        worker.start(CameraSettings(1, 4, 2), thresholds, timing)
        _run_until(update)
        assert "test camera 1" in worker.camera_status
        assert worker.alphabet_status == "test alphabet ready" and worker.custom_status == "test custom unavailable"
        assert [event["label"] for event in events] == ["A"]
        assert previews[-1].data == bytes([1]) * 24
        assert worker.tick() == ([], None)
        events.clear()
        previews.clear()
        worker.start(CameraSettings(2, 4, 2), thresholds, timing)
        _run_until(update)
        assert "test camera 2" in worker.camera_status
        assert [event["label"] for event in events] == ["B"]
        assert previews[-1].data == bytes([2]) * 24
        _close(worker)
        assert worker.tick() == ([], None)
    finally:
        _close(worker)


def test_stuck_native_child_is_reaped_without_blocking_updates():
    worker = VideoSession(target=_stuck, backends=("ANY",))
    try:
        worker.start(CameraSettings(0, 4, 2), Thresholds(), Timing())

        def initialized_capture():
            worker.tick()
            return worker.state == "initializing"

        _run_until(initialized_capture)
        start = time.monotonic()
        worker.stop()
        ticks = []

        def reaped():
            before = time.monotonic()
            assert worker.tick() == ([], None)
            ticks.append(time.monotonic() - before)
            return not worker.active

        _run_until(reaped, 5)
        assert time.monotonic() - start < 4
        assert len(ticks) > 8 and max(ticks) < 0.20
    finally:
        _close(worker)


def test_stall_requires_explicit_retry(monkeypatch):
    from utils import video_session

    monkeypatch.setattr(video_session, "STREAM_TIMEOUT", 0.25)
    worker = VideoSession(target=_fake_video, backends=("ANY",))
    settings = CameraSettings(1, 4, 2)
    thresholds, timing = Thresholds(), Timing()

    def stalled():
        worker.tick()
        return worker.state == "failed"

    def running():
        events, preview = worker.tick()
        assert all(event["label"] != "Z" for event in events)
        return worker.state == "running" and preview is not None

    try:
        worker.start(settings, thresholds, timing)
        _run_until(running)
        _run_until(stalled)
        assert "stall" in worker.camera_status.lower()
        _run_until(lambda: stalled() and not worker.active)
        assert worker.tick() == ([], None)
        worker.start(settings, thresholds, timing)
        _run_until(running)
    finally:
        _close(worker)
