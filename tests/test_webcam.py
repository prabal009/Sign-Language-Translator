import time

import webcam
from utils.camera import CameraSettings


def _blocked_probe(*_):
    time.sleep(60)


def test_diagnostic_bounds_blocked_native_open(monkeypatch):
    monkeypatch.setattr(webcam, "_diagnose", _blocked_probe)
    monkeypatch.setattr(webcam, "START_TIMEOUT", 0.15)
    begin = time.monotonic()
    opened, error = webcam._probe(CameraSettings(0), "ANY", False, False)
    assert not opened and "timed out" in error
    assert time.monotonic() - begin < 4


def test_probe_falls_back_only_before_first_frame(monkeypatch, capsys):
    attempts = []

    def probe(settings, backend, preview, landmarks):
        attempts.append((settings.index, backend, preview, landmarks))
        return (False, "can't open") if backend == "MSMF" else (True, "stream failed")

    monkeypatch.setattr(webcam, "backend_names", lambda: ("MSMF", "DSHOW", "ANY"))
    monkeypatch.setattr(webcam, "_probe", probe)
    assert webcam.main(["--camera", "5", "--preview", "--landmarks"]) == 1
    assert attempts == [(5, "MSMF", True, True), (5, "DSHOW", True, True)]
    assert "stream failed" in capsys.readouterr().err
