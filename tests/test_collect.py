import sys
from types import SimpleNamespace

import numpy as np

from utils import detector
from utils.observations import Detection, Hand


def test_collection_toggles_limits_sampling_and_releases(tmp_path, monkeypatch):
    import collect_data
    from utils.dataset import DatasetManager

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    released = []
    capture = SimpleNamespace(release=lambda: released.append("capture"))
    camera = SimpleNamespace(CameraError=RuntimeError, CameraSettings=int,
                             open_first=lambda _: (capture, frame, None), frames=lambda *_: iter([frame] * 7))
    monkeypatch.setitem(sys.modules, "utils.camera", camera)
    keys = iter([32, -1, -1, -1, 32, -1, 113])
    cv2 = SimpleNamespace(imshow=lambda *_: None, waitKey=lambda _: next(keys), WND_PROP_VISIBLE=0,
                          destroyAllWindows=lambda: released.append("windows"),
                          getWindowProperty=lambda *_: 1, error=OSError)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "utils.visualization", SimpleNamespace(draw_status=lambda *_: None))
    times = iter([0.00, 0.05, 0.12, 0.249, 0.26, 0.52, 0.75])
    monkeypatch.setattr(collect_data, "monotonic", lambda: next(times))

    class Detector:
        def __enter__(self):
            self.index = 0
            return self

        def __exit__(self, *_):
            released.append("detector")

        def detect(self, _):
            self.index += 1
            points = np.column_stack((np.arange(21), np.arange(21) ** 2, np.full(21, 0.5))) * 0.001
            points[9, 1] += self.index / 100
            return Detection((Hand(points, "Left", 0.91),))

    monkeypatch.setattr(detector, "HandDetector", Detector)
    monkeypatch.setattr(detector, "render_preview", lambda *_: frame)
    path = tmp_path / "SPACE.csv"
    with DatasetManager(path, "SPACE") as dataset:
        collect_data.collect(SimpleNamespace(camera=0, label="SPACE"), dataset)
        assert dataset.count == 2
    assert released == ["detector", "capture", "windows"]
    assert len(path.read_text().splitlines()) == 3
