"""Collect local feature rows for one custom gesture in one independent session."""

import argparse
import sys
from time import monotonic

from utils.config import MAX_INDEX
from utils.custom_gestures import DESCRIPTIONS, pose_eligible, sample_detection
from utils.dataset import DatasetError, DatasetManager, valid_session
from utils.model_assets import data_dir


def collect(args, dataset):
    from utils.camera import CameraError, CameraSettings, frames, open_first
    from utils.detector import HandDetector, render_preview
    from utils.visualization import draw_status

    import cv2

    name = "Custom gesture collector"
    recording = False
    window_created = False
    sampled = accepted = rejected = duplicates = 0
    next_sample = 0.0
    initial = dataset.count
    print(f"{args.label}: {DESCRIPTIONS[args.label]}")
    print(f"{initial} existing unique rows; aim for at least 30. Space starts/stops; Q quits.")
    capture, first, _ = open_first(CameraSettings(args.camera))
    try:
        with HandDetector() as detector:
            for frame in frames(capture, first):
                if frame is not None:
                    now = monotonic()
                    detection = detector.detect(frame)
                    h, w = frame.shape[:2]
                    sample = sample_detection(detection, w / h)
                    eligible = sample is not None and pose_eligible(args.label, sample)
                    if recording and now >= next_sample:
                        next_sample = now + 0.20
                        sampled += 1
                        if not eligible:
                            rejected += 1
                        elif dataset.append(sample):
                            accepted += 1
                        else:
                            duplicates += 1
                    preview = render_preview(frame, detection)
                    draw_status(preview, (
                        f"{args.label}: {DESCRIPTIONS[args.label]}",
                        f"{'RECORDING' if recording else 'PAUSED'} | {dataset.count} unique rows | SPACE toggle | Q quit",
                        "Pose eligible" if eligible else "Waiting for a valid target pose",
                    ))
                    cv2.imshow(name, preview)
                    window_created = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")) or (
                    window_created and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                if key == ord(" "):
                    recording = not recording
    except cv2.error as exc:
        raise CameraError(f"OpenCV preview failed: {exc}") from exc
    finally:
        capture.release()
        if window_created:
            cv2.destroyAllWindows()
        print(f"Sampled {sampled}: {accepted} saved, {rejected} rejected, {duplicates} duplicate; "
              f"{dataset.count} total unique.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="Camera index (default: first listed)")
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True, choices=DESCRIPTIONS)
    args = parser.parse_args(argv)
    if args.camera is not None and not 0 <= args.camera <= MAX_INDEX:
        parser.error("--camera must be a nonnegative OpenCV index")
    if not valid_session(args.session):
        parser.error("--session must match [A-Za-z0-9_-]{1,48}")
    from utils.camera import CameraError, default_camera

    if args.camera is None:
        args.camera = default_camera()

    path = data_dir() / "custom-data" / args.session / f"{args.label}.csv"
    try:
        with DatasetManager(path, args.label) as dataset:
            print(f"Writing feature rows only: {path}")
            collect(args, dataset)
    except (CameraError, DatasetError, OSError) as exc:
        print(f"Collection stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())