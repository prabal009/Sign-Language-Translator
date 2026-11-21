"""List, probe or preview one camera without loading recognition models."""

import argparse
import multiprocessing as mp
import sys
from contextlib import ExitStack
from time import monotonic

from utils.camera import (
    START_TIMEOUT,
    STREAM_TIMEOUT,
    CameraSettings,
    backend_names,
    camera_choices,
    default_camera,
    frames,
    open_camera,
)
from utils.config import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def _diagnose(settings, backend, preview, landmarks, messages, stop, progress):
    opened = False
    with messages:
        try:
            with ExitStack() as stack:
                capture, first, info = open_camera(settings, backend)
                stack.callback(capture.release)
                opened = True
                progress.value = monotonic()
                messages.send(("opened", info.describe()))
                if preview:
                    import cv2

                    from utils.detector import HandDetector, render_preview

                    name = "Camera diagnostic (Q to quit)"
                    detector = stack.enter_context(HandDetector()) if landmarks else None
                    stack.callback(cv2.destroyAllWindows)
                    displayed = False
                    for frame in frames(capture, first, stop):
                        if frame is not None:
                            image = (render_preview(frame, detector.detect(frame), indices=True)
                                     if detector else cv2.flip(frame, 1))
                            cv2.imshow(name, image)
                            displayed = True
                            progress.value = monotonic()
                        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                            break
                        if displayed and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                            break
            messages.send(("done", ""))
        except Exception as exc:
            messages.send(("stream-error" if opened else "open-error", str(exc)))


def _probe(settings, backend, preview, landmarks):
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    progress = ctx.RawValue("d", 0.0)
    process = ctx.Process(target=_diagnose, args=(settings, backend, preview, landmarks, sender, stop, progress))
    started = monotonic()
    opened = False
    with receiver, sender:
        process.start()
        sender.close()
        try:
            while True:
                if receiver.poll(0.10):
                    try:
                        kind, detail = receiver.recv()
                    except EOFError:
                        return opened, f"{backend} diagnostic process exited unexpectedly"
                    if kind == "opened":
                        opened = True
                        print(detail, flush=True)
                    elif kind == "done":
                        return True, ""
                    else:
                        return kind == "stream-error", detail
                now = monotonic()
                if not opened and now - started >= START_TIMEOUT:
                    return False, f"{backend} timed out opening and reading the first frame"
                if opened and now - progress.value >= STREAM_TIMEOUT:
                    return True, "Preview stalled; reconnect the camera and rerun the diagnostic"
        finally:
            stop.set()
            process.join(0.75)
            if process.is_alive():
                process.terminate()
                process.join(0.75)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List video nodes or candidate indices without opening them")
    parser.add_argument("--camera", type=int, help="Camera index (default: first listed)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--preview", action="store_true", help="Show mirrored preview; Q or Esc quits")
    parser.add_argument("--landmarks", action="store_true", help="Annotate hand labels and landmark indices in preview")
    args = parser.parse_args(argv)
    if args.landmarks and not args.preview:
        parser.error("--landmarks requires --preview")
    if args.list:
        if choices := camera_choices():
            for index, name in choices:
                print(f"{index}: {name}")
        else:
            print("No devices enumerated; specify a nonnegative index (for example --camera 0).")
        return 0
    try:
        settings = CameraSettings(default_camera() if args.camera is None else args.camera,
                                  args.width, args.height, args.fps)
    except ValueError as exc:
        parser.error(str(exc))
    for backend in backend_names():
        try:
            opened, error = _probe(settings, backend, args.preview, args.landmarks)
        except KeyboardInterrupt:
            return 130
        except OSError as exc:
            print(f"Cannot start camera diagnostic process: {exc}", file=sys.stderr)
            return 1
        if not error:
            return 0
        print(error, file=sys.stderr)
        if opened:
            return 1
    print("No frame. Check camera selection, OS camera permissions and other apps using the device.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())