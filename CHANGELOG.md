# Changelog

## Unreleased — native desktop stabilization

### Added

- A–Z now uses SignSpeak `1.0.0-realsign`, pinned to the upstream commit. Explicit install and verify commands check both artifact hashes before inference. The app never downloads models at launch.
- The native PyQt6 window now features a structured Live Recognition inspector (large detected sign display, source label, confidence meter, hold-to-commit progress bar, and repeat-release cue), camera selection with friendly Linux V4L2 device names, Start/Resume/Retry, Pause, Stop, visible manual text-edit buttons (Space, Backspace, Clear Word, Clear, Copy), an in-app **Settings** dialog (live threshold/hold/cooldown tuning, `--mirror-input` toggle for left-handed/mirrored webcams, and Dark/Light/System themes), and an offline **Sign Guide** dialog. `webcam.py` lists and probes devices and can preview landmarks.
- Added `GeometricCustomGestureRecognizer` so all five webcam commands (`0`, `1`, `SPACE`, `BACKSPACE` = 1 middle finger, `CLEAR_WORD` = 2 middle fingers) work out of the box with 0/260 false positives on RealSign A–Z (optional local `custom_gestures.json` training still takes precedence when installed).
- Improved live two-hand and side-angle tracking (`min_detection_confidence=0.30`, `min_tracking_confidence=0.30`, duplicate-handedness slot resolution, and edge landmark clamping), added 3-frame probability smoothing for calmer A–Z guesses, added a live preview HUD pill (`Hands | Sign %` + hold bar), enlarged the translation display to 22pt DemiBold, and added a 1-click **Mirror input** toolbar toggle.
- Added `docs/realsign-alphabet-reference.png`, composed directly from the 26 CC0-1.0 RealSign `Letters/A.jpg`–`Letters/Z.jpg` reference photographs used to train SignSpeak `1.0.0-realsign`, embedded in `README.md` and bundled into PyInstaller builds for the offline Sign Guide.
- The five optional webcam commands (`0`, `1`, `SPACE`, `BACKSPACE`, `CLEAR_WORD`) train from three independent local sessions with an `OTHER` rejection class. Training holds out an entire session and installs bounded JSON parameters only if validation passes. No validated custom training data or model is shipped.
- Added locked Linux/Windows x64 dependencies for CPython 3.11/3.12, hash-verified setup scripts (`--require-hashes`, `SIGN_TRANSLATOR_VENV` on both Linux and Windows), deterministic tests, required real-model checks, and native PyInstaller smoke jobs that upload passing build artifacts to CI only.
- Added beginner setup guides, model and dataset credits, an updated native screenshot, RealSign evaluation metrics, and a [platform and verification audit](docs/AUDIT_REPORT.md).

### Changed

- Each active camera runs in a separate process with bounded preview and event channels, `CAP_PROP_BUFFERSIZE=1` for lower capture latency, early downscaling of oversized frames, single-pass `BGR→RGB` detection and preview rendering, and fast live Qt preview scaling. Failed or stalled streams stop and surface an actionable error; closing the window does not wait indefinitely on native capture.
- Gesture commits use timed holds, confidence and pose eligibility, and a genuine no-hands release for repeats. Destructive webcam gestures have stricter thresholds and cannot override a conflicting accepted alphabet prediction.
- The UI owns translation text. Manual editing and copy work independently of the camera and models; Pause and Stop keep the translation.

### Removed

- Removed the bundled Windows Python installer, the old mixed classifier runtime, and the duplicate collection, inference, landmark and handwritten smoke entry points. Use `python app.py`, `python webcam.py --preview --landmarks`, the custom training workflow, and pytest instead.
- Removed uncredited gesture charts and unsupported readiness claims. See the [README](README.md) for actual platform and device evidence.
