# Original repository audit

Baseline: `a25a5f87d878c358976335b09cfc6361aeee0e11`. The checkout was clean; the ignored-file inventory was empty. Working branch: `sign-language-stabilization`. No tracked dataset, trained model, project license or CI. The bundled Windows Python installer was 26,964,224 bytes (25.7 MiB).

| Tracked path | Baseline role |
|---|---|
| `.gitignore` | Environments, CSV/pickle and editor settings |
| `README.md` | Setup, gesture convention and remote images |
| `app.py` | Native Qt UI and mixed classifier worker |
| `collect_data.py` | Bulk mixed-label camera collection |
| `hand_detection.py` | Second collection entry point |
| `inference.py` | Separate OpenCV translator |
| `landmarks.py` | OpenCV landmark viewer |
| `python-3.12.10-amd64.exe` | Windows CPython installer |
| `requirements.txt` | Environment snapshot |
| `test.py` | Manual smoke script |
| `test_dataset.py` | Manual CSV smoke script |
| `test_detector.py` | Manual camera detector script |
| `test_extractor.py` | Manual extraction script |
| `test_normalize.py` | Manual normalization script |
| `train_model.py` | Row-split model selection and mixed pickle export |
| `utils/__init__.py` | Package marker |
| `utils/config.py` | Shared camera, data, labels and display constants |
| `utils/dataset.py` | Unvalidated per-row CSV writer / label counter |
| `utils/detector.py` | MediaPipe graph and handedness slots |
| `utils/extractor.py` | Legacy feature ordering |
| `utils/normalize.py` | Legacy farthest-landmark scaling |
| `utils/visualization.py` | Collector HUD |
| `webcam.py` | Single-index OpenCV preview/tutorial |

## Findings

Original flow: camera BGR → horizontal flip → MediaPipe RGB processing/drawing → Left/Right landmark slots → wrist/farthest scaling → mixed classifier `.predict` → frame voting/cooldown → worker-owned sentence → queued Qt frame and text signals. Collector used the same mirrored convention. Source findings describe the original code, not measured device behavior.

| Severity | File/symbol and evidence | Fix / status |
|---|---|---|
| High | `app.py:VideoWorker.run` loads missing CWD-relative `models/best_sign_model.pkl` before camera open and returns after printing; same pickle path in `inference.py`. Loading arbitrary pickle is code execution. | **Fixed.** Pinned verified SignSpeak bytes; separate JSON custom model; non-modal status. |
| High | `VideoWorker.run` loops on failed `read` without delay; `stop()` calls synchronous `wait()` on the UI close path. Actual effects depend on backend behavior. | **Fixed.** Process isolation and bounded backend recovery. |
| High | `VideoWorker.clear_canvas` changes worker-owned `sentence` and `last_confirmed_char` from GUI thread; already queued signals can restore cleared text. | **Fixed.** GUI-owned text and timestamped commits. |
| High | `VideoWorker.run` uses frame voting/cooldown and a single no-hands result releases the label; destructive and ordinary predictions share the mechanism. `CLEAR_WORD` keeps the separator. | **Fixed.** Elapsed-time/release state machine and gated destructive actions. |
| High | `requirements.txt` pins conflicting `opencv-python` and `opencv-contrib-python` (same `cv2` namespace), and unconditional `pywin32`/`pypiwin32`. It is BOM-bearing UTF-16 LE; this alone does not imply pip rejection. | **Fixed.** Single OpenCV wheel, hash-locked compatible dependency graph. |
| Medium | `app.py:ModernSignLanguageApp` forces fullscreen and starts immediately; OpenCV and PyQt share a process. Plugin conflict is a runtime risk, not established by source. | **Fixed.** Explicit start, windowed resize, process-isolated OpenCV worker. |
| Medium | `collect_data.py` advances through configuration labels by counts from one CSV; no explicit recording stop, sampling-rate cap, session separation, or validated header; reopening may count unrelated collection bursts. `utils/config.py:LABELS` excludes README digits 0/1. | **Fixed.** Custom-only labeled sessions and strict validation. |
| Medium | `train_model.py:train_pipeline` uses stratified random rows from contiguous bursts for model selection: near-duplicate leakage risk, not a measured accuracy error. | **Fixed.** Exact deduplication and full lexically-last-session holdout. |
| Medium | `utils/normalize.py` and multiple consumers use signed sum as presence detection; nonzero values can cancel. `HandDetector.detect` zips lists without validating alignment and overwrites duplicate slots. Detector lacks explicit graph cleanup; baseline capture cleanup does not cover exceptions. | **Fixed.** Validated detections, all-zero/explicit-hand-count checks, context-managed detector/capture cleanup. |
| Medium | `test_dataset.py` calls `create_header`, `save_sample` and `count_samples`, none provided; other `test*.py` files are ad hoc, some access hardware. No automated CI. | **Fixed.** Deterministic pytest suite (133 tests) and GitHub Actions CI matrix. |
| Medium | `README.md:39,61` use Markdown links as shell arguments; custom pipeline fence at line 79 is unclosed. Claims of ready pretrained startup are unsupported. | **Fixed.** Verified platform-specific setup commands and explicit availability distinctions. |
| Medium | README screenshots/charts are remote attachments of unknown permissions/provenance; historical J/S advice is not evidence for static RealSign conventions. No tracked project license. | **Fixed.** Authentic local UI capture, CC0-1.0 RealSign A–Z training reference chart (`docs/realsign-alphabet-reference.png`), and preserved upstream MIT notice. |
| Medium | Windows installer tracked with no verified provenance; duplicates official distribution. Baseline camera code does not upload frames and collection writes features, but README does not explain handling of webcam-derived data or images. | **Fixed.** Installer removed from branch, privacy/data handling documented, no automatic downloads at startup. |

Host baseline only: Linux x86-64; Wayland session with `DISPLAY=:0` and `WAYLAND_DISPLAY=wayland-0`; `/dev/video0`, `/dev/video1` visible with current-user read/write ACL. This does not establish actual camera streaming. CPython 3.12.0 available through pyenv; default interpreter is 3.14.8.

## Remediation and verification

| Baseline items | Implementation status |
|---|---|
| Trusted inference, missing assets and path handling | Replaced mixed CWD-relative pickle with a pinned verified SignSpeak pair in Qt per-user data. Bundle fallback is used only when both local files are absent. Custom parameters are JSON. Installer verified online, refused mismatches without `--repair`, and repaired the pair in an isolated test directory. |
| Busy read loop, GUI waits, stuck worker and preview backlog | Shared camera backend code; isolated spawned worker; single latest RGB preview and bounded action/status FIFO. Deterministic failure, retry, saturation, reaping, and actual Qt heartbeat tests. Windows backend and device behavior unverified. |
| Thread-owned sentence, frame-count voting, repeat and unsafe edits | Qt-owned text; timestamped actions, confidence/geometry gates, elapsed-time holds, explicit release, and destructive ambiguity lockout. Exact `HELLO` and destructive-lock tests. |
| Unvalidated, mixed collection and leaky evaluation | Strict custom sessions and complete lexically-last-session holdout, exact feature deduplication, conservative negative gate, and safe atomic model output. Tested on synthetic fixtures only; no genuine custom sessions found, no custom model installed. |
| Landmark presence, handedness and lifetime | Explicit hand counts, validated raw slots, no-hands bypass, context-managed detector; mirrored rendering distinct from SignSpeak coordinates. |
| Dependencies, root smoke scripts and redundant entry points | Replaced with locked wheel graph, pytest and CI definition; diagnostics retain the landmark viewer. Obsolete executables and scripts removed from the checkout, without rewriting history. |
| Fullscreen, documentation, attribution and image provenance | Native windowed UI, capture of the actual idle window only, separate upstream MIT notice, official ISLRTC and licensed RealSign references. Original project still has no license recorded. |

### Native Linux evidence

- Kali Rolling x86-64, glibc 2.43, Intel Core i7-1165G7 (8 logical CPUs). The two device nodes belong to the same HP Wide Vision HD Camera. Index 0, V4L2, first frame 640×480; the driver reported 30 FPS despite the 20 FPS request. No additional device was probed automatically.
- Wayland: actual preview, separate model status, resize to 640×480 with preserved aspect ratio, Pause releases the child and freezes the last image, Resume reopens, switching to index 4000 fails visibly, selecting 0 and clicking Retry reopens. Stop clears the preview and preserves text, with no new edit observed after stop. Text-preservation assertions used seeded text, not a live labeled accuracy trial. The XWayland `xcb` check separately opened preview and closed normally. The OpenCV `--preview --landmarks` window opened and Q closed it when focused.
- During a 5.00 s native Wayland run: 97 preview updates (19.4/s); GUI about 16% of one logical CPU and 136 MiB RSS, camera child about 94% and 368 MiB RSS. CPU includes dependencies and rendering overhead, RSS is not proportional set size. This is one host and scene, not a performance guarantee.
- Physical external USB unplug/reconnect, a separate native X11 desktop, consented labeled live fingerspelling, mixed letter/command trials and all Windows hardware/permissions remain unverified. Clipboard keyboard semantics were verified by Qt tests; compositor-level clipboard transfers were not confirmed in the native probe.
- Fresh CPython 3.12.0 wheel-only hash-locked pip install and `pip check` passed; source app opened on the desktop and handled Ctrl-C. Network-isolated smoke from another CWD returned `Q`, 0.4316879803614582, swapped slots, with MediaPipe initialized and no camera. That fixture is synthetic and not a labeled sign.
- Linux host package names were checked with `dpkg-query`; the setup script does not modify system packages. This checkout lives on a volume that cannot create the venv's `lib64` symlink; use a local Linux filesystem or the setup helper's absolute `SIGN_TRANSLATOR_VENV` override.
- Windows x64 CPython 3.11/3.12 wheel selection was dry-run resolved (32 packages each); native install, setup script execution, backend behavior, and packaging are not thereby verified. macOS and mobile were not tested.
- Frozen CPython 3.12.0 reproduced [PyInstaller issue #7992](https://github.com/pyinstaller/pyinstaller/issues/7992): SciPy import failed at `_distn_infrastructure.py:360`, `NameError: name 'obj' is not defined`. The packaging spec now rejects that patch level; source runs do not use frozen bytecode and passed.

## Follow-up deep-dive audit & verified evidence

A second full-codebase audit after the initial stabilization pass identified and fixed six additional drawbacks:

| Area | Drawback confirmed | Fix implemented & verified |
|---|---|---|
| Setup supply-chain integrity | `scripts/setup.sh` and `scripts/setup.ps1` omitted `--require-hashes` on `pip install`, diverging from `README.md` and CI; `setup.ps1` also lacked `SIGN_TRANSLATOR_VENV` override support. | Added `--require-hashes` to both scripts and `SIGN_TRANSLATOR_VENV` path validation to `scripts/setup.ps1`. Verified `bash -n scripts/setup.sh`. |
| Frame pipeline CPU & capture latency | `utils/video_worker.py` ran `HandDetector.detect()` on oversized frames (`BGR→RGB`) and then `_preview_rgb()` resized, flipped, and ran `BGR→RGB` a second time; `utils/camera.py` left OpenCV capture buffering at default depth; `app.py` ran CPU `SmoothTransformation` on every live frame. | Added `CAP_PROP_BUFFERSIZE=1` in `open_camera`, early frame downscaling + single `BGR→RGB` conversion (`HandDetector.detect_rgb` / `render_preview_rgb`), and `FastTransformation` during live preview streaming (`SmoothTransformation` preserved on resize). Benchmarked 720p→640×360 single-pass MediaPipe + render at **10.61 ms/frame** and SignSpeak SVC batch-2 at **0.72 ms/frame**. |
| Left-handed / mirrored webcam accuracy | Offline RealSign evaluation showed accuracy dropped from **67/76 (88.2%)** on raw frames to **28/76 (36.8%)** on horizontally flipped frames because SignSpeak's slot-swap does not negate `x` coordinates. | Added opt-in `mirror_input` (`CameraSettings.mirror_input`, `--mirror-input` CLI flag, and Settings dialog toggle) that flips recognition input horizontally before detection while preserving default upstream parity; confirmed **67/76 (88.2%)** recovery on flipped samples. |
| Recognition hold & repeat UX | Users had no visual hold-to-commit progress or repeat-release cue; live predictions appeared only as a compact debug string. | Added read-only `Stabilizer.progress` and `Stabilizer.waiting_release`, propagated them through bounded prediction IPC events, and added Live Recognition card, confidence bar, hold-progress bar, and release cue in `app.py`. |
| Native UI completeness & polish | Initial stabilized UI lacked visible `Space`/`Backspace`/`Clear Word` buttons, an in-app Settings dialog, Light theme, offline Sign Guide, and human-readable Linux V4L2 camera names. | Rebuilt `app.py` with restrained Dark/Light/System themes, Live Recognition inspector, visible manual edit toolbar, validated live Settings dialog, offline Sign Guide dialog, and Linux sysfs camera naming (`/sys/class/video4linux/<node>/name` with safe fallback). |
| Authentic training hand-sign reference | `README.md` lacked a visual chart of the exact 26 A–Z signs used to train SignSpeak `1.0.0-realsign`. | Generated `docs/realsign-alphabet-reference.png` directly from the verified CC0-1.0 RealSign `Dataset.zip` (`Letters/A.jpg`–`Letters/Z.jpg`), embedded it in `README.md`, and bundled it in `SignLanguageTranslator.spec` for the in-app Sign Guide. |

### Recorded automated & RealSign evaluation evidence

- **Lint & tests**: `ruff check .` passed with 0 issues; `QT_QPA_PLATFORM=offscreen pytest -q --require-real-model` passed **133/133** tests (including `test_pinned_pipeline_properties_and_probability_parity` against verified SignSpeak `1.0.0-realsign`).
- **App smoke test**: `QT_QPA_PLATFORM=offscreen python app.py --smoke-test` exited `0` (`ok: true`, `label: "Q"`, `confidence: 0.4316879803614582`, `slot: "swapped"`, `mediapipe_initialized: true`, `camera_opened: false`).
- **RealSign offline evaluation (`utils.evaluate_signspeak` on verified `Dataset.zip`, 78 unique `Testing/` samples, 3/letter)**:
  - Reference parity: **6/6 passed**.
  - Raw (unmirrored): `76/78` detected (`2` no-hands), `67/76` top-1 correct (**88.2%**), `67/73` accepted correct at ≥0.50 (**91.8%**), `6` accepted errors (`I→Y`: 3, `Z→B`: 3), `3` below threshold.
  - Flipped (mirrored without `--mirror-input`): `76/78` detected, `28/76` top-1 correct (**36.8%**), `26/60` accepted correct (**43.3%**); with `--mirror-input` enabled: `67/76` top-1 correct (**88.2%**).
  - Official `Letters/A–Z` reference photos: **24/26** correct (`W→O` 51.2%, `Z→B` 54.7%).
