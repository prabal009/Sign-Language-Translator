# PyInstaller native onedir build. Run the verified asset installer first.
import sys
from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files

if sys.version_info[:3] == (3, 12, 0):
    raise SystemExit("Build with CPython 3.12.1 or newer: 3.12.0 breaks frozen SciPy imports (PyInstaller #7992).")

root = Path(SPECPATH)
sys.path.insert(0, str(root))
from utils.model_assets import ASSETS, verified_pair

model_data = Path(CONF["workpath"]) / "verified-signspeak"
model_data.mkdir(parents=True, exist_ok=True)
data = collect_data_files("mediapipe")
for asset, content in zip(ASSETS, verified_pair(), strict=True):
    staged = model_data / asset.name
    staged.write_bytes(content)
    data.append((str(staged), "signspeak-assets"))
data.append((str(root / "docs" / "SignSpeak-MIT.txt"), "signspeak-assets"))
reference_chart = root / "docs" / "realsign-alphabet-reference.png"
if reference_chart.is_file():
    data.append((str(reference_chart), "docs"))

analysis = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    datas=data,
    # The pinned artifact references this NumPy 2 pickle path; NumPy 1.26 exposes a compatibility module.
    hiddenimports=["numpy._core", "numpy._core.multiarray"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SignLanguageTranslator",
    console=sys.platform != "win32",
    upx=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    name="SignLanguageTranslator",
    upx=False,
)
