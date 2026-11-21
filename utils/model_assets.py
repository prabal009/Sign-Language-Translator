"""Install or verify the immutable upstream SignSpeak release."""

import argparse
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

APP_NAME = "SignLanguageTranslator"
COMMIT = "6fc64c720e168d32f3c1a968d797be4af56e36cb"
BASE_URL = f"https://raw.githubusercontent.com/arpita1505/SignSpeak-AI/{COMMIT}/artifacts/"
INSTALL_HINT = "Run python -m utils.model_assets install (add --repair for corrupt files)."


class AssetError(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    digest: str


MODEL = Asset("signspeak_model.joblib", 2992157,
              "f9e39d339bf7a4cc882ba0bbb112039898a10f69c13a35429f4a75fda7d1c9d0")
METADATA = Asset("model_metadata.json", 3898,
                 "8b65fe0f5ab400ebb7a396906cd72841775d4ebdb484c3eb0c179d3edac2637d")
ASSETS = MODEL, METADATA


def configure_qt_identity():
    from PyQt6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName(APP_NAME)
    QCoreApplication.setApplicationName(APP_NAME)


def data_dir() -> Path:
    from PyQt6.QtCore import QStandardPaths

    configure_qt_identity()
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not location:
        raise AssetError("Qt did not provide a per-user application data directory")
    return Path(location)


def check_bytes(asset: Asset, content: bytes) -> bytes:
    if len(content) != asset.size or hashlib.sha256(content).hexdigest() != asset.digest:
        raise AssetError(f"Unverified {asset.name}: pinned size or SHA-256 differs. {INSTALL_HINT}")
    return content


def present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read(directory: Path, asset: Asset) -> bytes:
    try:
        with (directory / asset.name).open("rb") as file:
            return check_bytes(asset, file.read(asset.size + 1))
    except OSError as exc:
        raise AssetError(f"Cannot read {asset.name}: {exc.strerror or type(exc).__name__}. {INSTALL_HINT} "
                         f"Asset directory: {directory}") from exc


def verified_pair(directory: Path | None = None, *, bundled: bool = False) -> tuple[bytes, bytes]:
    directory = directory if directory is not None else data_dir() / "models"
    if bundled and getattr(sys, "frozen", False) and not any(
        present(directory / asset.name) for asset in ASSETS
    ):
        directory = Path(sys._MEIPASS) / "signspeak-assets"
    return _read(directory, MODEL), _read(directory, METADATA)


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise AssetError("Non-HTTPS asset redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(asset: Asset) -> bytes:
    opener = build_opener(_HTTPSRedirects())
    with opener.open(BASE_URL + asset.name, timeout=20) as response:
        return check_bytes(asset, response.read(asset.size + 1))


def install(directory: Path, *, repair: bool = False):
    missing = []
    for asset in ASSETS:
        path = directory / asset.name
        if present(path):
            try:
                _read(directory, asset)
                continue
            except AssetError:
                if not repair:
                    raise
        missing.append((asset, present(path)))
    if not missing:
        return
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for asset, existed in missing:
            content = _download(asset)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".signspeak-", delete=False) as file:
                staged.append((Path(file.name), directory / asset.name, existed))
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        for source, destination, existed in staged:
            if existed or repair:
                os.replace(source, destination)
            else:
                # Publish atomically without clobbering a concurrent install or an unrelated file.
                try:
                    os.link(source, destination)
                except FileExistsError:
                    _read(directory, next(asset for asset in ASSETS if asset.name == destination.name))
    finally:
        for source, _, _ in staged:
            source.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("install").add_argument("--repair", action="store_true")
    args = parser.parse_args(argv)
    try:
        directory = data_dir() / "models"
        if args.command == "install":
            install(directory, repair=args.repair)
        verified_pair(directory)
    except (AssetError, OSError, URLError) as exc:
        print(f"SignSpeak assets: {exc}", file=sys.stderr)
        return 1
    print(f"Verified SignSpeak 1.0.0-realsign: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
