import hashlib
from urllib.request import Request

import pytest

from utils import model_assets


@pytest.fixture
def pair(monkeypatch):
    files = {"alphabet.bin": b"model bytes", "alphabet.json": b"{}"}
    assets = tuple(model_assets.Asset(name, len(data), hashlib.sha256(data).hexdigest())
                   for name, data in files.items())
    monkeypatch.setattr(model_assets, "ASSETS", assets)
    monkeypatch.setattr(model_assets, "MODEL", assets[0])
    monkeypatch.setattr(model_assets, "METADATA", assets[1])
    monkeypatch.setattr(model_assets, "_download", lambda asset: files[asset.name])
    return files


def test_second_download_failure_does_not_publish_first(tmp_path, pair, monkeypatch):
    def fail_second(asset):
        if asset == model_assets.METADATA:
            raise model_assets.AssetError("download failed")
        return pair[asset.name]

    monkeypatch.setattr(model_assets, "_download", fail_second)
    with pytest.raises(model_assets.AssetError, match="download failed"):
        model_assets.install(tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_mismatch_preserved_until_explicit_repair(tmp_path, pair):
    model_path = tmp_path / model_assets.MODEL.name
    model_path.write_bytes(b"not the pinned file")
    with pytest.raises(model_assets.AssetError, match="Unverified"):
        model_assets.install(tmp_path)
    assert model_path.read_bytes() == b"not the pinned file"
    assert not (tmp_path / model_assets.METADATA.name).exists()
    model_assets.install(tmp_path, repair=True)
    assert model_assets.verified_pair(tmp_path) == tuple(pair.values())
    assert {file.name for file in tmp_path.iterdir()} == pair.keys()


def test_downgrade_redirect_denied():
    redirects = model_assets._HTTPSRedirects()
    with pytest.raises(model_assets.AssetError, match="Non-HTTPS"):
        redirects.redirect_request(Request("https://example.com"), None, 302, "", {}, "http://example.com")


def test_bundle_used_only_when_both_local_files_are_absent(tmp_path, pair, monkeypatch):
    local = tmp_path / "models"
    bundle = tmp_path / "bundle"
    installed = bundle / "signspeak-assets"
    model_assets.install(installed)
    monkeypatch.setattr(model_assets.sys, "frozen", True, raising=False)
    monkeypatch.setattr(model_assets.sys, "_MEIPASS", str(bundle), raising=False)
    assert model_assets.verified_pair(local, bundled=True) == tuple(pair.values())
    local.mkdir()
    (local / model_assets.MODEL.name).write_bytes(pair[model_assets.MODEL.name])
    with pytest.raises(model_assets.AssetError, match=model_assets.METADATA.name):
        model_assets.verified_pair(local, bundled=True)
