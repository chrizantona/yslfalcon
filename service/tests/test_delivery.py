"""Offline delivery includes application sources but never local credentials or observations."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "delivery", Path(__file__).parents[1] / "scripts/package_offline.py"
)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def test_bundle_excludes_local_credentials_databases_and_uploaded_media(tmp_path):
    excluded = [
        ".env",
        ".env.production",
        ".local/access-token.txt",
        "falcon.db",
        "falcon.db-wal",
        "local.sqlite-wal",
        "data/photo.jpg",
        "falcon-data/queries/image.png",
        "models/local.json",
        "node_modules/dependency.js",
        "delivery/old/images.tar",
    ]
    included = [
        "backend/app.py",
        "backend/falcon_service/models/__init__.py",
        "backend/falcon_service/models/galleries.py",
        "backend/falcon_service/response_contracts/common.py",
        "runtime/fetch_weights.py",
        ".env.example",
        "frontend/public/fonts/font.ttf",
        "docs/README.md",
    ]
    for relative in excluded + included:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    actual = {str(path.relative_to(tmp_path)) for path in delivery.source_files(tmp_path)}
    assert actual == set(included)


def test_delivery_rejects_an_image_for_the_wrong_host(monkeypatch):
    monkeypatch.setattr(
        delivery.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps([{"Os": "linux", "Architecture": "arm64", "Id": "sha256:test"}]),
    )
    with pytest.raises(ValueError, match="requested linux/amd64"):
        delivery.image_inventory({"api": "falcon:example"}, "linux/amd64")


def test_offline_delivery_pins_architecture_and_disables_external_network():
    override = delivery.compose_override(
        {"api": "falcon:example", "web": "falcon-web:example"}, "linux/amd64"
    )
    assert override["services"]["api"]["platform"] == "linux/amd64"
    assert override["networks"]["default"]["internal"] is True
