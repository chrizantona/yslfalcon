"""Offline delivery includes application sources but never local credentials or observations."""

import importlib.util
from pathlib import Path

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
    included = ["backend/app.py", ".env.example", "frontend/public/fonts/font.ttf", "docs/README.md"]
    for relative in excluded + included:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    actual = {str(path.relative_to(tmp_path)) for path in delivery.source_files(tmp_path)}
    assert actual == set(included)
