"""Build a local offline bundle with image checksums, excluding secrets and user data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".local",
    "node_modules",
    "data",
    "falcon-data",
    "models",
    "delivery",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "test-results",
    "playwright-report",
    "coverage",
}


def source_files(root: Path):
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [
            name
            for name in directories
            if name not in EXCLUDED_DIRS
            and not name.endswith(".egg-info")
            and not (Path(directory) / name).is_symlink()
        ]
        for name in filenames:
            path = Path(directory) / name
            if path.is_symlink() or name in {".coverage", ".DS_Store"}:
                continue
            if name.startswith(".env") and name != ".env.example":
                continue
            if path.suffix in {".db", ".log", ".pyc"} or ".db-" in name or ".sqlite" in name:
                continue
            yield path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_inventory(images: dict[str, str], platform: str) -> dict[str, dict[str, str]]:
    """Reject mixed/wrong architectures before producing a delivery labelled for a host."""
    inventory = {}
    for name in sorted(set(images.values())):
        details = json.loads(subprocess.check_output(["docker", "image", "inspect", name], text=True))[0]
        actual = f"{details['Os']}/{details['Architecture']}"
        if actual != platform:
            raise ValueError(
                f"Image {name} has platform {actual}; requested {platform}. Rebuild/pull it first."
            )
        inventory[name] = {"id": details["Id"], "platform": actual}
    return inventory


def compose_override(images: dict[str, str], platform: str) -> dict:
    services = {name: {"image": value, "platform": platform} for name, value in images.items()}
    # Docker does not publish host ports for an internal-only container. Only the
    # fixed Nginx gateway joins ingress; API, runtime, workers and DB stay isolated.
    services["web"]["networks"] = ["default", "ingress"]
    return {"services": services, "networks": {"default": {"internal": True}, "ingress": {}}}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "delivery/falcon-offline")
    parser.add_argument("--platform", choices=("linux/amd64", "linux/arm64"), default="linux/amd64")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error("Output already exists; choose a new directory to preserve the previous bundle")
    configuration = json.loads(
        subprocess.check_output(
            ["docker", "compose", "config", "--format", "json"],
            cwd=root,
            text=True,
        )
    )
    images = {name: value["image"] for name, value in configuration["services"].items()}
    # Keep only image names from Compose; its expanded environment contains secrets.
    del configuration
    try:
        inventory = image_inventory(images, args.platform)
    except ValueError as exc:
        parser.error(str(exc))
    files = list(source_files(root))
    target.mkdir(parents=True)
    source = target / "source"
    for path in files:
        destination = source / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    subprocess.run(
        [
            "docker",
            "save",
            "--platform",
            args.platform,
            "--output",
            str(target / "images.tar"),
            *sorted(set(images.values())),
        ],
        check=True,
    )
    override = compose_override(images, args.platform)
    (target / "images.compose.json").write_text(json.dumps(override, indent=2) + "\n")
    start = target / "start.sh"
    start.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
        "python3 source/scripts/verify_bundle.py .\n"
        "docker load --input images.tar\n"
        "cd source\npython3 scripts/init_env.py\n"
        "docker compose -f docker-compose.yml -f ../images.compose.json up --no-build --pull never --detach --wait\n"
    )
    start.chmod(0o755)
    hashes = {str(path.relative_to(target)): digest(path) for path in target.rglob("*") if path.is_file()}
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "platform": args.platform,
                "network": "internal_backend_with_web_ingress",
                "images": images,
                "image_inventory": inventory,
                "files": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Offline bundle created: {target}")
    print("Contains the service stub only. No credentials, uploaded images or trained weights included.")


if __name__ == "__main__":
    main()
