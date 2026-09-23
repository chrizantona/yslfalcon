"""Download the pinned weights at image build time: public files, verified by sha256, no credentials.

configs/weights.json names every file the submission needs (the encoder and the viewpoint head) with its digest;
nothing else is accepted, and a file whose checksum does not match fails the build. WEIGHTS_BASE_URL points the
download at a mirror without touching the digests. The BuildKit cache at /var/cache/reid keeps the 1.7 GB encoder
between rebuilds, so only the first build pays for it.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

CHUNK = 8 * 1024 * 1024
CACHE = Path(os.environ.get("WEIGHTS_CACHE", "/var/cache/reid"))  # the BuildKit cache mount, or a scratch dir in tests


def digest_of(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            checksum.update(chunk)
    return checksum.hexdigest()


def fetch(url, digest, target):
    """Write url to target, refusing anything whose sha256 is not digest."""
    checksum = hashlib.sha256()
    temporary, cached = target.parent / (target.name + ".download"), CACHE / digest
    source = cached.open("rb") if cached.exists() else urlopen(
        Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "yslfalcon"}), timeout=120)
    try:
        with source as response, temporary.open("wb") as out:
            while chunk := response.read(CHUNK):
                checksum.update(chunk)
                out.write(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError(f"checksum mismatch for {target.name}")
        temporary.replace(target)
        if not cached.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            partial = CACHE / (digest + ".download")
            shutil.copyfile(target, partial)
            partial.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    manifest = json.loads(Path(os.environ.get("WEIGHTS_MANIFEST", "configs/weights.json")).read_text())
    base = os.environ.get("WEIGHTS_BASE_URL", manifest["base_url"]).rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("the weights must come over HTTPS")
    target_dir = Path(os.environ.get("CHECKPOINT_DIR", "/opt/checkpoints"))
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError(f"{name} must be a plain file name")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name} has no sha256 in the manifest")
        target = target_dir / name
        if target.exists() and digest_of(target) == digest:
            print(f"{name}: already in place", flush=True)
            continue
        fetch(f"{base}/{name}", digest, target)
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB, sha256 verified", flush=True)


if __name__ == "__main__":
    main()
