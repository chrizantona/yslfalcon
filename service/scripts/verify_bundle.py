"""Verify file integrity before loading an offline delivery."""

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f"Missing or invalid bundle path: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise SystemExit(f"Integrity check failed: {relative}")
    print(f"Verified {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
