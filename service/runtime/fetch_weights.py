"""Bake the trained model into the runtime image: download, verify, and describe it as a runtime bundle.

weights.json (a copy of main:configs/weights.json) names every file with its sha256; a file that does not
match fails the build. The result is /models with the two files and runtime-bundle.json in the service's
bundle schema, so the runtime reads one manifest whether the bundle was baked in here or mounted from the
host.

A baked bundle cannot name its own image digest - the digest exists only after the build - so runtime_image
says where the files live instead, and the adapter re-checks every size and digest at startup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

CHUNK = 8 * 1024 * 1024
MODEL_ID = "emb-so400m-224-v1"


def fetch(url, digest, target):
    checksum, temporary = hashlib.sha256(), target.parent / (target.name + ".download")
    try:
        with (
            urlopen(Request(url, headers={"User-Agent": "falcon-runtime"}), timeout=120) as response,
            temporary.open("wb") as out,
        ):
            while chunk := response.read(CHUNK):
                checksum.update(chunk)
                out.write(chunk)
        if checksum.hexdigest() != digest:
            raise SystemExit(f"checksum mismatch for {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("weights.json")))
    parser.add_argument("--out", default="/models")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    base = os.environ.get("WEIGHTS_BASE_URL", manifest["base_url"]).rstrip("/")
    if not base.startswith("https://"):
        raise SystemExit("the weights must come over HTTPS")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, digest in manifest["files"].items():
        if Path(name).name != name or len(digest) != 64:
            raise SystemExit(f"bad manifest entry {name}")
        fetch(f"{base}/{name}", digest, out / name)
        size = (out / name).stat().st_size
        files.append({"path": name, "role": "weight", "bytes": size, "sha256": digest})
        print(f"{name}: {size} bytes, sha256 verified", flush=True)
    bundle = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "runtime_image": "baked:/models",
        "total_weights_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (out / "runtime-bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"runtime-bundle.json: {bundle['total_weights_bytes']} weight bytes", flush=True)


if __name__ == "__main__":
    main()
