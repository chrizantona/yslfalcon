"""Upload verified local handoff files to this project's private GitHub Release.

Uses the existing Git credential helper, never stores or prints credentials.
Re-runs skip assets only when GitHub's SHA256 matches the local manifest.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode


REPO = "chrizantona/yslfalcon"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("artifacts/handoff"))
    parser.add_argument("--tag", default="exp-024")
    args = parser.parse_args()
    result = subprocess.run(["git", "credential", "fill"],
                            input=f"protocol=https\nhost=github.com\npath={REPO}.git\n\n",
                            text=True, capture_output=True, check=True)
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = credentials["password"]

    def request(method, route, body=None, *, host="api.github.com", content_type="application/json", length=None):
        if host not in ("api.github.com", "uploads.github.com"):
            raise ValueError("unexpected GitHub hostname")
        connection = http.client.HTTPSConnection(host, timeout=600, blocksize=8 * 1024 * 1024)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "yslfalcon-handoff",
                   "Content-Type": content_type}
        if length is not None:
            headers["Content-Length"] = str(length)
        try:
            connection.request(method, route, body=body, headers=headers)
            response = connection.getresponse()
            data = json.loads(response.read() or b"{}")
            if response.status >= 400:
                raise RuntimeError(f"GitHub {method} {route}: {response.status}, {data.get('message', '')}")
            return data
        finally:
            connection.close()

    repository = request("GET", f"/repos/{REPO}")
    if not repository["private"] or not repository["permissions"]["push"]:
        raise ValueError("expected a private repository with push access")
    manifest = json.loads((args.directory / "manifest.json").read_text())
    releases = request("GET", f"/repos/{REPO}/releases?per_page=100")
    release = next((r for r in releases if r["tag_name"] == args.tag), None)
    if release is None:
        body = {"tag_name": args.tag, "name": "EXP-022…024: validation artifacts and checkpoints",
                "draft": True, "prerelease": True,
                "body": "Verified AP per query, validation embeddings, logs and source snapshots for EXP-022…024. "
                        "Weights: exp022_seed0.pt (pretraining), exp024.pt (current quality baseline). "
                        "Check SHA256SUMS before use. val_v2, development fold 0; holdout fold 4 untouched. "
                        "H200 performance only; RTX A5000 pending. See README.md and DATA.md."}
        release = request("POST", f"/repos/{REPO}/releases", json.dumps(body).encode())
    assets = request("GET", f"/repos/{REPO}/releases/{release['id']}/assets?per_page=100")
    existing = {asset["name"]: asset for asset in assets}
    uploaded = []
    for name in [*manifest, "SHA256SUMS", "manifest.json"]:
        path = args.directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name in manifest:
            meta = manifest[name]
            if digest != meta["sha256"] or path.stat().st_size != meta["bytes"]:
                raise ValueError(f"manifest mismatch: {name}")
        if name in existing:
            asset = existing[name]
            if asset.get("digest") != f"sha256:{digest}":
                raise ValueError(f"existing Release asset differs: {name}")
            print(f"verified existing {name}", flush=True)
        else:
            print(f"uploading {name} ({path.stat().st_size} bytes)", flush=True)
            with path.open("rb") as stream:
                asset = request("POST", f"/repos/{REPO}/releases/{release['id']}/assets?" + urlencode({"name": name}),
                                stream, host="uploads.github.com", content_type="application/octet-stream",
                                length=path.stat().st_size)
            if asset.get("digest") != f"sha256:{digest}":
                raise ValueError(f"GitHub digest mismatch: {name}")
        uploaded.append({"name": name, "sha256": digest, "asset_api_url": asset["url"],
                         "download_url": asset["browser_download_url"], "bytes": asset["size"]})
        print(f"verified {name}", flush=True)
    if release["draft"]:
        release = request("PATCH", f"/repos/{REPO}/releases/{release['id']}", json.dumps({"draft": False}).encode())
    receipt = {"release_url": release["html_url"], "release_id": release["id"], "tag": args.tag, "assets": uploaded}
    (args.directory / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(release["html_url"], flush=True)


if __name__ == "__main__":
    main()
