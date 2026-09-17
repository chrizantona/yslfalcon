"""Download the pinned trained checkpoint at image build time; credentials come from a BuildKit secret."""
import hashlib
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).hostname != urllib.parse.urlparse(newurl).hostname:
            redirected.remove_header("Authorization")
        return redirected


def main():
    url, digest = os.environ["WEIGHTS_URL"], os.environ["WEIGHTS_SHA256"]
    if not url.startswith("https://") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("provide a HTTPS checkpoint URL and its sha256")
    headers = {"Accept": "application/octet-stream", "User-Agent": "yslfalcon"}
    secret = Path("/run/secrets/github_token")
    if secret.exists():
        if urllib.parse.urlparse(url).hostname != "api.github.com":
            raise ValueError("GitHub credentials may only be sent to api.github.com")
        headers["Authorization"] = "Bearer " + secret.read_text().strip()
    name = os.environ.get("CHECKPOINT_NAME", "exp024.pt")
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("checkpoint name must be a .pt basename")
    target = Path("/opt/checkpoints") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    checksum = hashlib.sha256()
    opener = urllib.request.build_opener(SafeRedirect())
    cache = Path("/var/cache/reid") / (digest + ".pt")
    try:
        source = cache.open("rb") if cache.exists() else opener.open(urllib.request.Request(url, headers=headers), timeout=120)
        with source as response, temporary.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                checksum.update(chunk)
                out.write(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("trained checkpoint checksum mismatch")
        temporary.replace(target)
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            cached_download = cache.with_suffix(".download")
            shutil.copyfile(target, cached_download)
            cached_download.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
