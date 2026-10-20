"""Generate a private local deployment configuration without printing credentials."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path


def initialize(directory: Path, port: int = 8090) -> bool:
    env_path = directory / ".env"
    if env_path.exists():
        return False
    access_token = secrets.token_urlsafe(32)
    values = {
        "FALCON_DB_PASSWORD": secrets.token_urlsafe(32),
        "FALCON_AUTH_TOKENS": json.dumps(
            {access_token: {"role": "admin", "workspace": "local"}}, separators=(",", ":")
        ),
        "FALCON_RUNTIME_TOKEN": secrets.token_urlsafe(32),
        "FALCON_IMAGE_TAG": "local-38c6da2",
        "FALCON_WEB_PORT": str(port),
        "FALCON_ALLOWED_ORIGINS": json.dumps(
            [f"http://localhost:{port}", f"http://127.0.0.1:{port}"], separators=(",", ":")
        ),
        "FALCON_COOKIE_SECURE": "false",
        "FALCON_LOCAL_AUTO_LOGIN": "true",
    }
    local_dir = directory / ".local"
    local_dir.mkdir(mode=0o700, exist_ok=True)
    access_path = local_dir / "access-token.txt"
    descriptor = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write("# Generated locally. Contains secrets. Do not commit.\n")
        for name, value in values.items():
            stream.write(f"{name}='{value}'\n")
    descriptor = os.open(access_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(access_token + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    created = initialize(args.directory.resolve(), args.port)
    print("Local configuration created." if created else "Existing configuration preserved.")
    print(f"API integration token file: {args.directory / '.local' / 'access-token.txt'}")


if __name__ == "__main__":
    main()
