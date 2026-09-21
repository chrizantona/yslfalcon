#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/init_env.py
docker compose up --build --detach --wait
printf '\nFalcon is running at http://localhost:8090\n'
printf 'The local login token is in service/.local/access-token.txt\n'
