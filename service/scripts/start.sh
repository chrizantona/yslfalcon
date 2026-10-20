#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/init_env.py
docker compose up --build --detach --wait
printf '\nFalcon is running at http://localhost:8090\n'
printf 'The interface opens automatically; no login is required.\n'
