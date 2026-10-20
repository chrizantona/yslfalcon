#!/usr/bin/env bash
# The service with the trained model, built from this checkout, on a host with an NVIDIA GPU and the container
# toolkit. The runtime image takes src/reid, configs/final_model.yaml and configs/weights.json from the repository
# root - the same files as the submission image - and bakes the weights in (first build: ~8 GB, long).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/init_env.py
docker compose -f docker-compose.yml -f docker-compose.model.yml up --build --detach --wait
printf '\nFalcon with the trained model is running at http://localhost:%s\n' "${FALCON_WEB_PORT:-8090}"
printf 'The interface opens automatically; no login is required.\n'
