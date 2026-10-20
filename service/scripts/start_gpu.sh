#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${FALCON_RUNTIME_IMAGE:?Export the GPU runtime image by @sha256 digest}"
: "${FALCON_MODEL_BUNDLE_DIR:?Export the absolute model bundle directory}"
: "${FALCON_EXPECTED_MODEL_ID:?Export the exact model ID from runtime-bundle.json}"

python3 scripts/init_env.py
preflight_report="$(python3 scripts/gpu_preflight.py)"
printf '%s\n' "$preflight_report"
FALCON_EXPECTED_WEIGHTS_BYTES="$(
  python3 -c 'import json, sys; print(json.load(sys.stdin)["weights_bytes"])' <<<"$preflight_report"
)"
export FALCON_EXPECTED_WEIGHTS_BYTES
docker image inspect "$FALCON_RUNTIME_IMAGE" >/dev/null
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up --no-build --pull never --detach --wait

printf '\nFalcon GPU profile is running at http://localhost:8090\n'
printf 'The API accepted the runtime model identity and checked weight inventory.\n'
