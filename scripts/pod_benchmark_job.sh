#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG}" "${GPU:?set GPU}"
source scripts/pod_setup.sh
stage benchmark
ARGS=(--data-config "$DATA_CFG" --config "$CONFIG")
if [ -n "${CHECKPOINT:-}" ]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
$PY -m reid.benchmark "${ARGS[@]}" > logs/benchmark.log 2>&1
stage done
touch status/DONE
