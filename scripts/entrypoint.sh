#!/usr/bin/env bash
# The image serves three jobs: build the submission (default), prove itself on a rented GPU, or hand over a shell.
set -euo pipefail
case "${1:-submit}" in
  submit)     shift || true
              exec python -m reid.submit --data-config configs/data_docker.yaml \
                   --config "${1:-configs/final_model.yaml}" ;;
  selfcheck)  shift || true
              exec python -m reid.selfcheck "$@" ;;
  *)          exec "$@" ;;
esac
