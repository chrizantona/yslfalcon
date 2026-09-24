#!/usr/bin/env bash
# Build the service images and push them, then print the immutable digests to run them by.
#
#   docker login docker.io                       # once, on the machine that builds
#   REGISTRY=docker.io/yourname ./scripts/build_push.sh
#   REGISTRY=ghcr.io/yourname TAG=v1 ./scripts/build_push.sh
#
# api, worker and the preview runtime share the API image; web is nginx over the built frontend. falcon-runtime is
# the GPU runtime with the trained model baked in (~8 GB: torch with CUDA 12.6 plus 1.7 GB of weights fetched from
# their public source and checked by sha256). SKIP_RUNTIME=1 builds only the two light images. No image carries
# secrets or competition data, so the build needs nothing but this checkout and a registry login.
# PLATFORM defaults to the architecture rented machines use; on an Apple Silicon host the build is emulated.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${REGISTRY:?Set REGISTRY, for example docker.io/yourname or ghcr.io/yourname}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M)}"
PLATFORM="${PLATFORM:-linux/amd64}"
API_IMAGE="$REGISTRY/falcon-api:$TAG"
WEB_IMAGE="$REGISTRY/falcon-web:$TAG"
RUNTIME_IMAGE="$REGISTRY/falcon-runtime:$TAG"
BUILDER="${BUILDER:-falcon-builder}"

docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
  || docker buildx create --name "$BUILDER" --driver docker-container --bootstrap >/dev/null

printf 'building %s for %s\n' "$API_IMAGE" "$PLATFORM"
docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --tag "$API_IMAGE" --push .

printf 'building %s for %s\n' "$WEB_IMAGE" "$PLATFORM"
docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --tag "$WEB_IMAGE" --push \
  --file deploy/web.Dockerfile frontend

if [ "${SKIP_RUNTIME:-0}" != 1 ]; then
  printf 'building %s for %s (the model runtime: long build, large push)\n' "$RUNTIME_IMAGE" "$PLATFORM"
  docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --tag "$RUNTIME_IMAGE" --push \
    --file deploy/runtime.Dockerfile .
fi

digest() { docker buildx imagetools inspect "$1" --format '{{.Manifest.Digest}}'; }
API_REF="$REGISTRY/falcon-api@$(digest "$API_IMAGE")"
WEB_REF="$REGISTRY/falcon-web@$(digest "$WEB_IMAGE")"
RUNTIME_REF=""
[ "${SKIP_RUNTIME:-0}" != 1 ] && RUNTIME_REF="$REGISTRY/falcon-runtime@$(digest "$RUNTIME_IMAGE")"

mkdir -p .local
umask 077
cat > .local/images.env <<EOF
# Written by scripts/build_push.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ); pinned by digest, not by tag.
FALCON_API_IMAGE=$API_REF
FALCON_WEB_IMAGE=$WEB_REF
FALCON_RUNTIME_IMAGE=$RUNTIME_REF
EOF

cat <<EOF

pushed:
  api  $API_IMAGE
       $API_REF
  web  $WEB_IMAGE
       $WEB_REF
  runtime ${RUNTIME_REF:-skipped (SKIP_RUNTIME=1)}

to run the stack from these images on any host with Docker:
  python3 scripts/init_env.py
  set -a; . .local/images.env; set +a
  # with the trained model, on a host with an NVIDIA GPU and the container toolkit:
  docker compose -f docker-compose.yml -f docker-compose.registry.yml -f docker-compose.model.yml \\
    up --no-build --pull always --detach --wait
  # or the preview stack without a GPU:
  docker compose -f docker-compose.yml -f docker-compose.registry.yml up --no-build --pull always --detach --wait
  open http://localhost:8090
EOF
