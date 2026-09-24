#!/usr/bin/env bash
# Build the two service images and push them, then print the immutable digests to run them by.
#
#   docker login docker.io                       # once, on the machine that builds
#   REGISTRY=docker.io/yourname ./scripts/build_push.sh
#   REGISTRY=ghcr.io/yourname TAG=v1 ./scripts/build_push.sh
#
# Three containers (api, worker, runtime) share the API image; web is nginx over the built frontend. Neither image
# carries weights, secrets or competition data, so the build needs nothing but this checkout and a registry login.
# PLATFORM defaults to the architecture rented machines use; on an Apple Silicon host the build is emulated.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${REGISTRY:?Set REGISTRY, for example docker.io/yourname or ghcr.io/yourname}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M)}"
PLATFORM="${PLATFORM:-linux/amd64}"
API_IMAGE="$REGISTRY/falcon-api:$TAG"
WEB_IMAGE="$REGISTRY/falcon-web:$TAG"
BUILDER="${BUILDER:-falcon-builder}"

docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
  || docker buildx create --name "$BUILDER" --driver docker-container --bootstrap >/dev/null

printf 'building %s for %s\n' "$API_IMAGE" "$PLATFORM"
docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --tag "$API_IMAGE" --push .

printf 'building %s for %s\n' "$WEB_IMAGE" "$PLATFORM"
docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --tag "$WEB_IMAGE" --push \
  --file deploy/web.Dockerfile frontend

digest() { docker buildx imagetools inspect "$1" --format '{{.Manifest.Digest}}'; }
API_REF="$REGISTRY/falcon-api@$(digest "$API_IMAGE")"
WEB_REF="$REGISTRY/falcon-web@$(digest "$WEB_IMAGE")"

mkdir -p .local
umask 077
cat > .local/images.env <<EOF
# Written by scripts/build_push.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ); pinned by digest, not by tag.
FALCON_API_IMAGE=$API_REF
FALCON_WEB_IMAGE=$WEB_REF
EOF

cat <<EOF

pushed:
  api  $API_IMAGE
       $API_REF
  web  $WEB_IMAGE
       $WEB_REF

to run the stack from these images on any host with Docker:
  python3 scripts/init_env.py
  set -a; . .local/images.env; set +a
  docker compose -f docker-compose.yml -f docker-compose.registry.yml up --no-build --pull always --detach --wait
  open http://localhost:8090
EOF
