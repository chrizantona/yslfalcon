#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET=""
OUTPUT="$ROOT/benchmarks/a5000-selectel-2026-09-24/output"
IMAGE="yslfalcon:a5000-benchmark"
BATCH_SIZES="1 8 16 32"
WARMUP_BATCHES=1
MEASURED_BATCHES=5
WORKERS=8
BUILD=1
FULL_INFERENCE=1

usage() {
  cat <<'EOF'
Usage: ./benchmarks/a5000-selectel-2026-09-24/run.sh --dataset /absolute/path [options]

Options:
  --output DIR             Result directory (default: benchmark output directory)
  --image IMAGE            Existing image with final weights; skips docker build
  --batch-sizes "1 8 16"   Batch sizes (default: "1 8 16 32")
  --warmup-batches N       Excluded warm-up batches per size (default: 1)
  --measured-batches N     Timed batches per size (default: 5)
  --workers N              JPEG/crop worker threads (default: 8)
  --skip-full-inference    Only collect short batch measurements
EOF
}

while (($#)); do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --image) IMAGE="$2"; BUILD=0; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --warmup-batches) WARMUP_BATCHES="$2"; shift 2 ;;
    --measured-batches) MEASURED_BATCHES="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --skip-full-inference) FULL_INFERENCE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATASET" ]] || { usage >&2; exit 2; }
DATASET="$(cd "$DATASET" && pwd)"
OUTPUT="$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd)"
chmod 0777 "$OUTPUT"
for file in images test_query.csv test_gallery.csv; do
  [[ -e "$DATASET/$file" ]] || { echo "Missing $DATASET/$file" >&2; exit 2; }
done

if ((BUILD)); then
  docker build --tag "$IMAGE" "$ROOT"
fi

docker run --rm --gpus all --shm-size 1g \
  -v "$DATASET:/data:ro" \
  -v "$OUTPUT:/results" \
  -v "$ROOT/benchmarks/a5000-selectel-2026-09-24/collect_batch_metrics.py:/benchmark.py:ro" \
  --entrypoint python "$IMAGE" /benchmark.py \
  --data-config configs/data_docker.yaml \
  --config configs/final_model.yaml \
  --checkpoint /opt/checkpoints/6a1c8e0b53f74d9fae2c7b81d40936af.pt \
  --batch-sizes $BATCH_SIZES \
  --warmup-batches "$WARMUP_BATCHES" \
  --measured-batches "$MEASURED_BATCHES" \
  --workers "$WORKERS" \
  --out /results/batch_metrics.json

if ((FULL_INFERENCE)); then
  start="$(date +%s)"
  docker run --rm --gpus all --shm-size 1g \
    -v "$DATASET:/data:ro" \
    -v "$OUTPUT:/app/artifacts" \
    "$IMAGE" submit configs/final_model.yaml
  end="$(date +%s)"
  printf '{\n  "wall_seconds": %s,\n  "query_rows": %s,\n  "gallery_rows": %s\n}\n' \
    "$((end - start))" "$(( $(wc -l < "$DATASET/test_query.csv") - 1 ))" \
    "$(( $(wc -l < "$DATASET/test_gallery.csv") - 1 ))" > "$OUTPUT/full_inference.json"
  docker run --rm --gpus all \
    -v "$DATASET:/data:ro" \
    -v "$OUTPUT:/app/artifacts" \
    --entrypoint python "$IMAGE" -m reid.check_submission \
    --data-config configs/data_docker.yaml --config configs/final_model.yaml \
    | tee "$OUTPUT/submission_check.txt"
fi

echo "Results: $OUTPUT"
