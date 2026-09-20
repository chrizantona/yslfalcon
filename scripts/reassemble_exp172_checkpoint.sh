#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parts_dir="$repo_root/artifacts/EXP-172/checkpoints"
output="$parts_dir/63bd5755271548778ed71b5f688f28dc_seed0.pt"
expected_size=4735720473
expected_sha=2997b65c29f73b21a00d056bf6d8dccf90f8a553e633ea31a68f0a70b5b8d0b1

cat "$output".part-* > "$output.tmp"
actual_size="$(wc -c < "$output.tmp" | tr -d ' ')"
actual_sha="$(shasum -a 256 "$output.tmp" | awk '{print $1}')"

if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
  rm -f "$output.tmp"
  echo "checkpoint verification failed: size=$actual_size sha256=$actual_sha" >&2
  exit 1
fi

mv "$output.tmp" "$output"
echo "$output"
echo "size=$actual_size sha256=$actual_sha"
