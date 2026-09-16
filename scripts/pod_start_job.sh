#!/usr/bin/env bash
# Start only after the complete, verified staging set, passing tests, and pushed experiment tags exist.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
trap 'echo "start checks failed ($(date -Is))" > status/start_FAILED' ERR
echo $$ > status/start_pid
echo wait_environment > status/start_stage
while [ ! -f status/env_DONE ]; do
  if [ -f status/env_FAILED ]; then cat status/env_FAILED; exit 1; fi
  sleep 5
done
echo wait_data > status/start_stage
python3 - <<'PY'
import hashlib,time
from pathlib import Path
root=Path('/dev/shm/lct2026')
files=[line.split('  ',1) for line in Path('SHA256SUMS').read_text().splitlines()]
while not all((root/name).is_file() for _,name in files):
    time.sleep(5)
for expected,name in files:
    with (root/name).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==expected, f'checksum mismatch: {name}'
Path('status/data_DONE').touch()
print('Verified staging data:',len(files),'files',flush=True)
PY
echo wait_pushed_tags > status/start_stage
while [ ! -f status/CODE_PUSHED ]; do sleep 5; done
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader) PYTHONPATH="$PWD/src"
echo preflight > status/start_stage
.venv/bin/python scripts/pod_preflight.py > logs/preflight.log 2>&1
echo night_running > status/start_stage
bash scripts/pod_night_job.sh
echo done > status/start_stage
touch status/start_DONE
