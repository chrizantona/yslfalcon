#!/usr/bin/env bash
# Restore the full unpacked dataset after the archive transfer; keep verified staging files intact.
set -Eeuo pipefail
ARCHIVE=/workspace/lct2026_data/dataset.zip
while [ ! -f "$ARCHIVE" ]; do sleep 10; done
python3 - "$ARCHIVE" <<'PY'
import hashlib,sys,zipfile
from pathlib import Path
archive=Path(sys.argv[1]); target=Path('/dev/shm/lct2026/dataset')
with archive.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for entry in z.infolist():
        if not (target/entry.filename).exists():
            z.extract(entry,target)
print('Full competition dataset verified and unpacked.',flush=True)
PY
touch /workspace/lct2026_data/DONE
