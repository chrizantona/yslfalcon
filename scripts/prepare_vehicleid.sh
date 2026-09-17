#!/usr/bin/env bash
# One-off: unpack the encrypted VehicleID archive for reid.external (source vehicleid) with the official static 7-Zip.
# The archive and its password come from the dataset authors on request (see DATA.md); keep the password in a file
# outside git with mode 600 and delete it afterwards. 7-Zip reads it from its command line inside this container.
# Usage on the pod: PASSWORD_FILE=/dev/shm/vehicleid_zip_password bash scripts/prepare_vehicleid.sh
set -Eeuo pipefail
: "${PASSWORD_FILE:?set PASSWORD_FILE}"
ARCHIVE=${ARCHIVE:-/workspace/ext_data/vehicleid/VehicleID_V1.0.zip}
OUT=${OUT:-/dev/shm/ext/vehicleid}
SEVEN_ZIP=https://github.com/ip7z/7zip/releases/download/26.03/7z2603-linux-x64.tar.xz
TOOLS=$(mktemp -d)
trap 'rm -rf "$TOOLS"' EXIT

echo "75a3c8ba0b1cfb4ac4a088c4d0c3c0fceb0ae45abc17a15513e9915531739952  $ARCHIVE" | sha256sum -c --quiet
curl -sSfL --retry 5 -o "$TOOLS/7z.tar.xz" "$SEVEN_ZIP"
echo "dc99eff5008f1ab79bd7084c68513701547a808a89502bf4133683535ab3c695  $TOOLS/7z.tar.xz" | sha256sum -c --quiet
python3 -c 'import sys, tarfile; tarfile.open(sys.argv[1]).extract("7zzs", sys.argv[2], filter="data")' "$TOOLS/7z.tar.xz" "$TOOLS"

mkdir -p "$OUT/images"
"$TOOLS/7zzs" e -y -bso0 -bsp0 -p"$(cat "$PASSWORD_FILE")" -o"$OUT" "$ARCHIVE" VehicleID_V1.0/attribute/img2vid.txt
"$TOOLS/7zzs" e -y -bso0 -bsp0 -p"$(cat "$PASSWORD_FILE")" -o"$OUT/images" "$ARCHIVE" 'VehicleID_V1.0/image/*.jpg'
echo "$(find "$OUT/images" -name '*.jpg' | wc -l) images, $(wc -l < "$OUT/img2vid.txt") labels -> $OUT"
