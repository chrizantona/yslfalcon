"""Prepare an external ReID set for pretraining: unpack the archive and write train.csv (image_id, vehicle_id, camera_id).

Supported: VERI-Wild test split from the Kaggle mirror mrkdagods/veriwild-test (images/<vehicle>/<image>.jpg plus
vehicle_info.txt with "vehicle/image;camera;time;model;type;color" for all 416,314 VERI-Wild images, no header).
Vehicle ids get a "vw_" prefix so they never collide with ours. Images are ready vehicle crops, read as
<dir>/images/<image_id>.jpg by the training dataset.

Usage: PYTHONPATH=src python -m reid.external --data-config configs/data.yaml --config configs/exp009_....yaml
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

from reid.data import load_yaml


def veriwild_table(archive):
    """train.csv rows for the images present in a VERI-Wild archive; vehicle_info.txt also lists absent images."""
    with zipfile.ZipFile(archive) as z:
        present = {name[len("images/"):-len(".jpg")] for name in z.namelist()
                   if name.startswith("images/") and name.endswith(".jpg")}
        rows = [line.split(";")[:2] for line in z.read("vehicle_info.txt").decode().splitlines() if line]
    table = pd.DataFrame(rows, columns=["image_id", "camera_id"])
    table = table[table.image_id.isin(present)].reset_index(drop=True)
    assert len(table) == len(present), "every image needs a vehicle_info row"
    table["vehicle_id"] = "vw_" + table.image_id.str.split("/").str[0]
    table["camera_id"] = table.camera_id.astype(int)
    return table[["image_id", "vehicle_id", "camera_id"]]


def prepare(archive, out_dir):
    """Unpack images once and (re)write train.csv; returns the table."""
    out_dir = Path(out_dir)
    table = veriwild_table(archive)
    if not (out_dir / "train.csv").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir, [name for name in z.namelist() if name.startswith("images/")])
    table.to_csv(out_dir / "train.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    source = load_yaml(args.data_config)["external"][load_yaml(args.config)["train"]["source"]]
    table = prepare(source["archive"], source["dir"])
    print(f"{len(table)} images, {table.vehicle_id.nunique()} vehicles, {table.camera_id.nunique()} cameras -> {source['dir']}")


if __name__ == "__main__":
    main()
