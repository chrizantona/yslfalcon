"""Prepare the external pretraining set: unpack the archive and write train.csv (image_id, vehicle_id, camera_id).

The one external set of the submission is the VERI-Wild test split from the Kaggle mirror mrkdagods/veriwild-test
(images/<vehicle>/<image>.jpg plus vehicle_info.txt with "vehicle/image;camera;time;model;type;color" for all 416,314
VERI-Wild images, no header). Vehicle ids get a "vw_" prefix, which keeps them apart from ours. Images are ready
vehicle crops, read as <dir>/images/<image_id>.jpg by the training dataset. Trimmed copy of reid.external on the
`experiments` branch, without the sets of rejected experiments.

Usage: PYTHONPATH=src python -m reid.external --data-config configs/data.yaml \
           --config configs/train/stage1_pretrain_veriwild.yaml
"""
import argparse
import shutil
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
    """Unpack the images once and (re)write the training table; returns the table."""
    out_dir = Path(out_dir)
    table = veriwild_table(archive)
    if not (out_dir / "train.csv").exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith("images/") and not name.endswith("/"):
                    (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as src, (out_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    table.to_csv(out_dir / "train.csv", index=False)
    return table


def load_table(out_dir):
    """The prepared table; ids stay strings."""
    return pd.read_csv(Path(out_dir) / "train.csv", dtype={"image_id": str, "vehicle_id": str})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    name = load_yaml(args.config)["train"]["source"]
    spec = load_yaml(args.data_config)["external"][name]
    table = prepare(spec["archive"], spec["dir"])
    print(f"{name}: {len(table)} images, {table.vehicle_id.nunique()} vehicles, "
          f"{table.camera_id.nunique()} cameras -> {spec['dir']}/train.csv")


if __name__ == "__main__":
    main()
