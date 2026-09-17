"""Prepare external ReID sets for pretraining: unpack the archive and write train.csv (image_id, vehicle_id, camera_id).

Supported:
- veriwild_test: VERI-Wild test split from the Kaggle mirror mrkdagods/veriwild-test (images/<vehicle>/<image>.jpg plus
  vehicle_info.txt with "vehicle/image;camera;time;model;type;color" for all 416,314 VERI-Wild images, no header).
  Vehicle ids get a "vw_" prefix.
- veri776: VeRi-776 from the Kaggle mirror abhyudaya12/veri-vehicle-re-identification-dataset (VeRi/image_train and
  VeRi/image_test, files <vehicle>_c<camera>_<time>_<n>.jpg). image_query repeats vehicles of image_test and is
  skipped. Vehicle ids get a "v776_" prefix.
- vehicleid: VehicleID (PKU, given by the authors on request). The archive is encrypted, so it is unpacked beforehand by
  scripts/prepare_vehicleid.sh into <dir>/images/<image>.jpg and <dir>/img2vid.txt ("image vehicle" per line); images
  without a vehicle label are skipped. Vehicle ids get a "vid_" prefix. There are no camera labels: every image counts
  as its own camera, so the PK sampler picks random frames of a vehicle.
The prefixes keep the vehicles of different sets, and ours, apart. Images are ready vehicle crops, read as
<dir>/images/<image_id>.jpg by the training dataset.

Usage: PYTHONPATH=src python -m reid.external --data-config configs/data.yaml --config configs/exp009_....yaml
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from reid.data import load_yaml

VERI776_SPLITS = ("image_train", "image_test")


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


def veriwild_members(names):
    """{archive member: path under the output dir} of the images to unpack."""
    return {name: name for name in names if name.startswith("images/") and not name.endswith("/")}


def veri776_members(names):
    members = {}
    for name in names:
        parts = name.split("/")
        if name.endswith(".jpg") and len(parts) >= 2 and parts[-2] in VERI776_SPLITS:
            members[name] = f"images/{parts[-2]}/{parts[-1]}"
    return members


def veri776_table(archive):
    """train.csv rows for the train and test images of a VeRi-776 archive; vehicle and camera from the file names."""
    with zipfile.ZipFile(archive) as z:
        targets = sorted(veri776_members(z.namelist()).values())
    image_ids = pd.Series([target[len("images/"):-len(".jpg")] for target in targets])
    fields = image_ids.str.split("/").str[1].str.split("_")  # <vehicle>_c<camera>_<time>_<n>
    return pd.DataFrame({"image_id": image_ids, "vehicle_id": "v776_" + fields.str[0],
                         "camera_id": fields.str[1].str[1:].astype(int)})


SOURCES = {"veriwild_test": (veriwild_table, veriwild_members), "veri776": (veri776_table, veri776_members)}


def vehicleid_table(out_dir):
    """train.csv rows for VehicleID already unpacked by scripts/prepare_vehicleid.sh."""
    out_dir = Path(out_dir)
    if not (out_dir / "img2vid.txt").exists():
        raise FileNotFoundError(f"{out_dir}: the VehicleID archive is encrypted, unpack it with scripts/prepare_vehicleid.sh")
    table = pd.read_csv(out_dir / "img2vid.txt", sep=r"\s+", header=None, names=["image_id", "vehicle"], dtype=str)
    present = {path.stem for path in (out_dir / "images").glob("*.jpg")}
    table = table[table.image_id.isin(present)].sort_values("image_id").reset_index(drop=True)
    table["vehicle_id"] = "vid_" + table.vehicle
    table["camera_id"] = range(len(table))
    return table[["image_id", "vehicle_id", "camera_id"]]


def prepare(archive, out_dir, kind="veriwild_test"):
    """Unpack the images once and (re)write train.csv; returns the table."""
    out_dir = Path(out_dir)
    if kind == "vehicleid":
        table = vehicleid_table(out_dir)
        table.to_csv(out_dir / "train.csv", index=False)
        return table
    make_table, members = SOURCES[kind]
    table = make_table(archive)
    if not (out_dir / "train.csv").exists():
        with zipfile.ZipFile(archive) as z:
            for name, target in members(z.namelist()).items():
                (out_dir / target).parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, (out_dir / target).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    table.to_csv(out_dir / "train.csv", index=False)
    return table


def source_names(train_cfg):
    """train.source names one external set or a list of them."""
    source = train_cfg["source"]
    return [source] if isinstance(source, str) else list(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    external = load_yaml(args.data_config)["external"]
    for name in source_names(load_yaml(args.config)["train"]):
        table = prepare(external[name]["archive"], external[name]["dir"], name)
        print(f"{name}: {len(table)} images, {table.vehicle_id.nunique()} vehicles, "
              f"{table.camera_id.nunique()} cameras -> {external[name]['dir']}")


if __name__ == "__main__":
    main()
