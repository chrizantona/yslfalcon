"""Store every training box as a JPEG crop (longer side <= crop_max_side) so training doesn't decode full frames.

Only train.csv is cropped: the test files never enter training. Trimmed copy of reid.crops on the `experiments` branch.

Usage: PYTHONPATH=src python -m reid.crops --data-config configs/data.yaml
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from reid.data import crop_box, load_csv, load_yaml


def save_crop(job):
    image_path, box, out_path, max_side, quality = job
    if os.path.exists(out_path):
        return
    crop = crop_box(cv2.imread(image_path), box, max_side)
    tmp_path = out_path[: -len(".jpg")] + ".tmp.jpg"
    cv2.imwrite(tmp_path, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    os.replace(tmp_path, out_path)  # no half-written crops if the job is killed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    cfg = load_yaml(parser.parse_args().data_config)

    dataset_dir, out = Path(cfg["dataset_dir"]), Path(cfg["crops_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(dataset_dir, "train")
    jobs = [(str(dataset_dir / "images" / f"{r.image_id}.jpg"), (r.x, r.y, r.w, r.h), str(out / f"{r.image_id}.jpg"),
             cfg["crop_max_side"], cfg["crop_jpeg_quality"]) for r in df.itertuples()]

    cv2.setNumThreads(1)
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as pool:
        for done, _ in enumerate(pool.map(save_crop, jobs, chunksize=64), 1):
            if done % 2000 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} crops", flush=True)


if __name__ == "__main__":
    main()
