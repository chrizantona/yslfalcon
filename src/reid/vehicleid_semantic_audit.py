"""Find possible identity-level VehicleID/LCT overlap with a frozen visual tower.

Exact hashes catch copied frames, but not another photograph of the same vehicle.  This second audit embeds every
official VehicleID-train image and every LCT crop, records cross-dataset nearest neighbours, and highlights mutual
matches with multi-frame support.  Its output is candidate evidence for visual review, never a training feature.

Embeddings are written incrementally as float16 memmaps on persistent storage.  Only one image batch and the compact
embedding matrices occupy memory; the source images are never copied into RAM or /dev/shm.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid.data import CSV_FILES, load_csv, load_yaml
from reid.external import load_table
from reid.models import build_backbone
from reid.train import CropDataset, _worker_init


def plain_recipe(train):
    return {**train, "flip_p": 0, "pad": 0, "erasing_p": 0, "color_jitter": None}


@torch.inference_mode()
def embed_to_memmap(encoder, table, input_hw, mean, std, recipe, path, dim, device, batch_size, workers):
    dataset = CropDataset(table, None, input_hw, mean, std, recipe)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                        worker_init_fn=_worker_init)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(table), dim))
    start = 0
    for batch, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            features = F.normalize(encoder(images).float(), dim=1)
        end = start + len(features)
        output[start:end] = features.cpu().numpy().astype(np.float16)
        start = end
        if batch % 100 == 0 or end == len(table):
            print(json.dumps({"stage": f"embed_{Path(path).stem}", "images": end, "total": len(table)}),
                  flush=True)
    output.flush()
    return path


@torch.inference_mode()
def cross_neighbours(external_embeddings, lct_embeddings, device, topk=5, chunk_size=256):
    """Return LCT -> VehicleID top-k and VehicleID -> LCT top-1 from the same chunked matrix products."""
    external = F.normalize(torch.tensor(np.asarray(external_embeddings), device=device), dim=1)
    best_value = torch.full((len(external),), -1.0, dtype=torch.float32, device=device)
    best_lct = torch.full((len(external),), -1, dtype=torch.long, device=device)
    values, indices = [], []
    for start in range(0, len(lct_embeddings), chunk_size):
        lct = F.normalize(torch.tensor(np.asarray(lct_embeddings[start:start + chunk_size]), device=device), dim=1)
        similarity = lct @ external.T
        value, index = similarity.topk(min(topk, len(external)), dim=1)
        values.append(value.float().cpu().numpy())
        indices.append(index.cpu().numpy())
        reverse_value, reverse_row = similarity.float().max(dim=0)
        update = reverse_value > best_value
        best_value[update] = reverse_value[update]
        best_lct[update] = reverse_row[update] + start
    return (np.concatenate(values), np.concatenate(indices), best_value.cpu().numpy(), best_lct.cpu().numpy())


def neighbour_table(lct, external, values, indices, reverse_lct):
    rows = []
    for lct_index in range(len(lct)):
        for rank, (similarity, external_index) in enumerate(zip(values[lct_index], indices[lct_index]), start=1):
            ext = external.iloc[int(external_index)]
            item = lct.iloc[lct_index]
            rows.append({"lct_index": lct_index, "lct_split": item["split"], "lct_image_id": item.image_id,
                         "lct_vehicle_id": item.get("vehicle_id"), "lct_path": item.path,
                         "external_index": int(external_index), "external_image_id": ext.image_id,
                         "external_vehicle_id": ext.vehicle_id, "external_path": ext.path,
                         "rank": rank, "cosine": float(similarity),
                         "mutual_image_nn": int(reverse_lct[int(external_index)]) == lct_index})
    return pd.DataFrame(rows)


def identity_support(neighbours):
    """Rank labelled LCT-train/VehicleID identity pairs by independent image support."""
    train = neighbours[(neighbours.lct_split == "train") & neighbours.lct_vehicle_id.notna()]
    support = (train.groupby(["lct_vehicle_id", "external_vehicle_id"], as_index=False)
               .agg(lct_images=("lct_image_id", "nunique"), external_images=("external_image_id", "nunique"),
                    mutual_pairs=("mutual_image_nn", "sum"), max_cosine=("cosine", "max"),
                    mean_cosine=("cosine", "mean")))
    return support.sort_values(["lct_images", "external_images", "mutual_pairs", "max_cosine"],
                               ascending=False).reset_index(drop=True)


def _tile(path, title, size=(256, 192)):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    canvas = np.full((size[1] + 28, size[0], 3), 245, np.uint8)
    scale = min(size[0] / image.shape[1], size[1] / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    y, x = (size[1] - resized.shape[0]) // 2, (size[0] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, title[:46], (4, size[1] + 19), cv2.FONT_HERSHEY_SIMPLEX, .39, (20, 20, 20), 1,
                cv2.LINE_AA)
    return canvas


def save_montage(neighbours, path, limit=64):
    rows = []
    for match in neighbours.head(limit).itertuples():
        rows.append(np.concatenate([
            _tile(match.lct_path, f"LCT {match.lct_split}/{match.lct_image_id}"),
            _tile(match.external_path, f"VID {match.external_image_id} cos={match.cosine:.4f} m={match.mutual_image_nn}"),
        ], axis=1))
    if rows:
        cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default="vehicleid_train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--similarity-chunk", type=int, default=256)
    args = parser.parse_args()

    data, cfg, out = load_yaml(args.data_config), load_yaml(args.config), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = data["external"][args.source]
    external = load_table(spec["dir"], spec.get("table", "train.csv"))
    external["path"] = external.image_id.map(lambda image_id: str(Path(spec["dir"]) / "images" / f"{image_id}.jpg"))
    external["crops_dir"], external["label"] = str(Path(spec["dir"]) / "images"), np.arange(len(external))
    lct = pd.concat([load_csv(data["dataset_dir"], split).assign(split=split) for split in CSV_FILES],
                    ignore_index=True).drop_duplicates("image_id").reset_index(drop=True)
    lct["path"] = lct.image_id.map(lambda image_id: str(Path(data["crops_dir"]) / f"{image_id}.jpg"))
    lct["crops_dir"], lct["label"] = data["crops_dir"], np.arange(len(lct))

    encoder, dim, mean, std = build_backbone(cfg["model"], data.get("external_weights_dir"))
    encoder = encoder.eval().to(args.device)
    recipe = plain_recipe(cfg["train"])
    external_path, lct_path = out / "vehicleid_embeddings.npy", out / "lct_embeddings.npy"
    embed_to_memmap(encoder, external, cfg["model"]["input_hw"], mean, std, recipe, external_path, dim,
                    args.device, args.batch_size, args.workers)
    embed_to_memmap(encoder, lct, cfg["model"]["input_hw"], mean, std, recipe, lct_path, dim,
                    args.device, args.batch_size, args.workers)
    del encoder
    torch.cuda.empty_cache()

    external_embeddings, lct_embeddings = np.load(external_path, mmap_mode="r"), np.load(lct_path, mmap_mode="r")
    values, indices, reverse_values, reverse_lct = cross_neighbours(
        external_embeddings, lct_embeddings, args.device, args.topk, args.similarity_chunk)
    neighbours = neighbour_table(lct, external, values, indices, reverse_lct)
    neighbours.to_csv(out / "nearest.csv", index=False)
    support = identity_support(neighbours)
    support.to_csv(out / "identity_support.csv", index=False)
    ranked = neighbours.sort_values(["mutual_image_nn", "cosine"], ascending=False)
    save_montage(ranked, out / "nearest_montage.jpg")
    supported = (neighbours.merge(support, on=["lct_vehicle_id", "external_vehicle_id"], how="inner")
                 .sort_values(["lct_images", "external_images", "mutual_pairs", "max_cosine", "cosine"],
                              ascending=False)
                 .drop_duplicates(["lct_vehicle_id", "external_vehicle_id"]))
    supported.to_csv(out / "identity_support_representatives.csv", index=False)
    save_montage(supported, out / "identity_support_montage.jpg")
    report = {"source": args.source, "model": cfg["model"]["name"], "external_images": len(external),
              "external_vehicles": int(external.vehicle_id.nunique()), "lct_crops": len(lct), "topk": args.topk,
              "cosine_max_quantiles": {str(q): float(np.quantile(values[:, 0], q)) for q in (.5, .9, .99, .999, 1)},
              "mutual_image_pairs": int(neighbours.mutual_image_nn.sum()),
              "multi_frame_identity_candidates": int(((support.lct_images >= 2) &
                                                        (support.external_images >= 2)).sum()),
              "requires_manual_review": True}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
