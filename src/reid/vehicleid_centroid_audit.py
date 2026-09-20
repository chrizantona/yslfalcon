"""Identity-level review of saved VehicleID/LCT semantic-audit embeddings.

This post-process separates a VehicleID identity that resembles many generic LCT cars from a mutual identity match.
It also calibrates cross-dataset centroid cosine against true LCT cross-camera pairs and nearest foreign LCT IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from reid.data import CSV_FILES, load_csv, load_yaml
from reid.external import load_table
from reid.vehicleid_semantic_audit import save_montage


def grouped_centres(ids, embeddings, device):
    codes, names = pd.factorize(pd.Series(ids, dtype=str), sort=True)
    vectors = torch.tensor(np.asarray(embeddings), device=device)
    code_tensor = torch.as_tensor(codes, dtype=torch.long, device=device)
    sums = torch.zeros((len(names), vectors.shape[1]), dtype=torch.float32, device=device)
    sums.index_add_(0, code_tensor, vectors.float())
    counts = torch.bincount(code_tensor, minlength=len(names)).clamp_min(1)
    return np.asarray(names, dtype=str), F.normalize(sums / counts[:, None], dim=1)


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (.01, .1, .5, .9, .99, 1)}


def cross_camera_cosines(train, embeddings):
    values = []
    for _, identity in train.groupby("vehicle_key", sort=False):
        centres = []
        for _, camera in identity.groupby("camera_id", sort=False):
            centre = np.asarray(embeddings[camera.embedding_index.to_numpy()]).astype(np.float32).mean(0)
            centres.append(centre / np.linalg.norm(centre))
        if len(centres) > 1:
            similarity = np.asarray(centres) @ np.asarray(centres).T
            values.extend(similarity[np.triu_indices(len(centres), 1)].tolist())
    return np.asarray(values, np.float32)


def representative(identity_pairs, lct, external, lct_embeddings, external_embeddings, limit=64):
    lct_groups = {key: rows.index.to_numpy() for key, rows in lct.groupby("vehicle_key")}
    ext_groups = {key: rows.index.to_numpy() for key, rows in external.groupby("vehicle_id")}
    records = []
    for pair in identity_pairs.head(limit).itertuples():
        left, right = lct_groups[pair.lct_vehicle_id], ext_groups[pair.external_vehicle_id]
        similarity = (np.asarray(lct_embeddings[left], np.float32)
                      @ np.asarray(external_embeddings[right], np.float32).T)
        row, column = np.unravel_index(similarity.argmax(), similarity.shape)
        lct_row, ext_row = lct.loc[left[row]], external.loc[right[column]]
        records.append({"lct_path": lct_row.path, "lct_split": "train", "lct_image_id": lct_row.image_id,
                        "external_path": ext_row.path, "external_image_id": ext_row.image_id,
                        "cosine": float(similarity[row, column]), "mutual_image_nn": pair.mutual_identity_nn,
                        "lct_vehicle_id": pair.lct_vehicle_id,
                        "external_vehicle_id": pair.external_vehicle_id,
                        "centroid_cosine": pair.centroid_cosine})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--source", default="vehicleid_train")
    parser.add_argument("--semantic-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    data, out = load_yaml(args.data_config), Path(args.semantic_dir)
    spec = data["external"][args.source]
    external = load_table(spec["dir"], spec.get("table", "train.csv")).reset_index(drop=True)
    external["path"] = external.image_id.map(lambda x: str(Path(spec["dir"]) / "images" / f"{x}.jpg"))
    lct = pd.concat([load_csv(data["dataset_dir"], split).assign(split=split) for split in CSV_FILES],
                    ignore_index=True).drop_duplicates("image_id").reset_index(drop=True)
    lct["path"] = lct.image_id.map(lambda x: str(Path(data["crops_dir"]) / f"{x}.jpg"))
    lct["embedding_index"] = np.arange(len(lct))
    train = lct[(lct["split"] == "train") & lct.vehicle_id.notna()].copy()
    train["vehicle_key"] = train.vehicle_id.astype(int).astype(str)

    external_embeddings = np.load(out / "vehicleid_embeddings.npy", mmap_mode="r")
    lct_embeddings = np.load(out / "lct_embeddings.npy", mmap_mode="r")
    ext_names, ext_centres = grouped_centres(external.vehicle_id, external_embeddings, args.device)
    lct_names, lct_centres = grouped_centres(train.vehicle_key, lct_embeddings[train.embedding_index], args.device)
    similarity = lct_centres @ ext_centres.T
    values, indices = similarity.topk(min(args.topk, len(ext_centres)), dim=1)
    reverse_lct = similarity.argmax(0)
    records = []
    for i in range(len(lct_names)):
        for rank, (value, j) in enumerate(zip(values[i].tolist(), indices[i].tolist()), start=1):
            records.append({"lct_vehicle_id": lct_names[i], "external_vehicle_id": ext_names[j], "rank": rank,
                            "centroid_cosine": value, "mutual_identity_nn": int(reverse_lct[j]) == i})
    pairs = pd.DataFrame(records).sort_values(["mutual_identity_nn", "centroid_cosine"], ascending=False)
    pairs.to_csv(out / "identity_centroid_nearest.csv", index=False)

    lct_similarity = lct_centres @ lct_centres.T
    lct_similarity.fill_diagonal_(float("-inf"))
    nearest_foreign = lct_similarity.max(1).values.cpu().numpy()
    true_cross_camera = cross_camera_cosines(train, lct_embeddings)
    representatives = representative(pairs, train.set_index(train.index), external, lct_embeddings,
                                     external_embeddings)
    representatives.to_csv(out / "identity_centroid_representatives.csv", index=False)
    save_montage(representatives, out / "identity_centroid_montage.jpg")
    report = {"lct_train_identities": len(lct_names), "vehicleid_identities": len(ext_names), "topk": args.topk,
              "cross_dataset_max_centroid_cosine": quantiles(values[:, 0].cpu().numpy()),
              "nearest_foreign_lct_centroid_cosine": quantiles(nearest_foreign),
              "true_lct_cross_camera_cosine": quantiles(true_cross_camera),
              "mutual_identity_pairs": int(pairs.mutual_identity_nn.sum())}
    (out / "centroid_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
