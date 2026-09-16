"""Validate submission artifacts against the test csvs; lists every problem and exits with code 1 if there are any.

Usage: PYTHONPATH=src python -m reid.check_submission --data-config configs/data.yaml --config configs/exp000_clip_zeroshot.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_csv, load_yaml
from reid.metrics import TOP_K


def check(sub_dir, query_ids, gallery_ids):
    problems = []
    n_query = len(query_ids)

    emb = np.load(sub_dir / "embeddings.npy")
    emb_ok = emb.ndim == 2 and emb.shape[0] == n_query + len(gallery_ids)
    if not emb_ok:
        problems.append(f"embeddings.npy: shape {emb.shape}, expected ({n_query + len(gallery_ids)}, dim)")
    if emb.dtype != np.float32:
        problems.append(f"embeddings.npy: dtype {emb.dtype}, expected float32")
    if not np.isfinite(emb).all():
        problems.append("embeddings.npy: contains NaN or inf")

    sub = pd.read_csv(sub_dir / "submission.csv", dtype=str)
    columns = ["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)]
    if list(sub.columns) != columns:
        problems.append(f"submission.csv: columns {list(sub.columns)}, expected {columns}")
    else:
        ranked = sub[columns[1:]].to_numpy()
        if sub.query_id.tolist() != list(query_ids):
            problems.append("submission.csv: query_id must list every test query once, in test_query.csv order")
        if not np.isin(ranked, gallery_ids).all():
            problems.append("submission.csv: unknown gallery ids")
        if any(len(set(row)) != TOP_K for row in ranked):
            problems.append("submission.csv: repeated gallery ids within a row")
        if emb_ok and len(sub) == n_query:  # the organisers check that the ranking comes from embeddings.npy
            top = np.argsort(-(emb[:n_query] @ emb[n_query:].T), axis=1, kind="stable")[:, :TOP_K]
            if not (gallery_ids[top] == ranked).all():
                problems.append("submission.csv: ranking is not reproducible from embeddings.npy")

    cand = pd.read_csv(sub_dir / "candidates.csv", dtype=str, keep_default_na=False)
    if list(cand.columns) != ["query_id", "gallery_id", "confidence"]:
        problems.append(f"candidates.csv: columns {list(cand.columns)}, expected [query_id, gallery_id, confidence]")
    else:  # a refusal is a query without rows; empty ids or scores are not allowed
        if (cand.gallery_id == "").any() or (cand.confidence == "").any():
            problems.append("candidates.csv: empty gallery_id or confidence; a refused query must have no rows")
        if not cand.query_id.isin(query_ids).all():
            problems.append("candidates.csv: unknown query ids")
        if not cand.gallery_id[cand.gallery_id != ""].isin(gallery_ids).all():
            problems.append("candidates.csv: unknown gallery ids")
        if pd.to_numeric(cand.confidence, errors="coerce").isna().any():
            problems.append("candidates.csv: confidence must be a number")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    dataset_dir = data_cfg["dataset_dir"]
    query_ids = load_csv(dataset_dir, "query").image_id.to_numpy()
    gallery_ids = load_csv(dataset_dir, "gallery").image_id.to_numpy()
    problems = check(Path("artifacts") / cfg["exp_id"] / "submission", query_ids, gallery_ids)
    print("\n".join(problems) if problems else "submission OK")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
