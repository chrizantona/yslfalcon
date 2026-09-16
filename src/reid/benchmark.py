"""Speed and size of the feature extractor: single-vehicle latency, batch throughput, weight size.

Timing depends on the architecture and the input size, not on the trained weights, so an experiment can be
benchmarked from its config alone. Writes artifacts/<exp_id>/benchmark.json.

Usage: PYTHONPATH=src python -m reid.benchmark --data-config configs/data.yaml --config configs/exp003_clip_vitb16_bot_30ep.yaml
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from reid.data import crop_box, load_csv, load_yaml, to_model_input
from reid.embed import pick_device
from reid.models import build_backbone


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def single_vehicle_latency(encoder, rows, images_dir, input_hw, mean, std, max_side, device, warmup=20, runs=200):
    """Median ms per vehicle for the whole path: read the frame, crop by box, preprocess, embed with batch size 1."""
    full, forward = [], []
    for i in range(warmup + runs):
        r = rows.iloc[i % len(rows)]
        start = time.perf_counter()
        frame = cv2.imread(str(images_dir / f"{r.image_id}.jpg"))
        x = to_model_input(crop_box(frame, (r.x, r.y, r.w, r.h), max_side), input_hw, mean, std)[None].to(device)
        sync(device)
        after_preprocess = time.perf_counter()
        encoder(x)
        sync(device)
        end = time.perf_counter()
        if i >= warmup:
            full.append((end - start) * 1000)
            forward.append((end - after_preprocess) * 1000)
    return float(np.median(full)), float(np.median(forward))


@torch.no_grad()
def pipeline_throughput(encoder, rows, images_dir, input_hw, mean, std, max_side, device, batch_size,
                        threads=8, warmup_batches=2):
    """Steady-state images per second for the full pipeline: decode and crop in a thread pool, embed in batches.

    Threads instead of DataLoader workers: forking a process that already runs CUDA and OpenCV threads deadlocks
    the workers, and a single-process service is closer to real inference anyway. OpenCV releases the GIL while
    decoding and resizing, so the threads run in parallel."""

    def load(r):
        frame = cv2.imread(str(images_dir / f"{r.image_id}.jpg"))
        return to_model_input(crop_box(frame, (r.x, r.y, r.w, r.h), max_side), input_hw, mean, std)

    processed, start = 0, None
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(rows), batch_size):
            x = torch.stack(list(pool.map(load, rows.iloc[i: i + batch_size].itertuples())))
            encoder(x.to(device))
            sync(device)
            if i // batch_size < warmup_batches:
                continue
            if start is None:
                start = time.perf_counter()
                continue
            processed += len(x)
    return processed / (time.perf_counter() - start) if processed else float("nan")


@torch.no_grad()
def gpu_throughput(encoder, input_hw, device, batch_size, warmup=5, iters=20):
    """Images per second of the model alone, on an already preprocessed batch."""
    x = torch.zeros((batch_size, 3, *input_hw), device=device)
    for _ in range(warmup):
        encoder(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        encoder(x)
    sync(device)
    return iters * batch_size / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--images", type=int, default=1536, help="images used for the throughput measurement")
    parser.add_argument("--out")
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    device = pick_device()
    model_cfg = cfg["model"] if "backbone" in cfg["model"] else {"backbone": "clip", **cfg["model"]}
    encoder, feat_dim, mean, std = build_backbone(model_cfg, data_cfg.get("external_weights_dir"))
    if args.checkpoint:  # weights change nothing for timing; loaded only when the real file size is wanted
        state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        encoder.load_state_dict({k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")})
    encoder = encoder.eval().to(device)

    dataset_dir = Path(data_cfg["dataset_dir"])
    images_dir = dataset_dir / "images"
    rows = load_csv(dataset_dir, "query").iloc[:args.images]
    input_hw, max_side = cfg["model"]["input_hw"], data_cfg["crop_max_side"]

    latency_ms, forward_ms = single_vehicle_latency(encoder, rows, images_dir, input_hw, mean, std, max_side, device)
    params = sum(p.numel() for p in encoder.parameters())
    result = {
        "exp_id": cfg["exp_id"],
        "backbone": model_cfg["name"],
        "input_hw": list(input_hw),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "latency_bs1_ms": round(latency_ms, 2),
        "forward_bs1_ms": round(forward_ms, 2),
        "embedding_dim": int(feat_dim),
        "params_m": round(params / 1e6, 1),
        "weights_mb_fp32": round(params * 4 / 1e6, 1),
    }
    for batch_size in (32, 64, 128):
        result[f"fps_bs{batch_size}"] = round(pipeline_throughput(encoder, rows, images_dir, input_hw, mean, std,
                                                                 max_side, device, batch_size), 1)
        result[f"gpu_fps_bs{batch_size}"] = round(gpu_throughput(encoder, input_hw, device, batch_size), 1)
    if args.checkpoint:
        result["checkpoint"] = Path(args.checkpoint).name
        result["checkpoint_mb"] = round(Path(args.checkpoint).stat().st_size / 1e6, 1)

    out = Path(args.out) if args.out else Path("artifacts") / cfg["exp_id"] / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
