"""Speed and size of the feature extractor, timed the way the organisers do (answers of 2026-09-16).

- latency_b1: median of 300 runs after 50 warm-up runs of the full cycle for one vehicle - read the file, decode,
  crop, preprocess, forward, L2 normalisation - with CUDA synchronised before and after each run;
- throughput: steady images per second of the same cycle at batch 1 / 8 / 16 / 32, at least 10 s each; the best counts;
- also peak VRAM, weight loading time, weight size and determinism (two passes give identical embeddings).
Performance points (20 in total): 10 x latency score (<= 40 ms full, 80 ms zero) + 10 x throughput score
(>= 100 FPS full, 50 FPS zero). Measured on the pod's H200, not on the organisers' RTX A5000 with 2.0 GHz Xeons,
so the numbers are optimistic. Writes artifacts/<exp_id>/benchmark.json.

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
import torch.nn.functional as F

from reid.data import crop_box, load_csv, load_yaml, to_model_input
from reid.embed import FlipTTA, pick_device
from reid.inference import load_trained_encoder
from reid.models import build_backbone


class InferencePrecision(torch.nn.Module):
    """Apply the configured inference autocast to the complete encoder, including BNNeck."""

    def __init__(self, encoder, precision, device):
        super().__init__()
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {precision}")
        if precision != "fp32" and device != "cuda":
            raise ValueError("fp16/bf16 benchmark requires CUDA")
        self.encoder, self.precision, self.device = encoder, precision, device

    def forward(self, x):
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        with torch.autocast(self.device, dtype=dtype, enabled=self.precision != "fp32"):
            return self.encoder(x)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def single_vehicle_latency(encoder, rows, images_dir, input_hw, mean, std, max_side, device, warmup=50, runs=300):
    """Median ms of the full cycle for one vehicle, and of its forward pass alone."""
    full, forward = [], []
    for i in range(warmup + runs):
        r = rows.iloc[i % len(rows)]
        sync(device)
        start = time.perf_counter()
        frame = cv2.imread(str(images_dir / f"{r.image_id}.jpg"))
        x = to_model_input(crop_box(frame, (r.x, r.y, r.w, r.h), max_side), input_hw, mean, std)[None].to(device)
        sync(device)
        after_preprocess = time.perf_counter()
        F.normalize(encoder(x).float(), dim=-1).cpu()
        sync(device)
        end = time.perf_counter()
        if i >= warmup:
            full.append((end - start) * 1000)
            forward.append((end - after_preprocess) * 1000)
    return float(np.median(full)), float(np.median(forward))


@torch.no_grad()
def pipeline_throughput(encoder, rows, images_dir, input_hw, mean, std, max_side, device, batch_size,
                        threads=8, warmup_batches=3, seconds=10.0):
    """Steady images per second of the full cycle over at least `seconds`: decode and crop in a thread pool, embed in
    batches, cycling over rows.

    Threads instead of DataLoader workers: forking a process that already runs CUDA and OpenCV threads deadlocks
    the workers, and a single-process service is closer to real inference anyway. OpenCV releases the GIL while
    decoding and resizing, so the threads run in parallel."""

    def load(r):
        frame = cv2.imread(str(images_dir / f"{r.image_id}.jpg"))
        return to_model_input(crop_box(frame, (r.x, r.y, r.w, r.h), max_side), input_hw, mean, std)

    batch_no, processed, start = 0, 0, None
    with ThreadPoolExecutor(threads) as pool:
        while start is None or time.perf_counter() - start < seconds:
            idx = [(batch_no * batch_size + j) % len(rows) for j in range(batch_size)]
            x = torch.stack(list(pool.map(load, rows.iloc[idx].itertuples())))
            F.normalize(encoder(x.to(device)).float(), dim=-1).cpu()
            sync(device)
            batch_no += 1
            if batch_no == warmup_batches:
                start = time.perf_counter()
            elif batch_no > warmup_batches:
                processed += len(x)
    return processed / (time.perf_counter() - start)


def embed_twice(encoder, rows, images_dir, input_hw, mean, std, max_side, device, batch_size=32):
    """Largest absolute difference between the embeddings of two identical passes (0 means deterministic)."""
    x = torch.stack([to_model_input(crop_box(cv2.imread(str(images_dir / f"{r.image_id}.jpg")), (r.x, r.y, r.w, r.h),
                                             max_side), input_hw, mean, std) for r in rows.itertuples()])
    with torch.no_grad():
        passes = [torch.cat([F.normalize(encoder(x[i: i + batch_size].to(device)).float(), dim=-1).cpu()
                             for i in range(0, len(x), batch_size)]) for _ in range(2)]
    return float((passes[0] - passes[1]).abs().max())


def performance_points(latency_ms, fps):
    """Organisers' formula: 10 points for latency (40 -> 80 ms linear) + 10 for the best FPS (50 -> 100 linear)."""
    latency_score = min(max((80 - latency_ms) / 40, 0), 1)
    throughput_score = min(max((fps - 50) / 50, 0), 1)
    return round(10 * latency_score + 10 * throughput_score, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--images", type=int, default=1536, help="images cycled through by the throughput runs")
    parser.add_argument("--out")
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    device = pick_device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model_cfg = cfg["model"] if "backbone" in cfg["model"] else {"backbone": "clip", **cfg["model"]}
    if args.checkpoint:
        encoder, feat_dim, mean, std = load_trained_encoder(
            model_cfg, args.checkpoint, data_cfg.get("external_weights_dir"), cfg.get("checkpoint_sha256"))
    else:
        encoder, feat_dim, mean, std = build_backbone(model_cfg, data_cfg.get("external_weights_dir"))
    precision = cfg.get("benchmark", {}).get("precision", "fp32")
    encoder = InferencePrecision(encoder, precision, device).eval().to(device)
    if cfg.get("embed", {}).get("flip_tta"):
        encoder = FlipTTA(encoder).eval()
    sync(device)
    load_seconds = time.perf_counter() - load_start

    dataset_dir = Path(data_cfg["dataset_dir"])
    images_dir = dataset_dir / "images"
    rows = load_csv(dataset_dir, "query").iloc[:args.images]
    input_hw, max_side = cfg["model"]["input_hw"], data_cfg["crop_max_side"]

    latency_ms, forward_ms = single_vehicle_latency(encoder, rows, images_dir, input_hw, mean, std, max_side, device)
    params = sum(p.numel() for p in encoder.parameters())
    result = {
        "exp_id": cfg["exp_id"],
        "backbone": model_cfg.get("name", model_cfg["backbone"]),  # FastReID configs name the weights file, not a model
        "input_hw": list(input_hw),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "precision": precision,
        "bnneck": bool(args.checkpoint),
        "flip_tta": bool(cfg.get("embed", {}).get("flip_tta")),
        "weights_storage": "fp32",
        "latency_bs1_ms": round(latency_ms, 2),
        "forward_bs1_ms": round(forward_ms, 2),
        "embedding_dim": int(feat_dim),
        "params_m": round(params / 1e6, 1),
        "weights_mb_fp32": round(params * 4 / 1e6, 1),
        "load_seconds": round(load_seconds, 2),
    }
    for batch_size in (1, 8, 16, 32):
        result[f"fps_bs{batch_size}"] = round(pipeline_throughput(encoder, rows, images_dir, input_hw, mean, std,
                                                                 max_side, device, batch_size), 1)
    result["fps_best"] = max(result[f"fps_bs{b}"] for b in (1, 8, 16, 32))
    result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None
    result["determinism_max_abs_diff"] = embed_twice(encoder, rows.iloc[:256], images_dir, input_hw, mean, std,
                                                     max_side, device)
    result["performance_points"] = performance_points(result["latency_bs1_ms"], result["fps_best"])
    if args.checkpoint:
        result["checkpoint"] = Path(args.checkpoint).name
        result["checkpoint_mb"] = round(Path(args.checkpoint).stat().st_size / 1e6, 1)

    out = Path(args.out) if args.out else Path("artifacts") / cfg["exp_id"] / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
