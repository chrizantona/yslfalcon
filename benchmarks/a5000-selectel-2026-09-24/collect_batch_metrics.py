"""Collect short, reproducible A5000 batch measurements for the final model.

The first batch of each size is a warm-up and is excluded. Five following
batches include JPEG read, bbox crop, preprocessing, host-to-device transfer,
forward pass, L2 normalisation, and transfer of embeddings back to CPU.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from reid.benchmark import InferencePrecision, sync
from reid.data import crop_box, load_csv, load_yaml, to_model_input
from reid.inference import load_trained_encoder


def nvidia_smi() -> dict:
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "clocks.max.sm",
    ]
    values = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    return dict(zip(fields, values, strict=True))


def meminfo() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[f"{key}_mb"] = round(int(value.strip().split()[0]) / 1024, 1)
    return values


def process_memory() -> dict:
    wanted = {"VmRSS", "VmHWM"}
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            values[f"{key}_mb"] = round(int(value.strip().split()[0]) / 1024, 1)
    return values


def summary(samples: list[dict], batch_size: int) -> dict:
    totals = [sample["total_ms"] for sample in samples]
    forwards = [sample["forward_ms"] for sample in samples]
    total_seconds = sum(totals) / 1000
    return {
        "batch_size": batch_size,
        "measured_batches": len(samples),
        "images": batch_size * len(samples),
        "batch_total_ms_mean": round(statistics.mean(totals), 2),
        "batch_total_ms_median": round(statistics.median(totals), 2),
        "batch_total_ms_min": round(min(totals), 2),
        "batch_total_ms_max": round(max(totals), 2),
        "per_image_ms": round(sum(totals) / (batch_size * len(samples)), 2),
        "fps_end_to_end": round(batch_size * len(samples) / total_seconds, 1),
        "forward_ms_mean": round(statistics.mean(forwards), 2),
        "peak_vram_allocated_mb": max(sample["vram_allocated_mb"] for sample in samples),
        "peak_vram_reserved_mb": max(sample["vram_reserved_mb"] for sample in samples),
        "process_rss_mb_max": max(sample["process_memory"]["VmRSS_mb"] for sample in samples),
        "process_peak_rss_mb": max(sample["process_memory"]["VmHWM_mb"] for sample in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--measured-batches", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA is required")

    load_started = time.perf_counter()
    encoder, feature_dim, mean, std = load_trained_encoder(
        cfg["model"], args.checkpoint, data_cfg.get("external_weights_dir"), cfg.get("checkpoint_sha256")
    )
    precision = cfg.get("inference", {}).get("precision", "fp32")
    encoder = InferencePrecision(encoder, precision, device).eval().to(device)
    sync(device)
    load_seconds = time.perf_counter() - load_started

    dataset = Path(data_cfg["dataset_dir"])
    rows = load_csv(dataset, "query")
    images = dataset / "images"
    input_hw = cfg["model"]["input_hw"]
    max_side = data_cfg["crop_max_side"]

    def load(row):
        frame = cv2.imread(str(images / f"{row.image_id}.jpg"))
        return to_model_input(crop_box(frame, (row.x, row.y, row.w, row.h), max_side), input_hw, mean, std)

    results = []
    cursor = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for batch_size in args.batch_sizes:
            samples = []
            total_batches = args.warmup_batches + args.measured_batches
            def submit_batch():
                nonlocal cursor
                selection = [rows.iloc[(cursor + i) % len(rows)] for i in range(batch_size)]
                cursor += batch_size
                return [pool.submit(load, row) for row in selection]

            pending = submit_batch()
            for batch_index in range(total_batches):
                sync(device)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                batch = torch.stack([future.result() for future in pending]).to(device)
                pending = submit_batch()
                sync(device)
                after_preprocess = time.perf_counter()
                with torch.no_grad():
                    embedding = F.normalize(encoder(batch).float(), dim=-1).cpu()
                sync(device)
                finished = time.perf_counter()
                if not torch.isfinite(embedding).all():
                    raise RuntimeError("non-finite embedding")
                sample = {
                    "batch_index": batch_index,
                    "warmup": batch_index < args.warmup_batches,
                    "preprocess_ms": round((after_preprocess - started) * 1000, 2),
                    "forward_ms": round((finished - after_preprocess) * 1000, 2),
                    "total_ms": round((finished - started) * 1000, 2),
                    "vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
                    "vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 1),
                    "process_memory": process_memory(),
                }
                if batch_index >= args.warmup_batches:
                    samples.append(sample)
            results.append({"summary": summary(samples, batch_size), "samples": samples})

    report = {
        "methodology": {
            "batch_sizes": args.batch_sizes,
            "warmup_batches_per_size": args.warmup_batches,
            "measured_batches_per_size": args.measured_batches,
            "workers": args.workers,
            "timed_cycle": "pipelined jpeg read + bbox crop + resize/normalize + H2D + encoder + L2 + D2H",
            "cpu_gpu_prefetch_overlap": True,
        },
        "environment": {
            "gpu": nvidia_smi(),
            "cpu_logical_available": os.cpu_count(),
            "system_memory": meminfo(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "model": {
            "model_id": "emb-so400m-224-v1",
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
            "precision": precision,
            "embedding_dim": feature_dim,
            "load_seconds": round(load_seconds, 2),
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as stream:
        rows = [item["summary"] for item in results]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
