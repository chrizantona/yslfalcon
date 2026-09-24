"""Prove the image works on the GPU it was given, and measure it the way the organisers do.

The container is handed to a rented machine that nobody has logged into, so this answers, in one run: does CUDA see
the card, do the published weights arrive intact, does a forward pass produce a sane embedding, and what are the two
numbers that carry twenty points - time per vehicle and best FPS.

No competition data ships in the image. With a dataset mounted at /data the measurement uses it; otherwise the check
paints its own 1920x1080 frames, which exercises the same decode-crop-resize-forward cycle. JPEG decoding depends on
what the frame contains, so a synthetic number is a sanity check, not a submission-grade measurement - the line it
prints says which one it is.

Usage inside the image: python -m reid.selfcheck --config configs/final_model.yaml
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from reid.benchmark import InferencePrecision, pipeline_throughput, single_vehicle_latency, sync
from reid.data import load_yaml
from reid.inference import load_trained_encoder
from reid import view_head

LATENCY_BUDGET_MS = 40.0
FPS_BUDGET = 100.0
WEIGHTS_BUDGET_MB = 2048.0


def environment():
    """What the host actually gave us."""
    report = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "cpu_count": os.cpu_count()}
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
        for field, key in (("driver_version", "driver"), ("power.limit", "power_limit_w"),
                           ("power.default_limit", "power_limit_default_w")):
            try:
                value = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=20).stdout.strip()
                report[key] = value.split()[0] if value else None
            except (OSError, subprocess.SubprocessError):
                report[key] = None
    return report


def weights(cfg, checkpoint_dir):
    """Every published file present, with the size and digest the manifest pins."""
    manifest = json.loads(Path("configs/weights.json").read_text())
    total, files = 0, {}
    for name, digest in manifest["files"].items():
        path = Path(checkpoint_dir) / name
        if not path.exists():
            raise SystemExit(f"missing weight file in the image: {path}")
        checksum, size = hashlib.sha256(), path.stat().st_size
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise SystemExit(f"checksum mismatch for {name}")
        files[name] = round(size / 1e6, 1)
        total += size
    return {"files": files, "total_mb": round(total / 1e6, 1), "within_budget": total / 1e6 <= WEIGHTS_BUDGET_MB}


def painted_dataset(target, count, rng):
    """A dataset directory of plausible frames: gradients, road-like bands and a few solids, never real data."""
    images = target / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        y = np.linspace(0, 1, 1080, dtype=np.float32)[:, None]
        frame = (np.stack([y * rng.uniform(80, 180) + rng.uniform(20, 60)] * 3, -1)
                 + rng.normal(0, 6, (1080, 1920, 3))).clip(0, 255)
        cv2.rectangle(frame, (0, 700), (1920, 1080), tuple(float(v) for v in rng.uniform(40, 90, 3)), -1)
        for _ in range(6):
            x0, y0 = int(rng.uniform(0, 1500)), int(rng.uniform(200, 900))
            cv2.rectangle(frame, (x0, y0), (x0 + int(rng.uniform(80, 400)), y0 + int(rng.uniform(60, 300))),
                          tuple(float(v) for v in rng.uniform(0, 255, 3)), -1)
        cv2.ellipse(frame, (int(rng.uniform(400, 1500)), int(rng.uniform(400, 800))), (220, 150),
                    float(rng.uniform(0, 180)), 0, 360, tuple(float(v) for v in rng.uniform(0, 255, 3)), -1)
        name = f"synthetic_{i:05d}"
        cv2.imwrite(str(images / f"{name}.jpg"), frame.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        w, h = int(rng.uniform(420, 900)), int(rng.uniform(320, 660))
        rows.append({"image_id": name, "x": int(rng.uniform(0, 1920 - w)), "y": int(rng.uniform(0, 1080 - h)),
                     "w": w, "h": h})
    table = pd.DataFrame(rows)
    table.to_csv(target / "test_query.csv", index=False)
    return table, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_model.yaml")
    parser.add_argument("--data-config", default="configs/data_docker.yaml")
    parser.add_argument("--frames", type=int, default=96, help="synthetic frames when no dataset is mounted")
    parser.add_argument("--out", default="/app/artifacts/selfcheck.json")
    args = parser.parse_args()
    cfg, data_cfg = load_yaml(args.config), load_yaml(args.data_config)
    report = {"environment": environment()}
    print(json.dumps(report["environment"], indent=2), flush=True)

    report["weights"] = weights(cfg, data_cfg["checkpoint_dir"])
    print(json.dumps(report["weights"], indent=2), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    encoder, dim, mean, std = load_trained_encoder(cfg["model"], Path(data_cfg["checkpoint_dir"]) / cfg["checkpoint_name"])
    encoder = InferencePrecision(encoder.eval().to(device), cfg.get("inference", {}).get("precision", "fp16")
                                 if device == "cuda" else "fp32", device)
    sync(device)
    report["load_seconds"] = round(time.perf_counter() - load_start, 2)

    dataset = Path(data_cfg["dataset_dir"])
    real = (dataset / "test_query.csv").exists() and (dataset / "images").is_dir()
    if real:
        rows = pd.read_csv(dataset / "test_query.csv").iloc[: max(args.frames, 512)]
        images_dir = dataset / "images"
    else:
        rows, images_dir = painted_dataset(Path("/tmp/selfcheck"), args.frames, np.random.default_rng(0))
    report["frames"] = {"source": "mounted dataset" if real else "painted in the container", "count": int(len(rows))}

    input_hw, max_side = cfg["model"]["input_hw"], data_cfg["crop_max_side"]
    latency_ms, forward_ms = single_vehicle_latency(encoder, rows, images_dir, input_hw, mean, std, max_side, device)
    fps = {}
    for batch in (1, 8, 16, 32):
        fps[batch] = round(pipeline_throughput(encoder, rows, images_dir, input_hw, mean, std, max_side, device,
                                               batch_size=batch), 1)
    best = max(fps.values())
    report["speed"] = {"latency_bs1_ms": round(latency_ms, 2), "forward_bs1_ms": round(forward_ms, 2),
                       "fps": fps, "fps_best": best,
                       "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else None}

    # one honest forward: the embedding must be finite, unit length, and the viewpoint head must answer
    head = view_head.load(Path(data_cfg["checkpoint_dir"]) / cfg["view"]["head"])
    from reid.embed import embed_boxes
    emb, tokens = view_head.mean_patch_tokens(
        encoder, lambda: embed_boxes(encoder, rows.iloc[:8], images_dir, input_hw, mean, std, max_side,
                                     batch_size=8, num_workers=0, device=device))
    norms = np.linalg.norm(emb, axis=1)
    report["forward"] = {"dimension": int(emb.shape[1]), "finite": bool(np.isfinite(emb).all()),
                         "unit_norm": bool(np.allclose(norms, 1, atol=1e-3)),
                         "views": sorted(set(view_head.views(tokens, head).tolist()))}

    report["verdict"] = {
        "latency": f"{latency_ms:.2f} ms vs {LATENCY_BUDGET_MS:.0f} ms budget",
        "fps": f"{best:.1f} vs {FPS_BUDGET:.0f} budget",
        "weights": f"{report['weights']['total_mb']:.0f} MB vs {WEIGHTS_BUDGET_MB:.0f} MB budget",
        "passed": bool(latency_ms <= LATENCY_BUDGET_MS and best >= FPS_BUDGET
                       and report["weights"]["within_budget"] and report["forward"]["unit_norm"]),
    }
    print(json.dumps({k: report[k] for k in ("load_seconds", "frames", "speed", "forward", "verdict")}, indent=2),
          flush=True)
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"written to {out}", flush=True)
    except OSError:
        pass
    if not report["verdict"]["passed"] and not real:
        print("note: painted frames decode differently from photographs, so treat FPS as a sanity check", flush=True)


if __name__ == "__main__":
    main()
