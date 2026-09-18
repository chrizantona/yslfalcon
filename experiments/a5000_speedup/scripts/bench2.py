"""One-off: reid.benchmark numbers for EXP-043 with the candidate speedups, not part of the repo.

current_prefetch: current encoder; the throughput loop loads the next batch while the GPU runs.
optimised: + fp16 Linear/Conv weights under the same autocast + one CUDA graph per input shape.
Both are compared with the current encoder on 256 crops (max abs difference of the L2-normalised embeddings).
"""
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import torch
import torch.nn.functional as F

from reid.benchmark import InferencePrecision, performance_points, single_vehicle_latency
from reid.data import crop_box, load_csv, load_yaml, to_model_input
from reid.inference import load_trained_encoder


class Optimised(torch.nn.Module):
    """fp16 Linear/Conv weights (autocast casts them to fp16 on every call anyway) and a CUDA graph per input shape."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        for m in self.encoder.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                m.half()
        self.graphs = {}

    def eager(self, x):
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return self.encoder(x)

    @torch.no_grad()
    def forward(self, x):
        key = tuple(x.shape)
        if key not in self.graphs:
            static = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.eager(static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.eager(static)
            self.graphs[key] = (g, static, out)
        g, static, out = self.graphs[key]
        static.copy_(x)
        g.replay()
        return out.clone()


@torch.no_grad()
def prefetch_throughput(encoder, rows, images_dir, input_hw, mean, std, max_side, batch_size, threads=8,
                        warmup_batches=3, seconds=10.0):
    def load(r):
        return to_model_input(crop_box(cv2.imread(str(images_dir / f"{r.image_id}.jpg")), (r.x, r.y, r.w, r.h),
                                       max_side), input_hw, mean, std)

    with ThreadPoolExecutor(threads) as pool:
        def submit(k):
            idx = [(k * batch_size + j) % len(rows) for j in range(batch_size)]
            return [pool.submit(load, r) for r in rows.iloc[idx].itertuples()]
        pending, k, processed, start = submit(0), 0, 0, None
        while start is None or time.perf_counter() - start < seconds:
            x = torch.stack([f.result() for f in pending])
            pending = submit(k + 1)
            F.normalize(encoder(x.to("cuda")).float(), dim=-1).cpu()
            torch.cuda.synchronize()
            k += 1
            if k == warmup_batches:
                start = time.perf_counter()
            elif k > warmup_batches:
                processed += len(x)
    return processed / (time.perf_counter() - start)


cfg = load_yaml("configs/exp043_soup_exp042_seeds.yaml")
data_cfg = load_yaml("/root/bench_data.yaml")
enc, dim, mean, std = load_trained_encoder(cfg["model"], "/root/speed_ckpt/siglip2.pt", data_cfg["external_weights_dir"])
for p in enc.parameters():
    p.requires_grad_(False)
current = InferencePrecision(enc, "fp16", "cuda").eval().cuda()
optimised = Optimised(enc.cuda()).eval()
rows = load_csv(Path(data_cfg["dataset_dir"]), "query").iloc[:1536]
images = Path(data_cfg["dataset_dir"]) / "images"
hw, max_side = cfg["model"]["input_hw"], data_cfg["crop_max_side"]

x = torch.stack([to_model_input(crop_box(cv2.imread(str(images / f"{r.image_id}.jpg")), (r.x, r.y, r.w, r.h), max_side),
                                hw, mean, std) for r in rows.iloc[:256].itertuples()])
with torch.no_grad():
    ref = torch.cat([F.normalize(current(x[i:i + 32].cuda()).float(), dim=-1).cpu() for i in range(0, 256, 32)])
    opt = torch.cat([F.normalize(optimised(x[i:i + 32].cuda()).float(), dim=-1).cpu() for i in range(0, 256, 32)])
    single = torch.cat([F.normalize(optimised(x[i:i + 1].cuda()).float(), dim=-1).cpu() for i in range(0, 256)])
out = {"max_abs_diff_optimised_vs_current_bs32": float((opt - ref).abs().max()),
       "max_abs_diff_optimised_bs1_vs_current_bs32": float((single - ref).abs().max())}
print(out, flush=True)

for name, encoder in (("current_prefetch", current), ("optimised", optimised)):
    lat, fwd = single_vehicle_latency(encoder, rows, images, hw, mean, std, max_side, "cuda")
    res = {"latency_bs1_ms": round(lat, 2), "forward_bs1_ms": round(fwd, 2)}
    for b in (1, 8, 16, 32):
        res[f"fps_bs{b}"] = round(prefetch_throughput(encoder, rows, images, hw, mean, std, max_side, b), 1)
    res["fps_best"] = max(res[f"fps_bs{b}"] for b in (1, 8, 16, 32))
    res["performance_points"] = performance_points(res["latency_bs1_ms"], res["fps_best"])
    res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
    out[name] = res
    print(name, res, flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
