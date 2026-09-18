"""One-off profile of the reid.benchmark cycle for EXP-043 (SigLIP 2, fp16) on the rented A5000; not part of the repo.

A: stage split of the single-vehicle cycle exactly as reid.benchmark runs it.
B: forward variants at batch 1 and 32 - current autocast, fp16 Linear/Conv weights under the same autocast,
   the latter captured in a CUDA graph; max abs difference of the embeddings against the current path.
C: batch-32 throughput - current (load then forward) against loading the next batch while the GPU runs.
D: decode alternatives for one 1920x1080 frame.
"""
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.benchmark import InferencePrecision
from reid.data import crop_box, load_csv, load_yaml, to_model_input
from reid.inference import load_trained_encoder

perf = time.perf_counter
out = {}
cfg = load_yaml("configs/exp043_soup_exp042_seeds.yaml")
data_cfg = load_yaml("/root/bench_data.yaml")
enc, dim, mean, std = load_trained_encoder(cfg["model"], "/root/speed_ckpt/siglip2.pt", data_cfg["external_weights_dir"])
for p in enc.parameters():
    p.requires_grad_(False)
current = InferencePrecision(enc, "fp16", "cuda").eval().cuda()
rows = load_csv(Path(data_cfg["dataset_dir"]), "query").iloc[:1536]
images = Path(data_cfg["dataset_dir"]) / "images"
hw, max_side = cfg["model"]["input_hw"], data_cfg["crop_max_side"]
out["threads"] = {"cv2": cv2.getNumThreads(), "torch": torch.get_num_threads()}
med = lambda v: round(float(np.median(v)) * 1000, 3)


def load(r):
    return to_model_input(crop_box(cv2.imread(str(images / f"{r.image_id}.jpg")), (r.x, r.y, r.w, r.h), max_side),
                          hw, mean, std)


# A: stage split, 50 warm-up + 300 runs as in reid.benchmark
keys = ["read", "decode", "crop", "to_input", "h2d", "forward", "post", "total"]
t = {k: [] for k in keys}
big = 0
with torch.no_grad():
    for i in range(350):
        r = rows.iloc[i % len(rows)]
        torch.cuda.synchronize()
        t0 = perf()
        buf = np.fromfile(str(images / f"{r.image_id}.jpg"), np.uint8)
        t1 = perf()
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        t2 = perf()
        crop = crop_box(frame, (r.x, r.y, r.w, r.h), max_side)
        t3 = perf()
        x = to_model_input(crop, hw, mean, std)[None]
        t4 = perf()
        x = x.to("cuda")
        torch.cuda.synchronize()
        t5 = perf()
        y = current(x)
        torch.cuda.synchronize()
        t6 = perf()
        F.normalize(y.float(), dim=-1).cpu()
        t7 = perf()
        if i >= 50:
            for k, a, b in zip(keys, (t0, t1, t2, t3, t4, t5, t6, t0), (t1, t2, t3, t4, t5, t6, t7, t7)):
                t[k].append(b - a)
            big += max(r.w, r.h) > max_side
out["A_stage_ms"] = {k: med(v) for k, v in t.items()}
out["A_crops_downsized_by_max_side"] = f"{big}/300"
print(json.dumps(out), flush=True)


# B: forward variants
def fp16_weights(model):
    """Copy with Linear/Conv weights and biases in fp16; under autocast these ops cast to fp16 anyway."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            m.half()
    return model


class Autocast(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return self.m(x)


def graphed(model, shape):
    static = torch.zeros(shape, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        static_out = model(static)

    def run(x):
        static.copy_(x)
        g.replay()
        return static_out
    return run


def time_forward(fn, x, n=200):
    wall, gpu = [], []
    with torch.no_grad():
        for i in range(n + 20):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = perf()
            e0.record()
            y = fn(x)
            e1.record()
            torch.cuda.synchronize()
            if i >= 20:
                wall.append(perf() - t0)
                gpu.append(e0.elapsed_time(e1) / 1000)
    return {"wall_ms": med(wall), "gpu_ms": med(gpu)}


half = Autocast(fp16_weights(enc).eval().cuda())
xs = {b: torch.stack([load(r) for r in rows.iloc[:b].itertuples()]).cuda() for b in (1, 32)}
with torch.no_grad():
    ref = {b: F.normalize(current(x).float(), dim=-1) for b, x in xs.items()}
    out["B_forward"] = {}
    for b, x in xs.items():
        g = graphed(half, tuple(x.shape))
        for name, fn in (("autocast_current", current), ("fp16_weights", half), ("fp16_weights_cudagraph", g)):
            res = time_forward(fn, x)
            res["max_abs_diff_vs_current"] = float((F.normalize(fn(x).float(), dim=-1) - ref[b]).abs().max())
            out["B_forward"][f"bs{b}_{name}"] = res
            print(name, b, res, flush=True)
del half, g
torch.cuda.empty_cache()


# C: throughput at batch 32, current loop against prefetching the next batch
def throughput(fn, bs, threads, prefetch, seconds=10.0):
    loads, fwds = [], []
    with ThreadPoolExecutor(threads) as pool, torch.no_grad():
        def submit(k):
            idx = [(k * bs + j) % len(rows) for j in range(bs)]
            return [pool.submit(load, r) for r in rows.iloc[idx].itertuples()]
        pending, k, n, start = submit(0), 0, 0, None
        while start is None or perf() - start < seconds:
            t0 = perf()
            x = torch.stack([f.result() for f in pending])
            t1 = perf()
            if prefetch:
                pending = submit(k + 1)
            F.normalize(fn(x.cuda()).float(), dim=-1).cpu()
            torch.cuda.synchronize()
            t2 = perf()
            if not prefetch:
                pending = submit(k + 1)
            k += 1
            if k == 3:
                start = perf()
            elif k > 3:
                n += bs
                loads.append(t1 - t0)
                fwds.append(t2 - t1)
        return {"fps": round(n / (perf() - start), 1), "wait_load_ms": med(loads), "gpu_part_ms": med(fwds)}


out["C_throughput_bs32"] = {}
for threads in (8, 16):
    for prefetch in (False, True):
        res = throughput(current, 32, threads, prefetch)
        out["C_throughput_bs32"][f"threads{threads}_prefetch{int(prefetch)}"] = res
        print(threads, prefetch, res, flush=True)

# D: decode alternatives for one frame
r = rows.iloc[0]
buf = np.fromfile(str(images / f"{r.image_id}.jpg"), np.uint8)
dec = {}
for name, flag in (("cv2_full", cv2.IMREAD_COLOR), ("cv2_reduced2", cv2.IMREAD_REDUCED_COLOR_2),
                   ("cv2_reduced4", cv2.IMREAD_REDUCED_COLOR_4)):
    v = []
    for i in range(60):
        t0 = perf()
        cv2.imdecode(buf, flag)
        v.append(perf() - t0)
    dec[name] = med(v[10:])
try:
    from torchvision.io import decode_jpeg
    data = torch.from_numpy(buf)
    v = []
    for i in range(60):
        torch.cuda.synchronize()
        t0 = perf()
        img = decode_jpeg(data, device="cuda")
        torch.cuda.synchronize()
        v.append(perf() - t0)
    dec["nvjpeg_gpu"] = med(v[10:])
except Exception as e:  # noqa: BLE001 - report and keep the rest
    dec["nvjpeg_gpu"] = repr(e)
out["D_decode_ms"] = dec
print(json.dumps(out, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
