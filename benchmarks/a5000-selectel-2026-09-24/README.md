# RTX A5000 inference measurements - Selectel, 2026-09-24

These measurements use the final `emb-so400m-224-v1` model in Docker and the complete competition test archive.
The organiser batch sizes are **1, 8, 16, and 32**. Batch 1 is the single-vehicle latency measurement; steady
throughput is measured for all four sizes and the best FPS is used. On this host, **batch 32 is optimal**.

## Host and container configuration

| Component | Measured configuration |
|---|---|
| GPU | NVIDIA RTX A5000, 24,564 MiB, 230 W |
| GPU driver | 580.65.06 |
| CPU | Intel Xeon Gold 6336Y @ 2.40 GHz |
| vCPU topology | 16 vCPU, 2 sockets x 8 cores, 1 thread/core |
| RAM | 32,092 MiB, no swap |
| Disk | 250 GB QEMU virtual disk |
| OS | Ubuntu 24.04.3 LTS, Linux 6.8.0-85-generic |
| Docker | Engine 28.5.1, Compose 5.5.1 |
| Container | Python 3.13.7, PyTorch 2.10.0+cu126, CUDA runtime 12.6 |
| Model | SigLIP 2 So400m/14 at 224, fp16, embedding 1152 |
| Checkpoint | 1,717,961,753 bytes |

## Five-batch diagnostic

One warm-up batch per size is excluded. The following five batches are averaged. JPEG reading, bbox crop,
preprocessing, host-to-device transfer, forward pass, L2 normalisation, and result transfer are included. CPU work
for the next batch overlaps the current GPU forward, matching the production pipeline.

| Batch | Mean batch time | Per image | End-to-end FPS | Forward | Peak allocated VRAM | Peak reserved VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.43 ms | 20.43 ms | 48.9 | 6.81 ms | 889.6 MB | 956.3 MB |
| 8 | 49.08 ms | 6.14 ms | 163.0 | 37.12 ms | 975.8 MB | 1,069.5 MB |
| 16 | 100.41 ms | 6.28 ms | 159.3 | 65.55 ms | 1,054.4 MB | 1,170.2 MB |
| **32** | **135.61 ms** | **4.24 ms** | **236.0** | **129.82 ms** | **1,210.9 MB** | **1,554.0 MB** |

Peak process RAM while loading the model was **6,669.3 MB**; steady RSS reached **1,971.4 MB**. Raw per-batch
measurements are in [`results/batch_metrics.json`](results/batch_metrics.json), with the summary in
[`results/batch_metrics.csv`](results/batch_metrics.csv).

This short five-batch diagnostic is useful for resource checks. The organiser-style benchmark below runs each
throughput point for at least ten seconds and is the score-facing result.

## Organiser-style benchmark: three complete repeats

The benchmark performs 50 warm-ups and 300 batch-1 latency measurements. Throughput runs for at least ten seconds
at each batch size with CPU/GPU prefetch overlap.

| Metric | Run 1 | Run 2 | Run 3 | Reported median |
|---|---:|---:|---:|---:|
| Batch-1 latency | 27.32 ms | 25.61 ms | 24.86 ms | **25.61 ms** |
| FPS batch 1 | 43.9 | 43.7 | 44.1 | **43.9** |
| FPS batch 8 | 137.6 | 143.0 | 135.5 | **137.6** |
| FPS batch 16 | 186.6 | 194.0 | 182.2 | **186.6** |
| FPS batch 32 | 209.0 | 214.2 | 212.2 | **212.2** |
| Peak VRAM | 1,211 MB | 1,211 MB | 1,211 MB | **1,211 MB** |

The final model gets the full performance score: latency is below 40 ms, best throughput is above 100 FPS, and the
1,718 MB checkpoint is below the 2 GB limit. Embeddings were deterministic in every run (`max_abs_diff = 0.0`).

## Full test inference

The complete 1,110-query plus 750-gallery run took **63 seconds**, including container startup and model loading.
It produced all three submission artifacts, refused 62 queries (5.6%), and passed `reid.check_submission` with
`submission OK`. Details and artifact checksums are in [`results/full_inference.json`](results/full_inference.json).

## Reproduce from a clean checkout

Requirements: Linux, Docker with Compose/BuildKit, NVIDIA driver, and NVIDIA Container Toolkit.

```bash
chmod +x benchmarks/a5000-selectel-2026-09-24/run.sh
./benchmarks/a5000-selectel-2026-09-24/run.sh \
  --dataset /absolute/path/to/dataset \
  --batch-sizes "1 8 16 32" \
  --warmup-batches 1 \
  --measured-batches 5 \
  --workers 8
```

The dataset directory must contain `images/`, `test_query.csv`, and `test_gallery.csv`. The script builds the final
offline image from the repository, verifies the pinned public weights during the build, runs the short batch test,
runs full test inference, validates submission artifacts, and writes everything under this benchmark's `output/`
directory. Use `--image IMAGE` to reuse a compatible image built from the repository, or `--skip-full-inference` to
collect only batch measurements.

Published service images used for the independent service smoke test:

- `docker.io/timuyr/falcon-api:6e33a90`
- `docker.io/timuyr/falcon-web:6e33a90`
- `docker.io/timuyr/falcon-runtime:6e33a90`
