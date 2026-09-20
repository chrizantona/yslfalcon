# EXP-172 artifacts

Полный результат обучения SigLIP 2 Giant на официальном VehicleID train.

- `seed0/` — train log, метрики, float32 query/gallery embeddings, AP каждого запроса, кривая и порог отказа.
- `summary.json`, `slices.csv`, `benchmark.json` — итоговая агрегация, EDA-срезы и H200 FP16 benchmark.
- `pod_logs/`, `pod_status/` — исходные логи и маркеры стадий автономного запуска.
- `checkpoints/63bd5755271548778ed71b5f688f28dc_seed0.pt.part-*` — Git LFS-части checkpoint.
- `MANIFEST.sha256` — контрольные суммы опубликованных файлов.

Восстановление checkpoint после `git lfs pull`:

```bash
bash scripts/reassemble_exp172_checkpoint.sh
```

Ожидаемый размер: `4735720473` байта. Ожидаемый SHA-256:
`2997b65c29f73b21a00d056bf6d8dccf90f8a553e633ea31a68f0a70b5b8d0b1`.
