"""One-off: new InferencePrecision (fp16 weights + CUDA graphs) against plain autocast on the SigLIP 2 speed checkpoint,
through embed_boxes as reid.submit calls it (DataLoader, 8 workers, batch 256 with a partial last batch) and at batch 1."""
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from reid.benchmark import InferencePrecision
from reid.data import load_csv, load_yaml
from reid.embed import embed_boxes
from reid.inference import load_trained_encoder


class PlainAutocast(torch.nn.Module):
    """The wrapper before 18.09.2026."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.encoder(x)


cfg = load_yaml("configs/exp043_soup_exp042_seeds.yaml")
data_cfg = load_yaml("/root/bench_data.yaml")
enc, _, mean, std = load_trained_encoder(cfg["model"], "/root/speed_ckpt/siglip2.pt", data_cfg["external_weights_dir"])
old = PlainAutocast(copy.deepcopy(enc)).eval().cuda()
new = InferencePrecision(enc, "fp16", "cuda").eval().to("cuda")
rows = load_csv(Path(data_cfg["dataset_dir"]), "query")
images = Path(data_cfg["dataset_dir"]) / "images"
args = dict(input_hw=cfg["model"]["input_hw"], mean=mean, std=std, max_side=data_cfg["crop_max_side"],
            batch_size=256, num_workers=8, device="cuda")
out = {"rows": len(rows)}
for name, model in (("old", old), ("new", new), ("new_again", new)):
    torch.cuda.synchronize()
    start = time.perf_counter()
    emb = embed_boxes(model, rows, images, **args)
    torch.cuda.synchronize()
    out[f"{name}_submit_path_img_per_s"] = round(len(rows) / (time.perf_counter() - start), 1)
    out[name] = emb
out["max_abs_diff_submit_path"] = float(abs(out.pop("old") - out["new"]).max())
out["max_abs_diff_submit_path_rerun"] = float(abs(out.pop("new") - out.pop("new_again")).max())
x = torch.randn(64, 3, 224, 224, device="cuda")
with torch.no_grad():
    single_old = torch.cat([F.normalize(old(x[i:i + 1]).float(), dim=-1) for i in range(64)])
    single_new = torch.cat([F.normalize(new(x[i:i + 1]).float(), dim=-1) for i in range(64)])
out["max_abs_diff_batch1"] = float((single_old - single_new).abs().max())
out["graphs"] = [list(k[0]) for k in new.graphs]
print(json.dumps(out, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
