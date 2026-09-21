"""A copy of the dataset with everything but the vehicle painted grey, for training and evaluating on masked crops.

SAM segments each annotated box (the box is the prompt), the mask is dilated a little so edges survive, and every pixel
outside it takes the grey that SigLIP normalisation (mean 0.5) maps to zero. Frames keep their size and names, so the
copy serves configs/data_sam.yaml with the usual crops and embeddings code.

Runs in the pod's system python, which has transformers and the SAM weights in the HF cache; one shard per GPU:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 -m reid.sam_masks --data-config configs/data.yaml \
      --out-dir /dev/shm/lct2026/dataset_sam --shard 0/3
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from reid.data import CSV_FILES, load_csv, load_yaml

FILL = 128


def apply_mask(frame, mask, box, grow=0.015, fill=FILL):
    """frame HxWx3 and mask HxW bool -> the frame with every pixel outside the mask, dilated by grow of the box's
    longer side, set to fill."""
    r = max(1, round(grow * max(box[2], box[3])))
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    out = np.full_like(frame, fill)
    out[grown] = frame[grown]
    return out


def main():
    import torch
    from transformers import SamModel, SamProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="facebook/sam-vit-huge")
    parser.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th frame starting at k")
    args = parser.parse_args()
    src = Path(load_yaml(args.data_config)["dataset_dir"])
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if k == 0:
        for name in CSV_FILES.values():
            shutil.copy(src / name, out / name)
    rows = pd.concat([load_csv(src, split) for split in CSV_FILES]).drop_duplicates("image_id").iloc[k::n]

    processor = SamProcessor.from_pretrained(args.model, local_files_only=True)
    model = SamModel.from_pretrained(args.model, local_files_only=True).to("cuda").eval()
    stats = []
    for r in rows.itertuples():
        path = out / "images" / f"{r.image_id}.jpg"
        if path.exists():
            continue
        frame = cv2.imread(str(src / "images" / f"{r.image_id}.jpg"))
        box = (int(r.x), int(r.y), int(r.w), int(r.h))
        inputs = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           input_boxes=[[[box[0], box[1], box[0] + box[2], box[1] + box[3]]]],
                           return_tensors="pt").to("cuda")
        with torch.no_grad():
            pred = model(**inputs, multimask_output=False)
        mask = processor.post_process_masks(pred.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                            inputs["reshaped_input_sizes"].cpu())[0][0, 0].numpy()
        x0, y0 = max(box[0], 0), max(box[1], 0)
        inside = mask[y0:box[1] + box[3], x0:box[0] + box[2]]
        stats.append((r.image_id, float(pred.iou_scores.flatten()[0]), float(inside.mean()) if inside.size else 0.0))
        tmp = str(path)[: -len(".jpg")] + ".tmp.jpg"
        cv2.imwrite(tmp, apply_mask(frame, mask, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp, path)  # no half-written frames if the job is killed
    pd.DataFrame(stats, columns=["image_id", "sam_iou", "box_fraction"]).to_csv(out / f"sam_stats_{k}.csv", index=False)
    print(f"shard {k}/{n}: {len(stats)} frames masked", flush=True)


if __name__ == "__main__":
    main()
