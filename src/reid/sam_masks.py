"""A copy of the dataset with everything but the vehicle painted grey, for training and evaluating on masked crops.

SAM segments each annotated box (the box is the prompt), the mask is dilated a little so edges survive, and every pixel
outside it takes the grey that SigLIP normalisation (mean 0.5) maps to zero. Frames keep their size and names, so the
copy serves configs/data_sam.yaml with the usual crops and embeddings code.

With --center-point the prompt also carries a positive point at the box centre, and of SAM's three candidate masks
the best-scored one that covers that point is kept: when two cars share the box, the box alone often makes SAM pick the
other one, while the centre of an annotated box nearly always lies on the annotated car.

With --external NAME it masks an external pretraining set of vehicle crops instead (data.yaml external.NAME): the
whole crop is the box, its centre the point, and the dilated mask is written as a PNG to external.NAME.masks_dir for
the background-replacement augmentation (reid.background), in batches and bf16 since the set is ~140k crops.

Runs in the pod's system python, which has transformers and the SAM weights in the HF cache; one shard per GPU:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 -m reid.sam_masks --data-config configs/data.yaml \
      --out-dir /dev/shm/lct2026/dataset_sam --shard 0/3
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 -m reid.sam_masks --data-config configs/data.yaml \
      --external veriwild_test --shard 0/3
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


def pick_mask(masks, scores, point):
    """Index of the best-scored mask that covers point = (x, y); the best-scored one overall if none does."""
    x, y = point
    covering = [k for k in range(len(masks)) if masks[k][y, x]]
    return max(covering or range(len(masks)), key=lambda k: float(scores[k]))


def grown_mask(mask, box, grow=0.015):
    """The mask dilated as apply_mask does, as a uint8 0/255 image."""
    r = max(1, round(grow * max(box[2], box[3])))
    return (cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0).astype(np.uint8) * 255


def external_masks(processor, model, spec, k, n, batch=16):
    """Masks of an external set of vehicle crops: the whole crop is the box, its centre the point."""
    import torch

    root, out = Path(spec["dir"]), Path(spec["masks_dir"])
    table = pd.read_csv(root / spec.get("table", "train.csv")).iloc[k::n]
    todo = [i for i in table.image_id if not (out / f"{i}.png").exists()]
    done = 0
    for start in range(0, len(todo), batch):
        ids = todo[start:start + batch]
        images = [cv2.imread(str(root / "images" / f"{i}.jpg")) for i in ids]
        centres = [(im.shape[1] // 2, im.shape[0] // 2) for im in images]
        inputs = processor(images=[cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images],
                           input_boxes=[[[0, 0, im.shape[1], im.shape[0]]] for im in images],
                           input_points=[[[list(c)]] for c in centres], input_labels=[[[1]] for _ in ids],
                           return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(pred.pred_masks.float().cpu(), inputs["original_sizes"].cpu(),
                                             inputs["reshaped_input_sizes"].cpu())
        for j, image_id in enumerate(ids):
            candidates = masks[j][0].numpy()
            mask = candidates[pick_mask(candidates, pred.iou_scores[j].flatten().float().cpu(), centres[j])]
            path = out / f"{image_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(path)[: -len(".png")] + ".tmp.png"
            cv2.imwrite(tmp, grown_mask(mask, (0, 0, images[j].shape[1], images[j].shape[0])))
            os.replace(tmp, path)
        done += len(ids)
    print(f"shard {k}/{n}: {done} crops masked", flush=True)


def main():
    import torch
    from transformers import SamModel, SamProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--out-dir", help="the masked copy of our dataset")
    parser.add_argument("--model", default="facebook/sam-vit-huge")
    parser.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th frame starting at k")
    parser.add_argument("--center-point", action="store_true", help="also prompt with the box centre, see above")
    parser.add_argument("--external", help="mask this external set of crops instead of our dataset, see above")
    args = parser.parse_args()
    data_cfg = load_yaml(args.data_config)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.external:
        processor = SamProcessor.from_pretrained(args.model, local_files_only=True)
        model = SamModel.from_pretrained(args.model, local_files_only=True).to("cuda").eval()
        return external_masks(processor, model, data_cfg["external"][args.external], k, n)
    src = Path(data_cfg["dataset_dir"])
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
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
        centre = (min(max(box[0] + box[2] // 2, 0), frame.shape[1] - 1),
                  min(max(box[1] + box[3] // 2, 0), frame.shape[0] - 1))
        point = {"input_points": [[[list(centre)]]], "input_labels": [[[1]]]} if args.center_point else {}
        inputs = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           input_boxes=[[[box[0], box[1], box[0] + box[2], box[1] + box[3]]]],
                           return_tensors="pt", **point).to("cuda")
        with torch.no_grad():
            pred = model(**inputs, multimask_output=args.center_point)
        masks = processor.post_process_masks(pred.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                             inputs["reshaped_input_sizes"].cpu())[0][0].numpy()
        scores = pred.iou_scores.flatten().cpu()
        best = pick_mask(masks, scores, centre) if args.center_point else 0
        mask = masks[best]
        x0, y0 = max(box[0], 0), max(box[1], 0)
        inside = mask[y0:box[1] + box[3], x0:box[0] + box[2]]
        stats.append((r.image_id, float(scores[best]), float(inside.mean()) if inside.size else 0.0))
        tmp = str(path)[: -len(".jpg")] + ".tmp.jpg"
        cv2.imwrite(tmp, apply_mask(frame, mask, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp, path)  # no half-written frames if the job is killed
    pd.DataFrame(stats, columns=["image_id", "sam_iou", "box_fraction"]).to_csv(out / f"sam_stats_{k}.csv", index=False)
    print(f"shard {k}/{n}: {len(stats)} frames masked", flush=True)


if __name__ == "__main__":
    main()
