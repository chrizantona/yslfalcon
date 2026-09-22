import cv2
import numpy as np
import pandas as pd

from reid.background import paste, vehicle_mask
from reid.sam_masks import FILL
from reid.train import CropDataset


def test_the_mask_is_the_vehicle_even_where_the_vehicle_itself_is_grey():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    crop[10:30, 20:40] = FILL                                   # a grey door must stay part of the vehicle
    masked = np.full_like(crop, FILL)
    masked[5:35, 15:45] = crop[5:35, 15:45]
    mask = vehicle_mask(crop, masked, feather=0)
    assert mask[5:35, 15:45].all()
    assert mask[:3].sum() < 0.05 * mask[:3].size                # background: grey in the twin, not in the crop


def test_paste_keeps_the_vehicle_and_replaces_the_rest():
    crop = np.full((20, 30, 3), 200, np.uint8)
    mask = np.zeros((20, 30), np.float32)
    mask[5:15, 10:20] = 1
    out = paste(crop, mask, np.full((50, 50, 3), 10, np.uint8))
    assert out.shape == crop.shape
    assert (out[5:15, 10:20] == 200).all() and (out[:5] == 10).all()


def test_the_dataset_swaps_backgrounds_only_when_asked(tmp_path):
    crops, masks, bank = tmp_path / "crops", tmp_path / "masks", tmp_path / "bank"
    for d in (crops, masks, bank):
        d.mkdir()
    cv2.imwrite(str(crops / "a.jpg"), np.full((32, 32, 3), 250, np.uint8))
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(masks / "a.png"), mask)
    cv2.imwrite(str(bank / "00000.jpg"), np.zeros((64, 64, 3), np.uint8))
    df = pd.DataFrame({"image_id": ["a"], "label": [0]})
    cfg = {"flip_p": 0, "pad": 0, "erasing_p": 0}
    plain = CropDataset(df, crops, (32, 32), (0.5,) * 3, (0.5,) * 3, cfg)[0][0]
    swapped = CropDataset(df, crops, (32, 32), (0.5,) * 3, (0.5,) * 3, cfg,
                          {"p": 1.0, "masks_dir": masks, "bank_dir": bank})[0][0]
    assert swapped[:, 16, 16].sub(plain[:, 16, 16]).abs().max() < 0.05  # the vehicle is untouched
    assert (swapped[:, 1, 1] < plain[:, 1, 1] - 1).all()                # the corner is now the dark background
