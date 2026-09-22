import numpy as np

from reid.sam_masks import FILL, apply_mask


def test_everything_outside_the_grown_mask_is_grey_and_the_vehicle_is_untouched():
    frame = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[20:40, 30:50] = True
    out = apply_mask(frame, mask, (25, 15, 30, 30), grow=0.1)  # 3 px of dilation
    assert (out[20:40, 30:50] == frame[20:40, 30:50]).all()
    assert (out[18:42, 28:52] == frame[18:42, 28:52]).all()    # the dilated rim survives
    assert (out[:10] == FILL).all() and (out[:, :20] == FILL).all()


def test_an_empty_mask_leaves_a_grey_frame():
    frame = np.full((10, 10, 3), 7, np.uint8)
    assert (apply_mask(frame, np.zeros((10, 10), bool), (0, 0, 10, 10)) == FILL).all()


def test_the_mask_that_covers_the_box_centre_wins_over_a_better_scored_one():
    from reid.sam_masks import pick_mask
    masks = np.zeros((3, 10, 10), bool)
    masks[0, :, :4] = True   # the neighbour's car: best score, misses the centre
    masks[1, 3:8, 3:8] = True
    masks[2, 2:9, 2:9] = True
    assert pick_mask(masks, [0.99, 0.90, 0.95], (5, 5)) == 2
    assert pick_mask(masks, [0.99, 0.90, 0.95], (9, 0)) == 0  # nothing covers the point: the best score
