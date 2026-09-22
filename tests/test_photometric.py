import numpy as np

from reid.photometric import night_cast


def test_zero_strength_and_unit_gamma_leave_the_frame_alone():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert np.abs(night_cast(frame, 1.0, 0.0).astype(int) - frame.astype(int)).max() <= 1


def test_a_full_cast_turns_grey_into_amber_and_keeps_luminance_order():
    grey = np.full((4, 4, 3), 128, np.uint8)
    lit = night_cast(grey, 1.0, 1.0)
    assert lit[..., 2].mean() > lit[..., 1].mean() > lit[..., 0].mean()   # BGR: red above green above blue
    dark, bright = night_cast(np.full((4, 4, 3), 60, np.uint8), 1.0, 1.0), night_cast(grey, 1.0, 1.0)
    assert dark.mean() < bright.mean()


def test_gamma_below_one_brightens():
    frame = np.full((4, 4, 3), 100, np.uint8)
    assert night_cast(frame, 0.6, 0.0).mean() > frame.mean()
