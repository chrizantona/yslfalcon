import numpy as np
import pandas as pd

from reid.metrics import refusal_curve, zscore
from reid.refusal_confidence import confidences


def test_top1_z_confidence_reproduces_the_default_curve():
    rng = np.random.default_rng(0)
    sim = rng.normal(size=(12, 30))
    q_pid, q_cam = rng.integers(0, 6, 12), rng.integers(0, 3, 12)
    g_pid, g_cam = rng.integers(0, 6, 30), rng.integers(0, 3, 30)
    default = refusal_curve(sim, q_pid, q_cam, g_pid, g_cam)
    explicit = refusal_curve(sim, q_pid, q_cam, g_pid, g_cam, confidence=confidences(sim, g_pid)["z_top1"])
    pd.testing.assert_frame_equal(default, explicit)


def test_margins_are_taken_against_the_best_other_vehicle():
    sim = np.array([[0.9, 0.8, 0.5, 0.1]])  # the second item is the same vehicle as the top-1
    g_pid = np.array([7, 7, 3, 4])
    conf = confidences(sim, g_pid)
    assert np.isclose(conf["cos_top1"][0], 0.9) and np.isclose(conf["cos_margin"][0], 0.9 - 0.5)
    z = zscore(sim)[0]
    assert np.isclose(conf["z_top1"][0], z[0]) and np.isclose(conf["z_margin"][0], z[0] - z[2])
