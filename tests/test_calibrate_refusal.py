import numpy as np
import pandas as pd

from reid.calibrate_refusal import fit, per_query, plateau, score


def test_junk_on_top_counts_as_right_as_in_the_organisers_script():
    # one query of vehicle 1 from camera 10; its top gallery item is the same vehicle from the same camera (junk)
    query = pd.DataFrame({"vehicle_id": [1], "camera_id": [10]})
    gallery = pd.DataFrame({"vehicle_id": [1, 1, 2], "camera_id": [10, 11, 10]})
    similarity = np.array([[0.9, 0.8, 0.1]])
    conf, has_pair, right = per_query(similarity, np.array([5.0]), query, gallery)
    assert has_pair[0] and right[0]  # a cross-camera pair exists, and a same-vehicle top is right even if it is junk


def test_score_counts_like_evaluate_py():
    conf = np.array([9.0, 9.0, 1.0, 9.0, 1.0])
    has_pair = np.array([True, True, True, False, False])
    right = np.array([True, False, True, False, False])
    # answered at 5: q0 TP, q1 FP (wrong top), q3 FP (no pair); refused: q2 FN, q4 TN
    total, f1, tnr = score(conf, has_pair, right, threshold=5.0)
    assert np.isclose(f1, 2 * 1 / (2 * 1 + 2 + 1))
    assert np.isclose(tnr, 1 / 2)
    assert np.isclose(total, 0.7 * f1 + 0.3 * tnr)


def test_fit_finds_the_best_threshold_and_plateau_measures_its_flatness():
    conf = np.array([1.0, 2.0, 3.0, 4.0])
    has_pair = np.array([False, False, True, True])
    right = np.array([False, False, True, True])
    t, best, grid, scores = fit(conf, has_pair, right)
    assert t == 3.0 and np.isclose(best, 1.0)          # answer exactly the two queries with a pair
    width, centre = plateau(grid, scores, 1e-9)
    assert width == 0.0 and centre == 3.0
