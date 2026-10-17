import json
import sys

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


def test_fit_keeps_the_higher_threshold_on_a_score_tie():
    conf = np.array([1.0, 2.0, 3.0])
    has_pair = np.array([True, True, True])
    right = np.array([False, False, False])
    t, best, grid, scores = fit(conf, has_pair, right)
    assert t == 3.0 and best == 0.0
    np.testing.assert_array_equal(grid, conf)
    np.testing.assert_array_equal(scores, np.zeros(3))


def test_calibration_report_adds_official_diagnostics_without_changing_the_fit(monkeypatch, tmp_path):
    from reid import calibrate_refusal

    query = pd.DataFrame({"vehicle_id": [1, 2, 3, 4, 5], "camera_id": [10] * 5})
    gallery = pd.DataFrame({"vehicle_id": [1, 2, 3, 9], "camera_id": [11] * 4})
    similarity = np.array([[0.9, 0.1, 0.2, 0.0], [0.1, 0.8, 0.2, 0.9], [0.1, 0.2, 0.9, 0.0],
                           [0.1, 0.2, 0.0, 0.9], [0.1, 0.2, 0.0, 0.9]])
    confidence = np.array([5.0, 9.0, 1.0, 9.0, 1.0])
    config = {"eval_fold": 0, "exp_id": "synthetic", "splits": 1, "seed": 0,
              "candidates": {"synthetic": {"threshold": 5.0}}}
    configs = {"data": {}, "val": {"out_dir": "protocol"}, "calibration": config}
    monkeypatch.setattr(calibrate_refusal, "load_yaml", configs.__getitem__)
    monkeypatch.setattr(calibrate_refusal, "pick_device", lambda: "cpu")
    monkeypatch.setattr(calibrate_refusal, "load_fold", lambda *args: (query, gallery, np.array([], dtype=int)))
    monkeypatch.setattr(calibrate_refusal, "realistic", lambda *args: (similarity, confidence, None, None))
    monkeypatch.setattr(sys, "argv", ["calibrate_refusal", "--data-config", "data", "--val-config", "val",
                                     "--config", "calibration"])
    monkeypatch.chdir(tmp_path)
    calibrate_refusal.main()

    report = json.loads((tmp_path / "artifacts/synthetic/summary.json").read_text())["synthetic"]
    for section in ("old_threshold", "refitted"):
        row = report[section]
        assert row["threshold"] == 5.0
        assert (row["TP_official"], row["FP_official"], row["FN_official"], row["TN_official"]) == (1, 2, 1, 1)
        assert np.isclose(row["precision_official"], 1 / 3) and np.isclose(row["recall_official"], 1 / 2)
        assert row["answered_fraction_official"] == 3 / 5
        assert row["F1"] == 2 / 5 and row["TNR"] == 1 / 2
        assert row["refusal"] == 0.7 * (2 / 5) + 0.3 * (1 / 2)
