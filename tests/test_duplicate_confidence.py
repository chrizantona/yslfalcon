"""The duplicate-aware confidence and the scenario-B construction of EXP-207."""
import numpy as np
import pandas as pd

from reid.duplicate_confidence import duplicate_aware, refusal, same_camera_only


def test_duplicate_aware_ignores_the_near_duplicate():
    sim = np.array([[0.99, 0.60, 0.10], [0.85, 0.20, 0.05]])
    plain = duplicate_aware(sim, None)
    aware = duplicate_aware(sim, 0.97)
    assert plain[0] > aware[0]                      # the first query loses its duplicate and drops
    assert plain[1] == aware[1]                     # the second has none and is untouched
    assert np.isfinite(aware).all()


def test_duplicate_aware_falls_back_when_everything_is_a_duplicate():
    sim = np.array([[0.99, 0.98]])
    assert duplicate_aware(sim, 0.97) == duplicate_aware(sim, None)


def test_same_camera_only_leaves_the_no_match_vehicle_without_a_cross_camera_frame():
    query = pd.DataFrame({"image_id": range(4), "vehicle_id": [1, 1, 2, 2], "camera_id": [10, 11, 10, 10]})
    gallery = pd.DataFrame({"image_id": range(4, 8), "vehicle_id": [1, 1, 2, 2], "camera_id": [10, 11, 10, 11]})
    keep_q, keep_g = same_camera_only(query, gallery, [2])
    assert keep_q.all()                             # both queries of vehicle 2 are from camera 10 and stay
    assert list(gallery[keep_g].camera_id) == [10, 11, 10]   # vehicle 2 keeps only camera 10
    kept = gallery[keep_g]
    pair = (kept.vehicle_id == 2) & (kept.camera_id != 10)
    assert not pair.any()


def test_refusal_counts_as_the_organisers_do():
    conf = np.array([9.0, 5.0, 9.0, 5.0])
    has_pair = np.array([True, True, False, False])
    right = np.array([True, False, False, False])
    r = refusal(conf, has_pair, right, 7.0)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["TNR"] == 0.5
