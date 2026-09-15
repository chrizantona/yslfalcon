import numpy as np
import pandas as pd

from reid.metrics import paired_bootstrap, refusal_curve, reid_eval
from reid.protocol import assign_folds, pseudo_test
from reid.submit import build_submission


def test_perfect_embeddings_score_one():
    q_pid, q_cam = np.array([0, 1]), np.array([0, 0])
    g_pid, g_cam = np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0])
    sim = (q_pid[:, None] == g_pid[None, :]).astype(float)
    summary, _ = reid_eval(sim, q_pid, q_cam, g_pid, g_cam)
    assert summary["mAP"] == 1.0 and summary["R1"] == 1.0


def test_same_camera_match_is_ignored():
    q_pid, q_cam = np.array([7]), np.array([1])
    g_pid, g_cam = np.array([7, 7, 3]), np.array([1, 2, 1])  # same-camera twin ranked first, cross-camera match second
    summary, _ = reid_eval(np.array([[0.9, 0.8, 0.1]]), q_pid, q_cam, g_pid, g_cam)
    assert summary["mAP"] == 1.0 and summary["R1"] == 1.0 and summary["junk_in_top10"] == 1.0


def test_negative_above_the_match_halves_ap():
    q_pid, q_cam = np.array([7]), np.array([1])
    g_pid, g_cam = np.array([3, 7]), np.array([2, 2])
    summary, _ = reid_eval(np.array([[0.9, 0.8]]), q_pid, q_cam, g_pid, g_cam)
    assert summary["mAP"] == 0.5 and summary["R1"] == 0.0


def test_refusal_curve_separates_clear_matches_from_queries_without_match():
    rng = np.random.default_rng(0)
    g_pid, g_cam = np.arange(50), np.ones(50, int)
    q_pid, q_cam = np.array([0, 1, 100, 101]), np.zeros(4, int)  # vehicles 100 and 101 are not in the gallery
    sim = rng.uniform(0, 0.3, (4, 50))
    sim[0, 0] = sim[1, 1] = 1.0
    best = refusal_curve(sim, q_pid, q_cam, g_pid, g_cam).sort_values("F1").iloc[-1]
    assert best.F1 == 1.0 and best.TNR == 1.0


def test_paired_bootstrap_of_identical_models_is_zero():
    ap = np.array([0.2, 0.5, 0.9, 0.4])
    assert paired_bootstrap(ap, ap.copy(), [1, 1, 2, 3]) == (0.0, 0.0, 0.0)


def test_folds_group_vehicles_and_pseudo_test_follows_the_recipe():
    rows = [(f"{v}_{c}_{i}", v, c) for v in range(40) for c in (v % 7, 89 if v % 3 == 0 else 20) for i in range(2)]
    train = pd.DataFrame(rows, columns=["image_id", "vehicle_id", "camera_id"])
    folds = assign_folds(train, 5, seed=42)
    assert (train.assign(fold=folds).groupby("vehicle_id").fold.nunique() == 1).all()

    fold_df = train[folds == 0]
    query, gallery = pseudo_test(fold_df, seed=0)
    assert (gallery.groupby(["vehicle_id", "camera_id"]).size() == 1).all()
    assert len(query) + len(gallery) == len(fold_df)
    assert set(query.image_id).isdisjoint(gallery.image_id)


def test_build_submission_ranks_and_refuses():
    gallery_emb = np.eye(30, dtype=np.float32)
    query_emb = np.stack([gallery_emb[5], np.full(30, 30 ** -0.5, np.float32)])  # an exact match and a flat query
    gallery_ids = np.array([f"g{i}" for i in range(30)])
    submission, candidates = build_submission(query_emb, gallery_emb, np.array(["q0", "q1"]), gallery_ids, threshold=3.5)
    assert submission.shape == (2, 11) and submission.loc[0, "gallery_id_1"] == "g5"
    assert candidates[candidates.query_id == "q0"].gallery_id.tolist() == ["g5"]
    assert candidates[candidates.query_id == "q1"].gallery_id.tolist() == [""]
