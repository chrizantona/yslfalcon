import numpy as np
import pandas as pd
import torch

from reid.data import to_model_input
from reid.metrics import paired_bootstrap, pick_threshold, refusal_curve, reid_eval, reid_eval_top10
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


def test_refusal_scores_queries_by_their_top_candidate_like_the_organisers():
    rng = np.random.default_rng(0)
    g_pid, g_cam = np.arange(50), np.ones(50, int)
    q_pid, q_cam = np.array([0, 1, 2, 100, 101]), np.zeros(5, int)  # 100 and 101 have no pair in the gallery
    sim = rng.uniform(0, 0.3, (5, 50))
    sim[0, 0] = sim[1, 1] = 1.0   # clear correct answers
    sim[2, 7] = 0.99              # confident but wrong: a false positive, not a false negative
    best = pick_threshold(refusal_curve(sim, q_pid, q_cam, g_pid, g_cam))
    # answering 0, 1, 2 and refusing 100, 101: TP 2, FP 1, FN 0 -> F1 = 4/5, TNR = 1, score = 0.86
    assert np.isclose(best.F1, 0.8) and best.TNR == 1.0 and np.isclose(best.score, 0.86)


def test_refusal_skips_a_same_camera_twin_on_top_except_in_the_strict_variant():
    g_pid, g_cam = np.array([0, 0, 5, 6]), np.array([1, 2, 1, 1])
    q_pid, q_cam = np.array([0]), np.array([1])
    curve = refusal_curve(np.array([[0.9, 0.8, 0.1, 0.0]]), q_pid, q_cam, g_pid, g_cam)
    row = curve[curve.answered == 1.0].iloc[0]
    assert row.F1 == 1.0 and row.F1_strict == 0.0


def test_map_at_10_drops_junk_without_backfill_and_divides_by_min_npos_10():
    g_pid = np.array([7] * 12 + [3] * 3)
    g_cam = np.array([1] + [2] * 11 + [2] * 3)  # g0 is the same-camera twin, g1..g11 are 11 cross-camera positives
    sim = np.linspace(1, 0, 15)[None]           # ranking g0, g1, ..., g14
    summary, ap = reid_eval_top10(sim, np.array([7]), np.array([1]), g_pid, g_cam)
    # the top-10 list keeps g1..g9 after dropping g0: 9 hits at ranks 1..9, divided by min(11, 10)
    assert np.isclose(ap[0], 0.9) and summary["R1"] == 1.0

    summary, ap = reid_eval_top10(np.array([[0.5, 0.9]]), np.array([4]), np.array([1]), np.array([7, 3]), np.array([2, 2]))
    assert np.isnan(ap[0]) and summary["n_queries"] == 0  # no positive: excluded, not AP = 0


def test_paired_bootstrap_of_identical_models_is_zero():
    ap = np.array([0.2, 0.5, 0.9, 0.4])
    assert paired_bootstrap(ap, ap.copy(), [1, 1, 2, 3]) == (0.0, 0.0, 0.0)


def test_build_submission_ranks_and_refuses_with_no_rows():
    gallery_emb = np.eye(30, dtype=np.float32)
    query_emb = np.stack([gallery_emb[5], np.full(30, 30 ** -0.5, np.float32)])  # an exact match and a flat query
    gallery_ids = np.array([f"g{i}" for i in range(30)])
    submission, candidates = build_submission(query_emb, gallery_emb, np.array(["q0", "q1"]), gallery_ids, threshold=3.5)
    assert submission.shape == (2, 11) and submission.loc[0, "gallery_id_1"] == "g5"
    assert candidates[candidates.query_id == "q0"].gallery_id.iloc[0] == "g5" and len(candidates[candidates.query_id == "q0"]) == 10
    assert "q1" not in set(candidates.query_id)


def test_to_model_input_keeps_float32_with_plain_tuple_normalisation():
    crop = np.zeros((10, 20, 3), np.uint8)
    x = to_model_input(crop, (8, 8), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    assert x.dtype == torch.float32 and tuple(x.shape) == (3, 8, 8)

