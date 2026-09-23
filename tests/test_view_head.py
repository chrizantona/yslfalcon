"""The saved viewpoint head and the penalty it applies at submission time (EXP-208)."""
import numpy as np
import pytest

from reid.view_head import VIEWS, load, penalty, save, views


def test_head_round_trip(tmp_path):
    weight, bias = np.arange(6, dtype=np.float32).reshape(3, 2), np.zeros(3, np.float32)
    path = tmp_path / "head.npz"
    save(path, weight, bias, {"folds": [0, 1]})
    head = load(path)
    assert np.array_equal(head["weight"], weight) and head["meta"]["folds"] == [0, 1]


def test_views_pick_the_largest_logit():
    head = {"weight": np.eye(3, 2, dtype=np.float32), "bias": np.zeros(3, np.float32)}
    assert list(views(np.array([[5.0, 0.0], [0.0, 5.0]], np.float32), head)) == [VIEWS[0], VIEWS[1]]


def test_penalty_hits_only_matching_views():
    p = penalty(np.array(["front", "back"]), np.array(["front", "side", "back"]), 0.07)
    assert np.allclose(p, [[0.07, 0, 0], [0, 0, 0.07]])


def build(penalty_matrix, max_candidates=1, threshold=-10.0):
    from reid.submit import build_submission
    query = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    gallery = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], np.float32)
    return build_submission(query, gallery, np.array(["q0", "q1"]), np.array(["g0", "g1", "g2"]),
                            threshold, max_candidates, penalty_matrix)


def test_the_penalty_reorders_the_list_but_not_the_confidence():
    plain_sub, plain_cand = build(None)
    penalised_sub, penalised_cand = build(np.array([[0.5, 0, 0], [0, 0, 0.5]], np.float32))
    assert plain_sub.gallery_id_1.tolist() == ["g0", "g2"]          # the raw similarity puts the twin first
    assert penalised_sub.gallery_id_1.tolist() == ["g1", "g1"]      # the penalty demotes the same-view candidate
    assert plain_cand.confidence.tolist() == penalised_cand.confidence.tolist()   # refusal is decided before it


def test_a_refused_query_leaves_no_row():
    _, candidates = build(None, threshold=1.0)                      # top-1 z-scores are 0.926 and 1.136
    assert candidates.query_id.tolist() == ["q1"]                   # only the confident query is answered


def test_several_candidates_with_a_penalty_are_refused():
    with pytest.raises(ValueError):
        build(np.zeros((2, 3), np.float32), max_candidates=3)
