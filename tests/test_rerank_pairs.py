import numpy as np
import pandas as pd
import torch

from reid.rerank_pairs import PairReranker, reorder, take_tokens, window_groups, window_loss


def test_score_is_symmetric_in_the_pair():
    torch.manual_seed(0)
    model = PairReranker(32, width=16, blocks=2, heads=2).eval()
    a, b = torch.randn(4, 6, 32), torch.randn(4, 6, 32)
    with torch.no_grad():
        forward, backward = model(a, b), model(b, a)
    assert torch.allclose(forward, backward, atol=1e-5)  # a pair has one score, whichever side is called the query


def test_reorder_only_permutes_the_window():
    similarity = np.array([[0.9, 0.8, 0.7, 0.1, 0.05]], dtype=np.float32)
    candidates = np.array([[0, 1, 2]])
    scores = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)  # the re-ranker prefers the third candidate
    out = reorder(similarity, candidates, scores, weight=1.0)
    assert np.allclose(np.sort(out[0]), np.sort(similarity[0]))      # the row keeps its own values
    assert np.allclose(out[0, 3:], similarity[0, 3:])                # nothing outside the window moves
    assert out[0].argmax() == 2                                      # the preferred candidate now carries the top value


def test_reorder_with_zero_weight_keeps_the_encoder_order():
    rng = np.random.default_rng(0)
    similarity = rng.random((3, 12)).astype(np.float32)
    candidates = np.argsort(-similarity, axis=1)[:, :5]
    scores = rng.random((3, 5)).astype(np.float32)
    assert np.allclose(reorder(similarity, candidates, scores, weight=0.0), similarity)


def test_take_tokens_keeps_a_subset_and_may_use_the_mirrored_view():
    upright = torch.arange(3 * 8 * 2, dtype=torch.float16).view(3, 8, 2)
    kept = take_tokens([upright, upright.flip(1)], [0, 2], np.random.default_rng(0), keep=0.5, device="cpu")
    assert kept.shape == (2, 4, 2)  # half of the eight tokens survive
    whole = take_tokens([upright], [1], np.random.default_rng(0), keep=1.0, device="cpu")
    assert torch.allclose(whole[0], upright[1].float())


def unit(rows):
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_window_groups_label_positives_and_junk_apart():
    # rows 0 and 1 are one vehicle on two cameras; row 2 is the same vehicle on camera 10 again (junk for row 0)
    embeddings = unit([[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.0, 1.0], [-1.0, 0.0]])
    vehicle = np.array([0, 0, 0, 1, 2])
    camera = np.array([10, 11, 10, 10, 10])
    groups = window_groups(embeddings, vehicle, camera, window=3, device="cpu")
    anchor, candidates, positive, junk = next(g for g in groups if g[0] == 0)
    assert list(candidates) == [1, 2, 3]
    assert positive.tolist() == [True, False, False]   # same vehicle, another camera
    assert junk.tolist() == [False, True, False]       # same vehicle, same camera


def test_window_groups_force_a_positive_into_a_window_that_missed_it():
    embeddings = unit([[1.0, 0.0], [0.9, 0.3], [0.8, 0.4], [-0.2, 1.0]])
    vehicle = np.array([0, 1, 1, 0])
    camera = np.array([10, 10, 11, 11])           # row 3 is the only cross-camera partner of row 0 and ranks last
    groups = window_groups(embeddings, vehicle, camera, window=2, device="cpu")
    _, candidates, positive, _ = next(g for g in groups if g[0] == 0)
    assert positive.any() and candidates[positive][0] == 3


def test_window_groups_drop_anchors_without_a_cross_camera_partner():
    embeddings = unit([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    vehicle = np.array([0, 0, 1, 1])
    camera = np.array([10, 10, 10, 11])           # vehicle 0 was only ever seen by camera 10
    assert sorted(g[0] for g in window_groups(embeddings, vehicle, camera, 2, "cpu")) == [2, 3]


def test_window_loss_ignores_junk_and_rewards_positives_on_top():
    positive = torch.tensor([[True, False, False]])
    junk = torch.tensor([[False, True, False]])
    winning = torch.tensor([[4.0, -4.0, -4.0]])
    losing = torch.tensor([[-4.0, -4.0, 4.0]])
    assert window_loss(winning, positive, junk) < window_loss(losing, positive, junk)
    # moving the junk logit does not change the loss: the scorer drops junk, so it carries no reward
    moved = torch.tensor([[4.0, 4.0, -4.0]])
    assert torch.allclose(window_loss(winning, positive, junk), window_loss(moved, positive, junk), atol=1e-5)
