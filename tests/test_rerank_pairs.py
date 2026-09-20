import numpy as np
import torch

from reid.rerank_pairs import PairReranker, group_loss, reorder, training_groups


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


def test_training_groups_take_positives_from_other_cameras():
    import pandas as pd
    df = pd.DataFrame({"vehicle_id": [1, 1, 1, 2, 2, 3, 3],
                       "camera_id": [10, 10, 11, 10, 11, 10, 11]})
    embeddings = np.eye(len(df), dtype=np.float32)
    groups = training_groups(df, embeddings, n_negatives=2, hard_share=0.5, rng=np.random.default_rng(0))
    assert len(groups) == len(df)  # every row here has a same-vehicle frame from another camera
    for anchor, positives, negatives in groups:
        assert all(df.vehicle_id[p] == df.vehicle_id[anchor] for p in positives)
        assert all(df.camera_id[p] != df.camera_id[anchor] for p in positives)
        assert all(df.vehicle_id[n] != df.vehicle_id[anchor] for n in negatives)
        assert len(set(negatives)) == 2


def test_anchors_without_a_cross_camera_positive_are_dropped():
    import pandas as pd
    df = pd.DataFrame({"vehicle_id": [1, 1, 2, 2], "camera_id": [10, 10, 10, 11]})
    groups = training_groups(df, np.eye(4, dtype=np.float32), 1, 0.0, np.random.default_rng(0))
    assert sorted(a for a, _, _ in groups) == [2, 3]  # vehicle 1 was only ever seen by camera 10


def test_group_loss_falls_when_the_positive_wins():
    losing = torch.tensor([[0.0, 3.0, 3.0]])
    winning = torch.tensor([[3.0, 0.0, 0.0]])
    assert group_loss(winning) < group_loss(losing)
