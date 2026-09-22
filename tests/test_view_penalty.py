import numpy as np

from reid.view_penalty import VIEWS, fit_view_head, same_view, view_probs


def test_the_head_learns_separable_views():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 16)) * 3
    labels = np.repeat(np.arange(3), 50)
    x = centres[labels] + rng.normal(size=(150, 16))
    head = fit_view_head(x, labels, epochs=200, lr=0.05, weight_decay=0.0)
    probs = view_probs(x, head)
    assert probs.shape == (150, len(VIEWS)) and np.allclose(probs.sum(1), 1)
    assert (probs.argmax(1) == labels).mean() > 0.95


def test_same_view_hard_and_soft():
    q_views, g_views = np.array(["front", "back"]), np.array(["front", "side", "back"])
    assert same_view("clip_hard", q_views, g_views, None, None).tolist() == [[1, 0, 0], [0, 0, 1]]
    q_probs = np.array([[1.0, 0, 0], [0.5, 0.5, 0]])
    g_probs = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    assert np.allclose(same_view("head_soft", None, None, q_probs, g_probs), [[1, 0], [0.5, 0]])
    assert same_view("head_hard", None, None, q_probs, g_probs).tolist() == [[1, 0], [1, 0]]
