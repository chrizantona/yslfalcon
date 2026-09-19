import numpy as np

from reid.gallery_transforms import alpha_qe, apply_variant, centre, normalise, whiten


def sample(seed=0, n_query=7, n_gallery=11, dim=5):
    rng = np.random.default_rng(seed)
    return normalise(rng.normal(size=(n_query, dim))), normalise(rng.normal(size=(n_gallery, dim)))


def test_base_leaves_the_embeddings_alone():
    q, g = sample()
    q2, g2 = apply_variant(q, g, {"type": "base"})
    assert np.allclose(q2, q) and np.allclose(g2, g)


def test_centre_removes_the_gallery_mean_and_keeps_unit_length():
    q, g = sample()
    q2, g2 = centre(q, g)
    assert np.allclose(np.linalg.norm(q2, axis=1), 1) and np.allclose(np.linalg.norm(g2, axis=1), 1)
    mean = g.mean(0)
    assert np.allclose(q2, normalise(q - mean))


def test_whitening_power_zero_changes_nothing_and_full_whitening_decorrelates_the_gallery():
    q, g = sample(n_gallery=400, dim=4)
    q0, g0 = whiten(q, g, 0.0)
    assert np.allclose(q0, normalise(q - g.mean(0)))  # only the centring is left
    _, g1 = whiten(q, g, 1.0)
    covariance = np.cov(g1.T)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.abs(off_diagonal).max() < 0.05  # the gallery directions are decorrelated


def test_alpha_qe_moves_a_query_towards_its_nearest_gallery_item_and_leaves_the_gallery_alone():
    q = normalise(np.array([[1.0, 0.0]]))
    g = normalise(np.array([[1.0, 0.3], [-1.0, 0.0]]))  # the first item is the neighbour
    q2, g2 = alpha_qe(q, g, k=1, alpha=3.0)
    assert np.allclose(g2, g)
    assert np.allclose(np.linalg.norm(q2, axis=1), 1)
    assert q2[0] @ g[0] > q[0] @ g[0]  # closer to the neighbour than before
    assert q2[0, 1] > 0  # moved towards it, not away
