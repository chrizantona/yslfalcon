import numpy as np

from reid.rerank import jaccard_distance, k_reciprocal_rerank, k_reciprocal_vectors, streaming_similarity


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_sparse_jaccard_matches_the_dense_formula():
    rng = np.random.default_rng(0)
    x = unit(rng.normal(size=(30, 8))).astype(np.float32)
    dist = 2 - 2 * x @ x.T
    V = k_reciprocal_vectors(dist / dist.max(1, keepdims=True), k1=5, k2=3)
    dense = np.array([[1 - np.minimum(a, b).sum() / np.maximum(a, b).sum() for b in V] for a in V[:10]])
    assert np.allclose(V.sum(1), 1, atol=1e-5)
    assert np.allclose(jaccard_distance(V, 10), dense, atol=1e-5)


def test_lambda_one_keeps_the_original_ranking():
    rng = np.random.default_rng(1)
    q, g = unit(rng.normal(size=(5, 8))), unit(rng.normal(size=(20, 8)))
    reranked = k_reciprocal_rerank(q, g, k1=5, k2=3, lam=1.0)
    assert reranked.shape == (5, 20)
    assert (np.argsort(reranked, 1)[:, 0] == np.argmax(q @ g.T, 1)).all()


def test_clusters_rank_their_own_gallery_first():
    rng = np.random.default_rng(2)
    centres = unit(rng.normal(size=(4, 16)))
    labels = np.repeat(np.arange(4), 6)
    x = unit(centres[labels] + 0.1 * rng.normal(size=(24, 16)))
    query, gallery, q_lab, g_lab = x[::2], x[1::2], labels[::2], labels[1::2]
    top = np.argsort(k_reciprocal_rerank(query, gallery, k1=4, k2=2), 1)[:, 0]
    assert (g_lab[top] == q_lab).all()


def test_streaming_rerank_scores_each_query_without_the_others():
    rng = np.random.default_rng(3)
    query, gallery = unit(rng.normal(size=(6, 8))), unit(rng.normal(size=(40, 8)))
    together = streaming_similarity(query, gallery, k1=5, k2=3, lam=0.3)
    alone = np.vstack([streaming_similarity(q[None], gallery, k1=5, k2=3, lam=0.3) for q in query])
    assert together.shape == (6, 40) and np.array_equal(together, alone)


def test_streaming_rerank_keeps_clusters_and_the_lambda_one_order():
    rng = np.random.default_rng(2)
    centres = unit(rng.normal(size=(4, 16)))
    labels = np.repeat(np.arange(4), 6)
    x = unit(centres[labels] + 0.1 * rng.normal(size=(24, 16)))
    query, gallery, q_lab, g_lab = x[::2], x[1::2], labels[::2], labels[1::2]
    top = np.argmax(streaming_similarity(query, gallery, k1=4, k2=2, lam=0.3), 1)
    assert (g_lab[top] == q_lab).all()
    plain = streaming_similarity(query, gallery, k1=4, k2=2, lam=1.0)
    assert (np.argmax(plain, 1) == np.argmax(query @ gallery.T, 1)).all()
