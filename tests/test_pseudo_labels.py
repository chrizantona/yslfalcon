import numpy as np

from reid.pseudo_labels import components, mutual_edges, score_clusters


def unit(rows):
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_mutual_edges_are_symmetric_and_need_both_sides_to_agree():
    # 0 and 1 are a tight pair; 2 is close to 1 but 1 prefers 0; 3 is far from everything
    emb = unit([[1.0, 0.0], [0.99, 0.1], [0.9, 0.4], [-1.0, 0.0]])
    edges = mutual_edges(emb, k=1, z_threshold=-10)
    assert edges == [(0, 1)]  # only the pair that lists each other first
    assert all(i < j for i, j in edges)


def test_z_threshold_drops_links_that_do_not_stand_out():
    emb = unit(np.random.default_rng(0).normal(size=(20, 8)))
    assert mutual_edges(emb, k=3, z_threshold=10) == []  # nothing stands out that far in a random cloud
    assert mutual_edges(emb, k=3, z_threshold=-10) != []


def test_components_drop_singletons_and_oversized_groups():
    labels = components(6, [(0, 1), (2, 3), (3, 4), (4, 2)], max_size=2)
    assert labels[0] == labels[1] >= 0  # the pair survives
    assert set(labels[2:5]) == {-1}  # the triangle is over max_size
    assert labels[5] == -1  # the singleton is dropped


def test_score_clusters_counts_purity_and_split_vehicles():
    labels = np.array([0, 0, 1, 1, -1])
    truth = ["a", "a", "b", "c", "b"]
    scores = score_clusters(labels, truth)
    assert scores["images"] == 4 and scores["clusters"] == 2
    assert scores["pure_clusters"] == 0.5  # one clean cluster, one mixing b and c
    assert scores["image_purity"] == 0.75  # three of four images sit with their cluster's majority vehicle
    assert scores["coverage"] == 0.8
