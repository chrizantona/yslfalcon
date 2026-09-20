import numpy as np

from reid.vehicleid_centroid_audit import grouped_centres


def test_grouped_centres_average_then_l2_normalise():
    embeddings = np.array([[1, 0], [0, 1], [0, 2]], np.float32)
    names, centres = grouped_centres(["b", "a", "a"], embeddings, "cpu")

    assert names.tolist() == ["a", "b"]
    np.testing.assert_allclose(centres.numpy(), [[0, 1], [1, 0]], atol=1e-6)
