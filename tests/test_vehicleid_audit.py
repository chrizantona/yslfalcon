import cv2
import numpy as np

from reid.vehicleid_audit import hamming_matrix, near_duplicate_pairs, perceptual_hash


def test_perceptual_hash_is_stable_and_hamming_pairs_are_thresholded(tmp_path):
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    cv2.imwrite(str(tmp_path / "a.jpg"), gradient)
    cv2.imwrite(str(tmp_path / "b.jpg"), gradient)
    flipped = np.fliplr(gradient)
    cv2.imwrite(str(tmp_path / "c.jpg"), flipped)
    hashes = np.asarray([perceptual_hash(tmp_path / name) for name in ("a.jpg", "b.jpg", "c.jpg")])
    distances = hamming_matrix(hashes[:1], hashes)
    assert distances[0, 0] == distances[0, 1] == 0
    assert distances[0, 2] > 0
    assert list(near_duplicate_pairs(hashes[:1], hashes, threshold=0)) == [(0, 0, 0), (0, 1, 0)]
