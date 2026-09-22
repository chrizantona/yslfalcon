import numpy as np
import torch

from reid.view_tokens import token_features


class Tower:
    class trunk:
        @staticmethod
        def forward_features(x):
            return x  # the "tokens" are the batch itself: (n, 16, d), a 4 x 4 grid


def test_quarters_are_the_means_of_the_four_corners_of_the_grid():
    tokens = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    out = token_features(Tower, [(tokens, None)], "cpu")
    assert out["mean"].shape == (2, 3) and out["quarters"].shape == (2, 12)
    grid = tokens.reshape(2, 4, 4, 3).numpy()
    assert np.allclose(out["quarters"][:, :3], grid[:, :2, :2].mean((1, 2)))    # top left
    assert np.allclose(out["quarters"][:, 9:], grid[:, 2:, 2:].mean((1, 2)))    # bottom right
    assert np.allclose(out["mean"], grid.mean((1, 2)))
