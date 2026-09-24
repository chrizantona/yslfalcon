"""The GPU self-check of the delivered image (reid.selfcheck)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.selfcheck import painted_dataset, weights


def test_painted_dataset_writes_frames_and_boxes(tmp_path):
    table, images = painted_dataset(tmp_path, 3, np.random.default_rng(0))
    assert list(table.columns) == ["image_id", "x", "y", "w", "h"]
    assert len(list(images.glob("*.jpg"))) == 3
    assert pd.read_csv(tmp_path / "test_query.csv").shape == (3, 5)
    inside = (table.x + table.w <= 1920).all() and (table.y + table.h <= 1080).all()
    assert inside, "a painted box must stay inside the frame"


def test_weights_refuse_a_file_that_does_not_match(tmp_path, monkeypatch):
    manifest = json.loads(Path("configs/weights.json").read_text())
    name = next(iter(manifest["files"]))
    (tmp_path / name).write_bytes(b"not the published file")
    try:
        weights({}, tmp_path)
    except SystemExit as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("a wrong file must fail the check")
