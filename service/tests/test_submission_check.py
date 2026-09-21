import csv
import json

import numpy as np
import pytest
from PIL import Image

from falcon_service.cli import preview_export
from falcon_service.submission_check import validate_artifacts


@pytest.fixture
def export(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    query, gallery = tmp_path / "query.csv", tmp_path / "gallery.csv"
    for path, ids in [(query, ["q2", "q1"]), (gallery, [f"g{number}" for number in range(12)])]:
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            for i, identity in enumerate(ids):
                writer.writerow([identity, 0, 0, 12, 12])
                Image.new("RGB", (12, 12), (20 * i, 40, 160)).save(images / f"{identity}.png")
    output = tmp_path / "output"
    preview_export(query, gallery, images, output)
    return query, gallery, output


def test_preview_export_validates_and_preserves_csv_order_without_claiming_readiness(export):
    query, gallery, output = export
    report = validate_artifacts(query, gallery, output)
    assert report["format_valid"]
    assert report["row_provenance"] == "manifest_consistent"
    assert report["accepted_rows"] == 0
    assert not report["submission_ready"]
    assert not report["model_quality_verified"]
    matrix = np.load(output / "embeddings.npy", allow_pickle=False)
    assert matrix.shape == (14, 192)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["embedding_order"]["query"] == ["q2", "q1"]


@pytest.mark.parametrize("change", ["missing_refusal", "duplicate_candidate", "unknown_candidate", "header"])
def test_bad_submission_rows_are_rejected_even_for_refused_queries(export, change):
    query, gallery, output = export
    path = output / "submission.csv"
    rows = list(csv.reader(path.open()))
    if change == "missing_refusal":
        rows.pop()
    elif change == "duplicate_candidate":
        rows[0][2] = rows[0][1]
    elif change == "unknown_candidate":
        rows[0][1] = "unknown"
    else:
        rows.insert(0, ["query_id", "gallery_id"])
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="submission.csv"):
        validate_artifacts(query, gallery, output)


@pytest.mark.parametrize("change", ["dtype", "row_count", "nonfinite", "reorder"])
def test_corrupt_embedding_delivery_is_rejected(export, change):
    query, gallery, output = export
    path = output / "embeddings.npy"
    matrix = np.load(path, allow_pickle=False)
    if change == "dtype":
        matrix = matrix.astype(np.float64)
    elif change == "row_count":
        matrix = matrix[:-1]
    elif change == "nonfinite":
        matrix[0, 0] = np.nan
    else:
        matrix[[0, 1]] = matrix[[1, 0]]
    np.save(path, matrix, allow_pickle=False)
    with pytest.raises(ValueError):
        validate_artifacts(query, gallery, output)


def test_manifest_absence_does_not_claim_verified_row_provenance(export):
    query, gallery, output = export
    (output / "artifact_manifest.json").unlink()
    assert validate_artifacts(query, gallery, output)["row_provenance"] == "not_supplied"


def test_empty_embedding_file_has_a_controlled_diagnostic(export):
    query, gallery, output = export
    (output / "embeddings.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        validate_artifacts(query, gallery, output)
