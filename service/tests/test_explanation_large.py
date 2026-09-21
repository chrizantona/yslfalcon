"""A valid compressed upload must remain usable by the explanation workflow."""

from io import BytesIO

import numpy as np
from PIL import Image

from falcon_service.inference import PreviewEncoder, explain_pair, validate_image


def test_high_detail_jpeg_can_be_explained_without_png_upload_limit_failure():
    pixels = np.random.default_rng(42).integers(0, 256, (2500, 3000, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, "JPEG", quality=90)
    upload = stream.getvalue()
    validate_image(upload)
    result = explain_pair(PreviewEncoder(), upload, None, upload, None, grid=2)
    assert len(result["query_map"]) == 2
    assert np.isfinite(result["query_map"]).all()
    assert result["mode"] == "preview"
    assert "analysis" in result["score_kind"]
