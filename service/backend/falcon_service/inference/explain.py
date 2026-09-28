"""Measured occlusion sensitivity of the preview descriptor, not model saliency."""

from io import BytesIO

import numpy as np
from PIL import Image

from .images import validate_image

ANALYSIS_MAX_EDGE = 512


def _analysis_pixels(data, bbox):
    original, (x, y, width, height), _, _ = validate_image(data, bbox)
    with Image.open(BytesIO(original)) as frame:
        crop = frame.convert("RGB").crop((x, y, x + width, y + height))
        crop.thumbnail((ANALYSIS_MAX_EDGE, ANALYSIS_MAX_EDGE), Image.Resampling.BILINEAR)
        return np.asarray(crop)


def _png(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def explain_pair(encoder, query_image, query_bbox, candidate_image, candidate_bbox, grid=6):
    if type(grid) is not int or not 2 <= grid <= 10:
        raise ValueError("Explanation grid must be between 2 and 10")
    # Analyze bounded views: a small valid JPEG may expand beyond the upload limit
    # as a PNG. This keeps each internal encoding bounded and leaves originals intact.
    query = _analysis_pixels(query_image, query_bbox)
    candidate = _analysis_pixels(candidate_image, candidate_bbox)
    q = encoder.encode(_png(query), None).embedding
    c = encoder.encode(_png(candidate), None).embedding
    baseline = float(q @ c)

    def sensitivity(pixels, other):
        height, width = pixels.shape[:2]
        if min(height, width) < grid:
            raise ValueError("Crop is smaller than the explanation grid")
        fill = np.rint(pixels.mean(axis=(0, 1))).astype(np.uint8)
        result = np.zeros((grid, grid), dtype=np.float32)
        for row in range(grid):
            for column in range(grid):
                masked = pixels.copy()
                masked[
                    row * height // grid : (row + 1) * height // grid,
                    column * width // grid : (column + 1) * width // grid,
                ] = fill
                embedding = encoder.encode(_png(masked), None).embedding
                result[row, column] = baseline - float(embedding @ other)
        return result.tolist()

    return {
        "method": "occlusion_sensitivity",
        "score_kind": "analysis_crop_embedding_cosine",
        "analysis_max_edge": ANALYSIS_MAX_EDGE,
        "analysis_sizes": {"query": list(query.shape[1::-1]), "candidate": list(candidate.shape[1::-1])},
        "grid": grid,
        "baseline_score": baseline,
        "query_map": sensitivity(query, c),
        "candidate_map": sensitivity(candidate, q),
        "fill": "crop_mean_rgb",
        "model_id": encoder.model_id,
        "mode": encoder.mode,
        "semantics": "Baseline cosine minus masked cosine on aspect-preserving analysis crops, "
        "resized to at most 512 pixels per edge. This may differ from the original full-resolution score. "
        "Positive values support this descriptor similarity; "
        "this is not causal evidence, identity probability, or an explanation of reranked positions.",
        "preview_warning": "Preview RGB descriptor only; no trained vehicle model is connected."
        if encoder.mode == "preview"
        else None,
    }
