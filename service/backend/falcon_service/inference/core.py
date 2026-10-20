"""Query-independent retrieval and explicit, version-bound open-set decisions."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

CONFIDENCE_KIND = "global_cosine_top1_zscore_population_v1"
SCORE_DEFINITION = "cosine_or_pair_rank_permutation_v1"


@dataclass(frozen=True)
class FeatureBundle:
    embedding: np.ndarray
    tokens: np.ndarray | None
    model_id: str
    forward_ms: float | None = None  # the model's own forward pass for this encoding, when it was measured


class Encoder(Protocol):
    """Application-owned port; implementations do not access galleries or other queries."""

    mode: str
    model_id: str
    dimension: int
    rerank_window: int
    calibration_policy: dict | None
    real_model_connected: bool
    device: str
    precision: str
    weights_bytes: int

    def encode(self, image: bytes, bbox: tuple[int, int, int, int] | None = None) -> FeatureBundle: ...

    def rerank(
        self, query: FeatureBundle, candidates: list[FeatureBundle], similarities: np.ndarray
    ) -> np.ndarray: ...


def normalise(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Embedding must be a finite, nonzero vector")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def feature_order(scores, matrix):
    """Resolve numerical score ties using feature values, never IDs or CSV positions.

    Identical feature vectors are indistinguishable under this model and remain tied.
    """
    scores = np.asarray(scores)
    keys = [matrix[:, j] for j in range(matrix.shape[1] - 1, -1, -1)]
    return np.lexsort((*keys, -scores))


class ExactIndex:
    """Exact dot-product index. Optional FAISS acceleration preserves full-row statistics."""

    def __init__(self, matrix, backend="numpy"):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if self.matrix.ndim != 2 or not len(self.matrix) or not np.isfinite(self.matrix).all():
            raise ValueError("Gallery index must contain finite vectors")
        self.index = None
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("The faiss-cpu runtime dependency is unavailable") from exc
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
        elif backend != "numpy":
            raise ValueError("Unknown exact index backend")

    def similarities(self, query):
        query = np.ascontiguousarray(query, dtype=np.float32)
        if self.index is None:
            return self.matrix @ query
        values, positions = self.index.search(query[None], len(self.matrix))
        result = np.empty(len(self.matrix), dtype=np.float32)
        result[positions[0]] = values[0]
        return result


def _calibration_reason(encoder, policy, size, standard_deviation):
    if encoder.mode == "preview":
        return "preview_descriptor_is_not_vehicle_reidentification"
    if not isinstance(policy, dict):
        return "no_calibration_policy"
    if policy.get("model_id") != encoder.model_id:
        return "policy_model_mismatch"
    if policy.get("confidence_kind") != CONFIDENCE_KIND or policy.get("score_definition") != SCORE_DEFINITION:
        return "policy_score_definition_mismatch"
    if not policy.get("validation_reference"):
        return "policy_validation_reference_missing"
    lo, hi, threshold = (
        policy.get("gallery_size_min"),
        policy.get("gallery_size_max"),
        policy.get("threshold"),
    )
    if (
        type(lo) is not int
        or type(hi) is not int
        or not 2 <= lo <= hi
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        return "invalid_calibration_policy"
    if not lo <= size <= hi:
        return "gallery_size_outside_validated_range"
    if standard_deviation is not None and standard_deviation <= 1e-12:
        return "degenerate_gallery_scores"
    if threshold > math.sqrt(size - 1) + 1e-6:
        return "threshold_unreachable_for_gallery_size"
    return None


def calibration_needs_population(encoder, policy, size):
    """Whether a policy may make a decision after exact full-gallery statistics are known."""

    return _calibration_reason(encoder, policy, size, None) is None


def rank_candidates(
    query,
    candidates,
    candidate_indexes,
    encoder,
    top_n=10,
    policy=None,
    gallery_size=None,
    population_scores=None,
):
    """Exact-rescore and rerank a supplied candidate set.

    ANN callers provide global gallery positions and omit population_scores. A calibrated
    decision is then deliberately unavailable: a threshold validated on the exact top-1 score
    must not silently be transferred to approximate retrieval.
    """

    if not candidates:
        raise ValueError("Candidate set is empty")
    if type(top_n) is not int or not 1 <= top_n <= 50:
        raise ValueError("top_n must be between 1 and 50")
    if len(candidate_indexes) != len(candidates) or len(set(candidate_indexes)) != len(candidates):
        raise ValueError("Candidate indexes must be unique and aligned")
    if query.model_id != encoder.model_id or any(item.model_id != query.model_id for item in candidates):
        raise ValueError("Query and gallery model versions differ")
    vector = normalise(query.embedding)
    matrix = np.stack([normalise(item.embedding) for item in candidates])
    if matrix.shape[1] != len(vector):
        raise ValueError("Embedding dimensions differ")
    cosine = matrix @ vector
    scores = cosine.copy()
    window = min(len(candidates), getattr(encoder, "rerank_window", 0) or top_n)
    picked = feature_order(cosine, matrix)[:window]
    updated = np.asarray(
        encoder.rerank(query, [candidates[i] for i in picked], cosine[picked]), dtype=np.float32
    )
    if updated.shape != (window,) or not np.isfinite(updated).all():
        raise RuntimeError("Reranker returned invalid scores")
    # Rank permutation preserves the score population, and therefore the calibrated scale.
    if not np.allclose(np.sort(updated), np.sort(cosine[picked]), atol=1e-6, rtol=0):
        raise RuntimeError("Reranker violated score permutation protocol")
    scores[picked] = updated
    size = gallery_size if gallery_size is not None else len(candidates)
    if type(size) is not int or size < len(candidates):
        raise ValueError("Gallery size is inconsistent with the candidate set")
    if population_scores is None:
        confidence = None
        reason = _calibration_reason(encoder, policy, size, None)
        if reason is None:
            reason = "ann_confidence_unavailable"
    else:
        population = np.asarray(population_scores, dtype=np.float32)
        if population.shape != (size,) or not np.isfinite(population).all():
            raise ValueError("Population scores must contain one finite score per gallery item")
        deviation = float(population.std())
        confidence = float((population.max() - population.mean()) / max(deviation, 1e-12))
        reason = _calibration_reason(encoder, policy, size, deviation)
    threshold = float(policy["threshold"]) if reason is None else None
    accepted = reason is None and confidence is not None and confidence >= threshold
    order = feature_order(scores, matrix)[: min(top_n, len(candidates))]
    return {
        "decision": "calibration_unavailable"
        if reason
        else ("candidates" if accepted else "no_confident_match"),
        "ranked": [
            {
                "index": int(candidate_indexes[index]),
                "cosine": float(cosine[index]),
                "score": float(scores[index]),
                "accepted": bool(accepted and rank == 0),
            }
            for rank, index in enumerate(order)
        ],
        "query_confidence": confidence,
        "threshold": threshold,
        "confidence_kind": CONFIDENCE_KIND,
        "calibration_reason": reason,
    }


def rank_gallery(query, gallery, encoder, top_n=10, policy=None):
    if not gallery:
        raise ValueError("Gallery is empty")
    vector = normalise(query.embedding)
    matrix = np.stack([normalise(item.embedding) for item in gallery])
    if matrix.shape[1] != len(vector):
        raise ValueError("Embedding dimensions differ")
    cosine = ExactIndex(matrix, getattr(encoder, "index_backend", "numpy")).similarities(vector)
    return rank_candidates(
        query,
        gallery,
        list(range(len(gallery))),
        encoder,
        top_n=top_n,
        policy=policy,
        gallery_size=len(gallery),
        population_scores=cosine,
    )
