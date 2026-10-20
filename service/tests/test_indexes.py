"""HNSW retrieval keeps immutable gallery positions and exact calibration statistics separate."""

import numpy as np

from falcon_service.indexes import LoadedGalleryIndex, prepare_index_artifact
from falcon_service.inference import FeatureBundle, rank_candidates, rank_gallery
from falcon_service.storage import ArtifactStore


def _unit_rows(seed: int = 7, count: int = 256, dimension: int = 32):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dimension)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, [FeatureBundle(row, None, "test-model") for row in matrix]


class _Encoder:
    mode = "test"
    model_id = "test-model"
    rerank_window = 0

    def rerank(self, query, candidates, similarities):
        return similarities


def test_hnsw_shortlist_and_exact_population_use_the_same_persisted_graph(tmp_path):
    import faiss

    matrix, features = _unit_rows()
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = prepare_index_artifact(store, "workspace", "version", "test-model", features)
    index = faiss.deserialize_index(np.frombuffer(store.read(artifact.artifact_key), dtype=np.uint8))
    loaded = LoadedGalleryIndex(
        artifact.version_id,
        artifact.model_id,
        artifact.count,
        artifact.dimension,
        index,
    )

    approximate, population = loaded.retrieve(matrix[137], 10, exact_population=False)
    assert population is None
    assert len(approximate) == 100
    assert approximate[0] == 137

    candidates, population = loaded.retrieve(matrix[137], 10, exact_population=True)
    assert len(candidates) == 100
    assert candidates[0] == 137
    np.testing.assert_allclose(population, np.sort(matrix @ matrix[137])[::-1], atol=1e-6)


def test_exact_population_keeps_all_cutoff_ties_for_feature_owned_order(tmp_path):
    import faiss

    angles = np.linspace(0, 2 * np.pi, 256, endpoint=False, dtype=np.float32)
    matrix = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=1)
    features = [FeatureBundle(row, None, "test-model") for row in matrix]
    query = FeatureBundle(np.array([1, 0, 0], np.float32), None, "test-model")
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = prepare_index_artifact(store, "workspace", "version", "test-model", features)
    index = faiss.deserialize_index(np.frombuffer(store.read(artifact.artifact_key), dtype=np.uint8))
    loaded = LoadedGalleryIndex("version", "test-model", len(features), 3, index)

    positions, population = loaded.retrieve(query.embedding, 10, exact_population=True)
    assert len(positions) == len(features)
    actual = rank_candidates(
        query,
        [features[position] for position in positions],
        positions,
        _Encoder(),
        top_n=10,
        gallery_size=len(features),
        population_scores=population,
    )
    expected = rank_gallery(query, features, _Encoder(), top_n=10)
    assert [item["index"] for item in actual["ranked"]] == [item["index"] for item in expected["ranked"]]


def test_hnsw_recall_against_exact_for_independent_queries(tmp_path):
    import faiss

    matrix, features = _unit_rows(seed=41, count=512, dimension=48)
    rng = np.random.default_rng(91)
    queries = rng.normal(size=(20, matrix.shape[1])).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = prepare_index_artifact(store, "workspace", "version", "test-model", features)
    index = faiss.deserialize_index(np.frombuffer(store.read(artifact.artifact_key), dtype=np.uint8))
    loaded = LoadedGalleryIndex("version", "test-model", len(features), matrix.shape[1], index)

    recalls = []
    for query in queries:
        approximate, _ = loaded.retrieve(query, 10, exact_population=False)
        exact = np.argsort(-(matrix @ query))[:10]
        recalls.append(len(set(approximate[:10]) & set(exact)) / 10)
    assert np.mean(recalls) >= 0.95
