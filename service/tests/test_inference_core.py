from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from falcon_service.inference import (
    FeatureBundle,
    PreviewEncoder,
    RepositoryEncoder,
    crop_image,
    explain_pair,
    rank_gallery,
    validate_image,
)
from falcon_service.inference.core import CONFIDENCE_KIND, SCORE_DEFINITION
from falcon_service.inference.encoders import redistribute_scores
from falcon_service.inference.wire import pack_features, unpack_features


def png(color="red", size=(40, 30)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


class TestEncoder:
    __test__ = False
    model_id = "fixture-only"
    mode = "test"
    rerank_window = 0

    def rerank(self, query, candidates, similarities):
        return similarities


def feature(values):
    return FeatureBundle(np.asarray(values, dtype=np.float32), None, TestEncoder.model_id)


def policy(threshold=1.0, **overrides):
    return {
        "model_id": TestEncoder.model_id,
        "confidence_kind": CONFIDENCE_KIND,
        "score_definition": SCORE_DEFINITION,
        "validation_reference": "synthetic-test-fixture",
        "gallery_size_min": 2,
        "gallery_size_max": 20,
        "threshold": threshold,
        **overrides,
    }


def test_preview_never_accepts_even_when_given_threshold():
    encoder = PreviewEncoder()
    query = encoder.encode(png())
    result = rank_gallery(
        query, [query, encoder.encode(png("blue"))], encoder, policy=policy(-100, model_id=encoder.model_id)
    )
    assert query.embedding.dtype == np.float32
    assert np.linalg.norm(query.embedding) == pytest.approx(1)
    assert result["decision"] == "calibration_unavailable"
    assert not any(item["accepted"] for item in result["ranked"])


def test_only_top1_can_be_accepted_and_queries_do_not_interact():
    gallery = [feature([1, 0]), feature([0, 1]), feature([-1, 0])]
    encoder = TestEncoder()
    before = rank_gallery(feature([1, 0]), gallery, encoder, policy=policy())
    rank_gallery(feature([0, 1]), gallery, encoder, policy=policy())
    after = rank_gallery(feature([1, 0]), gallery, encoder, policy=policy())
    assert before == after
    assert before["decision"] == "candidates"
    assert [row["accepted"] for row in before["ranked"]] == [True, False, False]


@pytest.mark.parametrize(
    "configured,reason",
    [
        (None, "no_calibration_policy"),
        (policy(model_id="other"), "policy_model_mismatch"),
        (policy(confidence_kind="cosine"), "policy_score_definition_mismatch"),
        (policy(validation_reference=""), "policy_validation_reference_missing"),
        (policy(gallery_size_min=4), "gallery_size_outside_validated_range"),
        (policy(threshold=7.2429), "threshold_unreachable_for_gallery_size"),
    ],
)
def test_invalid_policy_never_becomes_refusal(configured, reason):
    result = rank_gallery(
        feature([1, 0]),
        [feature([1, 0]), feature([0, 1]), feature([-1, 0])],
        TestEncoder(),
        policy=configured,
    )
    assert result["decision"] == "calibration_unavailable"
    assert result["calibration_reason"] == reason


def test_valid_policy_can_return_a_true_refusal():
    gallery = [feature([1, 0]), feature([0.9, 0.1]), feature([0, 1])]
    result = rank_gallery(feature([1, 0]), gallery, TestEncoder(), policy=policy(1.3))
    assert result["decision"] == "no_confident_match"


def test_degenerate_gallery_is_uncalibrated_and_empty_gallery_is_error():
    query = feature([1, 0])
    result = rank_gallery(query, [query, query], TestEncoder(), policy=policy())
    assert result["calibration_reason"] == "degenerate_gallery_scores"
    with pytest.raises(ValueError, match="empty"):
        rank_gallery(query, [], TestEncoder())


def test_feature_ties_use_features_without_ids_or_row_position():
    query, gallery = feature([1, 0, 0]), [feature([0.5, 0.5, 0]), feature([0.5, 0, 0.5])]
    first = rank_gallery(query, gallery, TestEncoder())["ranked"][0]["index"]
    second = rank_gallery(query, gallery[::-1], TestEncoder())["ranked"][0]["index"]
    np.testing.assert_array_equal(gallery[first].embedding, gallery[::-1][second].embedding)


def test_rank_permutation_preserves_population_and_window_size():
    values = np.array([0.9, 0.8, 0.7], np.float32)
    changed = redistribute_scores(values, np.array([0, 1, 2]), 1.0)
    np.testing.assert_array_equal(changed, values[::-1])
    assert np.mean(changed) == pytest.approx(np.mean(values))
    assert np.std(changed) == pytest.approx(np.std(values))

    class RecordingEncoder(TestEncoder):
        rerank_window = 20

        def rerank(self, query, candidates, similarities):
            assert len(candidates) == 20
            return similarities

    rng = np.random.default_rng(1)
    result = rank_gallery(
        feature([1, 0, 0]), [feature(row) for row in rng.normal(size=(40, 3))], RecordingEncoder(), top_n=35
    )
    assert len(result["ranked"]) == 35


def test_bbox_clips_and_preserves_original_encoding():
    image = png()
    original, bbox, width, height = validate_image(image, (-5, 2, 15, 100))
    assert original is image
    assert (bbox, width, height) == ((0, 2, 10, 28), 40, 30)
    assert Image.open(BytesIO(crop_image(image, bbox))).size == (10, 28)


@pytest.mark.parametrize(
    "box", [(100, 100, 1, 1), (0, 0, 0, 2), (0, 0, float("nan"), 2), (True, 0, 2, 2), (0.5, 0, 2, 2), [0, 1]]
)
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        validate_image(png(), box)


def test_corrupt_image_and_exif_rotation_rejected():
    with pytest.raises(ValueError):
        validate_image(b"not an image")
    stream = BytesIO()
    frame = Image.new("RGB", (20, 10))
    exif = frame.getexif()
    exif[274] = 6
    frame.save(stream, "JPEG", exif=exif)
    with pytest.raises(ValueError, match="EXIF"):
        validate_image(stream.getvalue())


def test_wire_roundtrip_and_pickle_rejection():
    original = FeatureBundle(np.array([1, 0], np.float32), np.zeros((2, 4), np.float16), "fixture")
    loaded, extras = unpack_features(pack_features([original], scores=np.array([0.9], np.float32)))
    np.testing.assert_array_equal(loaded[0].embedding, original.embedding)
    np.testing.assert_array_equal(loaded[0].tokens, original.tokens)
    assert loaded[0].model_id == original.model_id
    assert extras["scores"][0] == pytest.approx(0.9)
    malicious = BytesIO()
    np.savez(malicious, embeddings=np.array([{"bad": "object"}], dtype=object), model_id=np.asarray("x"))
    with pytest.raises(ValueError):
        unpack_features(malicious.getvalue())


def test_occlusion_is_measured_and_labeled_preview():
    pixels = np.zeros((24, 24, 3), dtype=np.uint8)
    pixels[:12, :12] = (255, 0, 0)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, "PNG")
    result = explain_pair(PreviewEncoder(), stream.getvalue(), None, stream.getvalue(), None, grid=2)
    assert result["baseline_score"] == pytest.approx(1)
    assert np.asarray(result["query_map"]).shape == (2, 2)
    assert result["query_map"][0][0] > 0
    assert result["mode"] == "preview"
    assert "no trained vehicle model" in result["preview_warning"].lower()


def test_the_trained_encoder_refuses_to_start_without_its_bundle(tmp_path):
    with pytest.raises(RuntimeError, match="manifest not found"):
        RepositoryEncoder(str(tmp_path / "does-not-exist.json"))
