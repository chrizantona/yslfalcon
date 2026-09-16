import math

import pandas as pd
import torch
import torch.nn.functional as F

from reid.models import MarginClassifier, ReIDModel, load_backbone
from reid.train import PKSampler, batch_hard_triplet


def margin_logits(kind, emb, labels, margin=0.5, scale=30.0):
    head = MarginClassifier(feat_dim=2, n_classes=2, kind=kind, scale=scale, margin=margin)
    head.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return head(emb, labels)


def test_cosface_subtracts_the_margin_from_the_target_cosine_only():
    emb = torch.tensor([[3.0, 3.0]])  # cosine 0.707 to both classes, the norm must not matter
    logits = margin_logits("cosface", emb, torch.tensor([0]), margin=0.35)
    cos = 1 / math.sqrt(2)
    assert torch.allclose(logits, torch.tensor([[30 * (cos - 0.35), 30 * cos]]))


def test_arcface_adds_the_margin_to_the_target_angle():
    theta = 0.3
    emb = torch.tensor([[math.cos(theta), math.sin(theta)]])
    logits = margin_logits("arcface", emb, torch.tensor([0]), margin=0.5)
    assert torch.allclose(logits, torch.tensor([[30 * math.cos(theta + 0.5), 30 * math.cos(math.pi / 2 - theta)]]))


def test_arcface_penalty_stays_monotonic_past_pi_minus_margin():
    angles = torch.tensor([2.5, 2.8, 3.1])  # all beyond pi - 0.5
    emb = torch.stack([torch.cos(angles), torch.sin(angles)], 1)
    target = margin_logits("arcface", emb, torch.zeros(3, dtype=torch.long))[:, 0]
    assert (target[1:] < target[:-1]).all()  # a larger angle to the class is always a lower logit
    assert torch.isfinite(torch.autograd.functional.jacobian(
        lambda e: margin_logits("arcface", F.normalize(e, dim=1), torch.zeros(3, dtype=torch.long)), emb)).all()


def test_pk_sampler_alternates_cameras_and_covers_every_vehicle_once():
    rows = [(vehicle, camera) for vehicle in range(8) for camera in (10, 20) for _ in range(3)]  # 2 cameras x 3 frames
    df = pd.DataFrame(rows, columns=["vehicle_id", "camera_id"])
    batches = list(PKSampler(df, P=4, K=4, seed=0))
    assert len(batches) == 2 and all(len(batch) == 16 for batch in batches)
    assert sorted(df.iloc[sum(batches, [])].vehicle_id.unique()) == list(range(8))
    for batch in batches:
        for _, frames in df.iloc[batch].groupby("vehicle_id"):
            assert frames.index.is_unique  # no frame repeated
            assert (frames.camera_id.value_counts() == 2).all()  # round-robin over the two cameras


def test_triplet_loss_is_zero_for_well_separated_vehicles():
    feat = torch.tensor([[0.0, 0.0], [0.0, 0.1], [5.0, 5.0], [5.0, 5.1]])
    assert batch_hard_triplet(feat, torch.tensor([0, 0, 1, 1]), margin=0.3).item() == 0.0


def test_triplet_loss_penalises_a_negative_closer_than_the_positive():
    feat = torch.tensor([[0.0], [2.0], [0.5], [3.0]])
    assert batch_hard_triplet(feat, torch.tensor([0, 0, 1, 1]), margin=0.3).item() > 0


def test_load_backbone_copies_the_backbone_and_keeps_the_new_head(tmp_path):
    torch.manual_seed(0)
    pretrained = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=10)
    torch.save({"state_dict": pretrained.state_dict()}, tmp_path / "pretrained.pt")
    model = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=5)  # another class count: the head can't be copied
    head = model.classifier.weight.clone()
    load_backbone(model, tmp_path / "pretrained.pt")
    assert torch.equal(model.backbone.weight, pretrained.backbone.weight)
    assert torch.equal(model.classifier.weight, head)
