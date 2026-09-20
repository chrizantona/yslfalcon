import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from reid.models import MarginClassifier, ReIDModel, load_backbone
from reid.train import (GradientReversal, PKSampler, batch_hard_triplet, camera_codes, camera_lambda,
                        class_centres, lookalike_vehicles, similarity_distillation, supervised_contrastive)


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


def test_supcon_matches_the_per_anchor_definition_and_skips_anchors_without_positives():
    torch.manual_seed(0)
    feat, labels, t = torch.randn(8, 5), torch.tensor([0, 0, 1, 1, 2, 2, 2, 3]), 0.1
    z = F.normalize(feat, dim=1)
    losses = []
    for i in range(8):
        others = [j for j in range(8) if j != i]
        positives = [j for j in others if labels[j] == labels[i]]
        if positives:  # vehicle 3 has a single image
            denominator = torch.logsumexp(torch.stack([z[i] @ z[j] / t for j in others]), 0)
            losses.append(-torch.stack([z[i] @ z[p] / t - denominator for p in positives]).mean())
    assert torch.allclose(supervised_contrastive(feat, labels, t), torch.stack(losses).mean(), atol=1e-5)


def test_load_backbone_copies_the_backbone_and_keeps_the_new_head(tmp_path):
    torch.manual_seed(0)
    pretrained = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=10)
    torch.save({"state_dict": pretrained.state_dict()}, tmp_path / "pretrained.pt")
    model = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=5)  # another class count: the head can't be copied
    head = model.classifier.weight.clone()
    load_backbone(model, tmp_path / "pretrained.pt")
    assert torch.equal(model.backbone.weight, pretrained.backbone.weight)
    assert torch.equal(model.classifier.weight, head)


def test_sub_centre_arcface_scores_a_class_by_its_closest_centre():
    head = MarginClassifier(feat_dim=2, n_classes=2, kind="arcface", scale=30.0, margin=0.5, sub_centers=2)
    # class 0: centres at 0 and 90 degrees; class 1: both at 180 degrees
    head.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]])
    closest = MarginClassifier(feat_dim=2, n_classes=2, kind="arcface", scale=30.0, margin=0.5)
    closest.weight.data = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # only the nearer centre of each class
    emb, labels = torch.tensor([[0.3, 1.0]]), torch.tensor([0])
    logits = head(emb, labels)
    assert logits.shape == (1, 2)
    assert torch.allclose(logits, closest(emb, labels), atol=1e-6)  # the far centre of class 0 never counts
    doubled = MarginClassifier(feat_dim=2, n_classes=2, kind="arcface", scale=30.0, margin=0.5, sub_centers=2)
    doubled.weight.data = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    single = MarginClassifier(feat_dim=2, n_classes=2, kind="arcface", scale=30.0, margin=0.5)
    single.weight.data = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # repeated centres reproduce the one-centre head
    assert torch.allclose(doubled(emb, labels), single(emb, labels), atol=1e-6)


def test_class_centres_separate_two_modes_and_stay_unit_length():
    front = F.normalize(torch.tensor([[1.0, 0.05], [1.0, -0.05], [1.0, 0.0]]), dim=1)
    rear = F.normalize(torch.tensor([[-0.05, 1.0], [0.05, 1.0]]), dim=1)
    centres = class_centres(torch.cat([front, rear]), 2)
    assert centres.shape == (2, 2)
    assert torch.allclose(centres.norm(dim=1), torch.ones(2), atol=1e-6)
    nearest = (torch.cat([front, rear]) @ centres.T).argmax(1)
    assert nearest[:3].unique().numel() == 1 and nearest[3:].unique().numel() == 1  # each mode gets its own centre
    assert nearest[0] != nearest[3]
    few = class_centres(front[:1], 3)  # fewer images than centres: the centre repeats
    assert few.shape == (3, 2) and torch.allclose(few[0], few[1]) and torch.allclose(few[0], few[2])


def test_similarity_distillation_is_zero_when_the_student_matches_and_ignores_a_row_shift():
    torch.manual_seed(0)
    teacher = F.normalize(torch.randn(6, 4), dim=1)
    assert similarity_distillation(teacher, teacher, 0.1).item() < 1e-6  # same metric, no loss
    scaled = teacher * 3.0  # L2 normalisation inside removes the scale, so the metric is unchanged
    assert similarity_distillation(scaled, teacher, 0.1).item() < 1e-6
    shuffled = teacher[[1, 0, 2, 3, 4, 5]]  # a different neighbourhood structure costs something
    assert similarity_distillation(shuffled, teacher, 0.1).item() > 0.01


def test_similarity_distillation_matches_the_soft_label_cross_entropy_and_its_gradient():
    torch.manual_seed(0)
    teacher = F.normalize(torch.randn(5, 3), dim=1)
    feat = torch.randn(5, 3, requires_grad=True)
    loss = similarity_distillation(feat, teacher, 0.2)
    student = F.normalize(feat, dim=1)
    off = ~torch.eye(5, dtype=torch.bool)
    z = (student @ student.T / 0.2)[off].view(5, 4)
    p = (teacher @ teacher.T / 0.2)[off].view(5, 4).softmax(1)
    entropy = -(p * p.clamp(min=1e-12).log()).sum(1).mean()
    cross_entropy = -(p * z.log_softmax(1)).sum(1).mean()
    assert torch.allclose(loss, cross_entropy - entropy, atol=1e-5)  # KL = cross-entropy minus the teacher entropy
    grad_z = torch.autograd.grad(similarity_distillation(feat, teacher, 0.2), feat, retain_graph=True)[0]
    assert torch.isfinite(grad_z).all() and grad_z.abs().sum() > 0


def test_gradient_reversal_passes_values_through_and_flips_the_gradient():
    x = torch.tensor([[1.0, -2.0]], requires_grad=True)
    y = GradientReversal.apply(x, 0.3)
    assert torch.equal(y, x)  # forward is the identity
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[-0.3, -0.3]]))  # backward is multiplied by -lambda


def test_camera_lambda_ramps_from_zero_and_then_holds():
    cfg = {"lambda": 0.3, "ramp_epochs": 10}
    assert camera_lambda(cfg, 0) == 0.0
    assert math.isclose(camera_lambda(cfg, 5), 0.15)
    assert math.isclose(camera_lambda(cfg, 10), 0.3) and math.isclose(camera_lambda(cfg, 30), 0.3)


def test_camera_codes_are_int64_indices_cross_entropy_can_take():
    df = pd.DataFrame({"camera_id": [89, 90, 89, 7]})
    codes = camera_codes(df)
    assert codes.dtype == np.int64  # pandas category codes are int8 and cross_entropy rejects them
    assert sorted(set(codes)) == [0, 1, 2] and codes[0] == codes[2]
    logits = torch.randn(4, int(codes.max()) + 1)
    torch.nn.CrossEntropyLoss()(logits, torch.as_tensor(codes))  # must not raise


def test_hard_identity_batches_are_built_around_lookalikes_and_still_cover_every_vehicle_once():
    rows = [(vehicle, camera) for vehicle in range(8) for camera in (10, 20) for _ in range(2)]
    df = pd.DataFrame(rows, columns=["vehicle_id", "camera_id"])
    sampler = PKSampler(df, P=4, K=4, seed=0, hard_share=0.5)
    sampler.set_neighbours({v: [(v + 1) % 8, (v + 2) % 8, (v + 3) % 8] for v in range(8)})  # ring of lookalikes
    batches = list(sampler)
    assert len(batches) == 2 and all(len(b) == 16 for b in batches)
    seen = sorted(df.iloc[sum(batches, [])].vehicle_id.unique())
    assert seen == list(range(8))  # every vehicle exactly once, as without mining
    first = list(dict.fromkeys(df.iloc[batches[0]].vehicle_id))
    assert first[1] == (first[0] + 1) % 8  # nothing is taken yet, so the anchor gets its nearest vehicle
    plain = PKSampler(df, P=4, K=4, seed=0, hard_share=0.5)  # no neighbour map yet: random batches, as before
    assert [sorted(b) for b in plain] == [sorted(b) for b in PKSampler(df, P=4, K=4, seed=0)]


def test_lookalike_vehicles_ranks_by_the_class_centres():
    head = MarginClassifier(feat_dim=2, n_classes=3, kind="arcface", scale=30.0, margin=0.5)
    head.weight.data = torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0]])  # 0 and 1 nearly coincide
    neighbours = lookalike_vehicles(head, ["a", "b", "c"], pool=2)
    assert neighbours["a"][0] == "b" and neighbours["b"][0] == "a"
    assert neighbours["c"] == ["b", "a"] or neighbours["c"] == ["a", "b"]  # both are far, order does not matter here
