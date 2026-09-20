import pytest
import torch

from reid.patch_distill import (gather_patches, global_relation_distillation, patch_relation_distillation,
                                sample_patches, select_diverse_patches)


def test_relation_losses_ignore_feature_width_and_detect_changed_geometry():
    teacher = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    student = torch.cat([teacher, torch.zeros(3, 1)], dim=1)
    assert global_relation_distillation(student, teacher).item() == pytest.approx(0, abs=1e-8)
    changed = student.clone()
    changed[1] = torch.tensor([1., 1., 0.])
    assert global_relation_distillation(changed, teacher).item() > 0

    teacher_patches = torch.stack([teacher[:2], teacher[1:]])
    student_patches = torch.cat([teacher_patches, torch.zeros(2, 2, 1)], dim=2)
    intra, inter = patch_relation_distillation(student_patches, teacher_patches)
    assert intra.item() == pytest.approx(0, abs=1e-8)
    assert inter.item() == pytest.approx(0, abs=1e-8)


def test_patch_selection_is_diverse_and_coordinates_map_to_another_grid():
    tokens = torch.eye(16).reshape(1, 16, 16)
    global_feature = torch.zeros(1, 16)
    global_feature[0, [0, 1, 15]] = torch.tensor([3., 2., 1.])
    indices, coordinates = select_diverse_patches(tokens, global_feature, k=3, suppression_radius=.3)
    assert indices.shape == (1, 3) and coordinates.shape == (1, 3, 2)
    distances = torch.cdist(coordinates, coordinates)[0]
    assert distances[torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)].min() > .3
    assert torch.equal(gather_patches(tokens, indices), tokens[:, indices[0]])

    coarse = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    sampled = sample_patches(coarse, torch.tensor([[[.25, .25], [.75, .75]]]))
    assert sampled.flatten().tolist() == pytest.approx([0., 3.])
