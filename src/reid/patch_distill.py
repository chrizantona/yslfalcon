"""Dimension-free global and patch relation losses for Giant -> So400m vehicle ReID distillation.

These are the core, model-independent pieces of the MM'22 patch-KD generalisation. They compare geometry rather than
feature coordinates, so teacher and student may have different widths and patch grids. Integration into reid.train is
deliberately deferred until the Giant token shapes and augmentation coordinate trace pass the preflight.
"""
import math

import torch
import torch.nn.functional as F


def _square_grid(tokens):
    side = math.isqrt(tokens.shape[1])
    if tokens.ndim != 3 or side * side != tokens.shape[1]:
        raise ValueError(f"expected B x square-patches x C tokens, got {tuple(tokens.shape)}")
    return side


def select_diverse_patches(tokens, global_feature, k=3, suppression_radius=0.2):
    """Select K teacher patches aligned with its global descriptor, suppressing spatially adjacent repeats.

    Returns BxK token indices and BxKx2 centre coordinates in [0, 1] as (x, y). The deterministic selector is the
    first ablation; a learnable Gumbel sampler belongs in a later experiment only if relation KD helps.
    """
    side = _square_grid(tokens)
    if tokens.shape[2] != global_feature.shape[1]:
        raise ValueError("teacher tokens and global feature must share a coordinate space for patch selection")
    if not 1 <= k <= tokens.shape[1]:
        raise ValueError("k must be between one and the number of patches")
    scores = torch.einsum("bnc,bc->bn", F.normalize(tokens.float(), dim=2),
                          F.normalize(global_feature.float(), dim=1))
    y, x = torch.meshgrid(torch.arange(side, device=tokens.device), torch.arange(side, device=tokens.device),
                          indexing="ij")
    coordinates = torch.stack(((x.flatten() + .5) / side, (y.flatten() + .5) / side), dim=1)
    chosen = []
    working = scores.clone()
    radius2 = suppression_radius ** 2
    for _ in range(k):
        index = working.argmax(1)
        chosen.append(index)
        centre = coordinates[index]
        close = ((coordinates[None, :, :] - centre[:, None, :]) ** 2).sum(2) <= radius2
        working = working.masked_fill(close, float("-inf"))
        # Very coarse grids or a large radius may suppress everything; fall back to the best unused patch.
        exhausted = ~torch.isfinite(working).any(1)
        if exhausted.any() and len(chosen) < k:
            used = torch.zeros_like(working, dtype=torch.bool)
            used.scatter_(1, torch.stack(chosen, 1), True)
            working[exhausted] = scores[exhausted].masked_fill(used[exhausted], float("-inf"))
    indices = torch.stack(chosen, 1)
    return indices, coordinates[indices]


def gather_patches(tokens, indices):
    return tokens.gather(1, indices[:, :, None].expand(-1, -1, tokens.shape[2]))


def sample_patches(tokens, coordinates):
    """Bilinearly sample a student's square token grid at teacher coordinates in [0, 1]."""
    side = _square_grid(tokens)
    grid = tokens.transpose(1, 2).reshape(len(tokens), tokens.shape[2], side, side)
    locations = coordinates.mul(2).sub(1).unsqueeze(2)  # B x K x 1 x 2 for grid_sample
    sampled = F.grid_sample(grid, locations, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.squeeze(3).transpose(1, 2)


def relation_matrix(features):
    """Cosine distance matrix over the penultimate axis; the feature width may differ across models."""
    features = F.normalize(features.float(), dim=-1)
    return 1 - features @ features.transpose(-1, -2)


def global_relation_distillation(student, teacher, beta=0.1):
    return F.smooth_l1_loss(relation_matrix(student), relation_matrix(teacher), beta=beta)


def patch_relation_distillation(student_patches, teacher_patches, beta=0.1):
    """(intra-image, inter-image) relation KD, each matching a distance matrix with Smooth-L1/Huber.

    Intra compares all KxK regions within each image. Inter compares BxB images separately for each of the K spatially
    corresponding teacher-selected regions, so identities are not mixed and feature widths need not match.
    """
    if student_patches.shape[:2] != teacher_patches.shape[:2]:
        raise ValueError("student and teacher need the same batch and selected-patch count")
    intra = F.smooth_l1_loss(relation_matrix(student_patches), relation_matrix(teacher_patches), beta=beta)
    inter = F.smooth_l1_loss(relation_matrix(student_patches.transpose(0, 1)),
                             relation_matrix(teacher_patches.transpose(0, 1)), beta=beta)
    return intra, inter
