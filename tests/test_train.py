import pandas as pd
import torch

from reid.train import PKSampler, batch_hard_triplet


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
