"""The training code of the submission (reid.train, reid.soup, reid.external) and the all-folds configs."""
import math
import zipfile

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from reid.external import load_table, prepare
from reid.models import MarginClassifier, ReIDModel, load_backbone
from reid.soup import average_state_dicts
from reid.train import PKSampler, batch_hard_triplet, optimizer_groups, supervised_contrastive, training_table


def load(name):
    with open(name) as stream:
        return yaml.safe_load(stream)


def margin_logits(kind, emb, labels, margin=0.5, scale=30.0):
    head = MarginClassifier(feat_dim=2, n_classes=2, kind=kind, scale=scale, margin=margin)
    head.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return head(emb, labels)


def test_arcface_adds_the_margin_to_the_target_angle():
    theta = 0.3
    emb = torch.tensor([[math.cos(theta), math.sin(theta)]])
    logits = margin_logits("arcface", emb, torch.tensor([0]), margin=0.5)
    assert torch.allclose(logits, torch.tensor([[30 * math.cos(theta + 0.5), 30 * math.cos(math.pi / 2 - theta)]]))


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


def test_optimizer_groups_split_backbone_and_head_and_leave_the_frozen_bnneck_bias_out():
    model = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=5)
    backbone, head = optimizer_groups(model, {"lr_backbone": 1e-5, "lr_head": 5e-4})
    assert backbone["lr"] == 1e-5 and head["lr"] == 5e-4
    assert [id(p) for p in backbone["params"]] == [id(p) for p in model.backbone.parameters()]
    assert [id(p) for p in head["params"]] == [id(model.bnneck.weight), id(model.classifier.weight)]


def test_load_backbone_copies_the_backbone_and_keeps_the_new_head(tmp_path):
    torch.manual_seed(0)
    pretrained = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=10)
    torch.save({"state_dict": pretrained.state_dict()}, tmp_path / "pretrained.pt")
    model = ReIDModel(torch.nn.Linear(4, 3), feat_dim=3, n_classes=5)  # another class count: the head can't be copied
    head = model.classifier.weight.clone()
    load_backbone(model, tmp_path / "pretrained.pt")
    assert torch.equal(model.backbone.weight, pretrained.backbone.weight)
    assert torch.equal(model.classifier.weight, head)


def test_soup_averages_floats_and_keeps_integer_buffers():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(7)}
    soup = average_state_dicts([a, b])
    assert torch.equal(soup["w"], torch.tensor([2.0, 4.0])) and soup["n"].item() == 5


def test_the_all_folds_configs_switch_off_both_guards():
    cfg, val = load("configs/train/stage2_finetune_all_folds.yaml"), load("configs/train/val_v2_all_folds.yaml")
    assert cfg["train"]["folds"] == [0, 1, 2, 3, 4]
    assert cfg["eval_fold"] is None and val["holdout_fold"] is None
    assert cfg["eval_fold"] not in cfg["train"]["folds"]          # the assertion reid.train makes
    assert val["holdout_fold"] not in cfg["train"]["folds"]


def test_training_table_takes_every_vehicle_of_every_fold():
    cfg, val = load("configs/train/stage2_finetune_all_folds.yaml"), load("configs/train/val_v2_all_folds.yaml")
    rows, crops_dir, classes = training_table(cfg, {"crops_dir": "/dev/null"}, val)
    assert len(classes) == rows.vehicle_id.nunique() == 1541      # every vehicle of the dataset
    assert sorted(rows.label.unique()) == list(range(len(classes)))


def test_prepare_keeps_present_images_prefixes_vehicles_and_unpacks(tmp_path):
    archive = tmp_path / "veriwild.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("images/00005/000021.jpg", "images/00005/000022.jpg", "images/00007/000030.jpg"):
            z.writestr(name, b"jpeg")
        z.writestr("vehicle_info.txt",  # the first vehicle is not in the archive (train split)
                   "00001/000001;14;2018-03-07 16:59:23;Changan;SUV;white\n"
                   "00005/000021;3;2018-03-08 10:00:01;Audi;sedan;black\n"
                   "00005/000022;9;2018-03-08 10:05:12;Audi;sedan;black\n"
                   "00007/000030;3;2018-03-09 08:00:00;BYD;SUV;gray\n")
    table = prepare(archive, tmp_path / "out")

    assert table.image_id.tolist() == ["00005/000021", "00005/000022", "00007/000030"]
    assert table.vehicle_id.tolist() == ["vw_00005", "vw_00005", "vw_00007"]
    assert table.camera_id.tolist() == [3, 9, 3]
    assert (tmp_path / "out" / "images" / "00007" / "000030.jpg").read_bytes() == b"jpeg"
    # pandas 3 reads strings back with its own string dtype, so compare values only
    pd.testing.assert_frame_equal(load_table(tmp_path / "out"), table, check_dtype=False)
