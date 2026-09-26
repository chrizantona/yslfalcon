from pathlib import Path

import yaml


def test_every_config_parses_and_experiments_have_an_id():
    for path in sorted(Path("configs").rglob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert isinstance(cfg, dict), path
        if path.name.startswith(("exp", "stage", "threshold")):
            assert cfg["exp_id"].startswith("EXP-"), path


def test_training_chain_links_up():
    """Every stage starts from the file the previous one writes, and the submission loads what the chain produces."""
    def load(name):
        return yaml.safe_load(Path("configs", name).read_text())

    pretrain, finetune = load("train/stage1_pretrain_veriwild.yaml"), load("train/stage2_finetune_all_folds.yaml")
    soup, head, final = load("train/stage3_soup.yaml"), load("train/stage4_view_head.yaml"), load("final_model.yaml")
    assert finetune["model"]["init_checkpoint"] == pretrain["checkpoint_name"]
    assert soup["checkpoints"] == [finetune["checkpoint_name"].format(seed=s) for s in finetune["seeds"]]
    assert head["candidate"]["checkpoint"] == soup["checkpoint_name"] == final["checkpoint_name"]
    assert head["head_name"] == final["view"]["head"]
    calibration = load("train/threshold_calibration.yaml")
    for fold, spec in calibration["folds"].items():
        run = load(spec["config"].removeprefix("configs/"))
        assert run["eval_fold"] == fold and fold not in run["train"]["folds"]   # scored only on the fold it never saw
        assert spec["checkpoint"] == run["checkpoint_name"].format(seed=0)
        assert run["model"]["init_checkpoint"] == pretrain["checkpoint_name"]
