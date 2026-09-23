"""The final model trains on every fold, so it has no dev fold: reid.train and reid.soup must accept eval_fold null."""
import yaml


def load(name):
    with open(name) as stream:
        return yaml.safe_load(stream)


def test_the_all_folds_configs_switch_off_both_guards():
    cfg, val = load("configs/exp205_all_folds.yaml"), load("configs/val_v2_all_folds.yaml")
    assert cfg["train"]["folds"] == [0, 1, 2, 3, 4]
    assert cfg["eval_fold"] is None and val["holdout_fold"] is None
    assert cfg["eval_fold"] not in cfg["train"]["folds"]          # the assertion reid.train makes
    assert val["holdout_fold"] not in cfg["train"]["folds"]
    assert load("configs/exp206_soup_all_folds.yaml")["eval_fold"] is None


def test_training_table_takes_every_vehicle_of_every_fold():
    from reid.train import training_table
    cfg, val = load("configs/exp205_all_folds.yaml"), load("configs/val_v2_all_folds.yaml")
    rows, crops_dir, classes = training_table(cfg, {"crops_dir": "/dev/null"}, val)
    assert len(classes) == rows.vehicle_id.nunique() == 1541      # every vehicle of the dataset
    assert sorted(rows.label.unique()) == list(range(len(classes)))
