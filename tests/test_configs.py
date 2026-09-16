from pathlib import Path

import yaml


def test_every_config_parses_and_experiments_have_an_id():
    for path in sorted(Path("configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert isinstance(cfg, dict), path
        if path.name.startswith("exp"):
            assert cfg["exp_id"].startswith("EXP-"), path
