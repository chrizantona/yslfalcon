"""Finish the path migration when the running night controller exits; never restart training."""
import json
import os
from pathlib import Path
import time

RUNS = Path('/workspace/yslfalcon_runs')
STATUS = RUNS / 'EXP-020/status'
CHECKPOINTS = Path('/workspace/checkpoints')
CONFIGS = {
    1: 'exp020_clip_vitl14_bot_30ep.yaml',
    2: 'exp021_clip_vitl14_fp16.yaml',
    3: 'exp022_pretrain_veriwild_test_vitl14.yaml',
    4: 'exp023_finetune_veriwild_init_vitl14.yaml',
    5: 'exp024_soup_exp023_seeds.yaml',
}
PAIRS = [(f'exp{i}_seed{seed}.pt', f'exp{i + 19:03d}_seed{seed}.pt')
         for i, seeds in [(1, (0, 1, 2)), (3, (0,)), (4, (0, 1, 2))] for seed in seeds]
PAIRS.append(('exp5.pt', 'exp024.pt'))
STAGES = {f'exp{i}': f'EXP-{i + 19:03d}' for i in CONFIGS}
STAGES['exp1_exp2'] = 'EXP-020_EXP-021'


def unlink_alias(path, target):
    if path.is_symlink():
        assert path.resolve() == target.resolve(), f'unexpected alias: {path}'
        path.unlink()


def main():
    (STATUS / 'names_pid').write_text(str(os.getpid()) + '\n')
    while True:
        stage = (STATUS / 'night_stage').read_text().strip()
        (STATUS / 'current_experiment').write_text(STAGES.get(stage, stage) + '\n')
        # A run that already loaded its old config may still save under the old name.
        for old_name, new_name in PAIRS:
            old, new = CHECKPOINTS / old_name, CHECKPOINTS / new_name
            if old.is_file() and not old.is_symlink() and not new.exists() and not new.is_symlink():
                new.symlink_to(old.name)
        if (STATUS / 'night_FAILED').exists():
            (STATUS / 'names_deferred').write_text('Controller failed; aliases retained for recovery.\n')
            return
        if (STATUS / 'night_DONE').exists():
            pid = int((STATUS / 'night_pid').read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        time.sleep(5)
    for old_name, new_name in PAIRS:
        old, new = CHECKPOINTS / old_name, CHECKPOINTS / new_name
        if old.is_file() and not old.is_symlink():
            assert not new.exists() or (new.is_symlink() and new.resolve() == old.resolve()), new
            old.replace(new)
        unlink_alias(old, new)
    for i in CONFIGS:
        run = RUNS / f'EXP-{i + 19:03d}'
        for cfg_i, name in CONFIGS.items():
            unlink_alias(run / f'configs/exp{cfg_i}.yaml', run / 'configs' / name)
        unlink_alias(RUNS / f'exp{i}', run)
    weights = Path('/workspace/ext_data/weights/clip')
    unlink_alias(weights / 'exp1.safetensors', weights / 'exp020.safetensors')
    (STATUS / 'names_DONE').touch()
    print('Canonical EXP names installed; compatibility aliases removed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (STATUS / 'names_FAILED').write_text(repr(exc) + '\n')
        raise
