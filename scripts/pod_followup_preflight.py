"""Validate recipe deltas, checkpoint hashes, folds, GPUs and a real ViT-L/14 280 training batch."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import pandas as pd
import torch

from reid.data import load_yaml, crop_box
from reid.crops import save_crop
from reid.models import ReIDModel, MarginClassifier, build_backbone, load_backbone
from reid.optim import optimizer_groups
from reid.train import batch_hard_triplet


def main():
    base = load_yaml('configs/exp023_finetune_veriwild_init_vitl14.yaml')
    configs = {26:'exp026_vitl14_llrd09.yaml',27:'exp027_vitl14_arcface.yaml',28:'exp028_vitl14_arcface_imprinting.yaml',29:'exp029_vitl14_280.yaml'}
    for n, name in configs.items():
        cfg = load_yaml(Path('configs') / name)
        expected = copy.deepcopy(base)
        expected.update(exp_id=cfg['exp_id'],parent=cfg['parent'],description=cfg['description'],checkpoint_name=cfg['checkpoint_name'])
        if n==26: expected['train']['layer_decay']=.9
        if n in (27,28): expected['train']['id_head']={'type':'arcface','scale':30.,'margin':.5}
        if n==28: expected['train']['id_head'].update(imprinting=True,imprint_batch_size=128)
        if n==29: expected['model'].update(input_hw=[280,280],interpolate_positional_embedding=True)
        assert cfg==expected, name
        for seed in cfg['seeds']: assert not (Path('/workspace/checkpoints') / cfg['checkpoint_name'].format(seed=seed)).exists()
    for name,digest in [('exp022_seed0.pt','47270c9fe75469ca19b7007480ecc1ea0f7a72b7c774ee466259bc8a08420368'),('exp024.pt','b1a937546b14b693d35196d3dab23fcbae48b1e7657fe30743307a4a8b125bde')]:
        with (Path('/workspace/checkpoints')/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
    folds=pd.read_csv('protocol/val_v2/folds.csv')
    train=folds[folds.fold.isin([1,2,3])]
    assert train.vehicle_id.nunique()==924 and len(train)==5725
    assert set(train.vehicle_id).isdisjoint(folds[folds.fold.isin([0,4])].vehicle_id)
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in gpu_rows.splitlines():
        index,free,util=map(int,line.split(','))
        if index in (1,2,3): assert free>=80000 and util<20,(index,free,util)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for r in train.sample(20,random_state=0).itertuples():
            probe=Path(td)/f'{r.image_id}.jpg'
            save_crop((f'/dev/shm/lct2026/dataset/images/{r.image_id}.jpg',(r.x,r.y,r.w,r.h),str(probe),512,95))
            assert probe.read_bytes()==Path(f'/dev/shm/lct2026/crops512/{r.image_id}.jpg').read_bytes(),r.image_id
    cfg=load_yaml('configs/exp029_vitl14_280.yaml')
    backbone,dim,mean,std=build_backbone(cfg['model'],load_pretrained=False)
    model=ReIDModel(backbone,dim,924)
    load_backbone(model,Path('/workspace/checkpoints/exp022_seed0.pt'),interpolate_position=True)
    assert len(model.backbone.transformer.resblocks)==24 and model.backbone.positional_embedding.shape==(401,1024)
    groups=optimizer_groups(model,{**cfg['train'],'layer_decay':.9})
    params=[id(p) for g in groups for p in g['params']]
    assert len(params)==len(set(params)) and set(params)=={id(p) for p in model.parameters() if p.requires_grad}
    model=model.cuda().train()
    x=torch.randn(128,3,280,280,device='cuda'); y=torch.arange(32,device='cuda').repeat_interleave(4)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16): feat,logits=model(x,y)
    loss=torch.nn.functional.cross_entropy(logits.float(),y,label_smoothing=.1)+batch_hard_triplet(feat.float(),y,.3)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    peak=round(torch.cuda.max_memory_allocated()/2**20)
    model.zero_grad(set_to_none=True)
    model.classifier=MarginClassifier(dim,924,'arcface',30.,.5).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16): feat,logits=model(x[:8],y[:8])
    loss=torch.nn.functional.cross_entropy(logits.float(),y[:8]);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result={'train_vehicles':924,'train_images':5725,'holdout_untouched':True,'input_280_grid':[20,20],'peak_batch128_mib':peak,'real_arcface_gradients_finite':True,'torch':torch.__version__}
    Path('status/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('status/preflight_DONE').touch()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
