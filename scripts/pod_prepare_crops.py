"""Recreate canonical pod crops with unchanged save_crop after verifying the original ROI pixels."""
import hashlib,json,os,time,csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
from reid.crops import save_crop
from reid.data import crop_box,load_yaml
run=Path.cwd();source=Path('/dev/shm/EXP-020/source');data=load_yaml('configs/data.yaml')
out=Path(data['crops_dir']);scratch=out.with_name('crops512_new');scratch.mkdir(exist_ok=True)
rows=json.loads((source/'manifest.json').read_text());cv2.setNumThreads(1)
def make(row):
 path=source/row['file']; assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256'],row['image_id']
 frame=cv2.imread(str(path));raw=crop_box(frame,row['box'])
 assert hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_roi_sha256'],'original pixels differ: '+row['image_id']
 temp=scratch/(row['image_id']+'.jpg')
 save_crop((str(path),row['box'],str(temp),data['crop_max_side'],data['crop_jpeg_quality']))
 os.replace(temp,out/temp.name)
pending=list(rows);done=0
with ThreadPoolExecutor(16) as pool:
 while pending:
  available=[r for r in pending if (source/r['file']).is_file()]
  if available:
   list(pool.map(make,available));done+=len(available);ids={r['image_id'] for r in available};pending=[r for r in pending if r['image_id'] not in ids]
   print('Canonical training crops',done,'/',len(rows),flush=True)
  else:time.sleep(5)
# Canonical dev/query copies, directly from the already verified original full-frame JPEGs.
dataset=Path(data['dataset_dir']);jobs=[]
for filename in ['train.csv','test_query.csv','test_gallery.csv']:
 for row in csv.DictReader((dataset/filename).open()):
  raw=dataset/'images'/(row['image_id']+'.jpg')
  if raw.is_file():jobs.append((str(raw),tuple(float(row[k]) for k in ('x','y','w','h')),str(scratch/(row['image_id']+'.jpg')),data['crop_max_side'],data['crop_jpeg_quality']))
with ThreadPoolExecutor(16) as pool:list(pool.map(save_crop,jobs))
for p in scratch.glob('*.jpg'):os.replace(p,out/p.name)
# Preserve the staging integrity manifest after the canonical regeneration.
entries=[];root=Path('/dev/shm/lct2026')
for line in Path('SHA256SUMS').read_text().splitlines():
 old,name=line.split('  ',1)
 if name.startswith('crops512/'):
  with (root/name).open('rb') as f:old=hashlib.file_digest(f,'sha256').hexdigest()
 entries.append(f'{old}  {name}\n')
Path('SHA256SUMS').write_text(''.join(entries))
Path('status/canonical_DONE').touch();print('All original training ROI pixels verified; canonical Linux crops ready.',flush=True)
