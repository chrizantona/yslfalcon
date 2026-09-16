"""Stage lossless JPEG regions for this pod run; verify every original car ROI before transfer.

The translated box is preprocessing metadata. Image IDs and hashes are used only
for lookup and integrity checks; they are never passed to the model.
"""
import csv,hashlib,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
root=Path('/Users/chrizantona/Downloads/dataset'); out=Path('/private/tmp/exp020_canonical'); (out/'images').mkdir(parents=True,exist_ok=True)
rows=[r for r in csv.DictReader(open('protocol/val_v2/folds.csv')) if int(r['fold']) in (1,2,3)]
cv2.setNumThreads(1)
def make(r):
 ident=r['image_id']; source=root/'images'/f'{ident}.jpg'; frame=cv2.imread(str(source));fh,fw=frame.shape[:2]
 x,y,w,h=(int(float(r[k])) for k in ('x','y','w','h'));bx0,by0,bx1,by1=max(x,0),max(y,0),min(x+w,fw),min(y+h,fh)
 raw=frame[by0:by1,bx0:bx1];sx=max(0,(bx0-32)//32*32);sy=max(0,(by0-32)//32*32)
 ex=min(fw,((bx1+32+31)//32)*32);ey=min(fh,((by1+32+31)//32)*32)
 path=out/'images'/f'{ident}.jpg'; geom=f'{ex-sx}x{ey-sy}+{sx}+{sy}'
 result=subprocess.run(['/opt/homebrew/bin/jpegtran','-copy','none','-crop',geom,'-outfile',str(path),str(source)],capture_output=True)
 chunk=cv2.imread(str(path)) if result.returncode==0 else None
 box=(bx0-sx,by0-sy,bx1-bx0,by1-by0)
 if chunk is None or not np.array_equal(chunk[box[1]:box[1]+box[3],box[0]:box[0]+box[2]],raw):
  path.unlink(missing_ok=True);path=path.with_suffix('.png');assert cv2.imwrite(str(path),raw);box=(0,0,raw.shape[1],raw.shape[0])
 return {'image_id':ident,'file':f'images/{path.name}','box':box,'raw_roi_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
with ThreadPoolExecutor(10) as pool:
 table=[]
 for i,row in enumerate(pool.map(make,rows),1):
  table.append(row)
  if i%1000==0:print('canonical source',i,flush=True)
(out/'manifest.json').write_text(json.dumps(table))
print('ready',len(table),'bytes',sum(p.stat().st_size for p in (out/'images').iterdir()),'PNG fallback',sum(r['file'].endswith('.png') for r in table),flush=True)
