"""Frozen, annotation-free 474-Val D/G/F view generator and cache audit."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('results/vflip_factorial')
ORIGINAL=Path('results/fn_analysis/cache')
IMAGES=Path('datasets/yolo_split/images/val')
WEIGHTS=Path('runs/rareos/yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/weights/best.pt')
VIEWS={'direction':('D','original',1), 'grid':('G','mirrored_y',0), 'full':('F','mirrored_y',1)}
PARAMS={'tile_size':1280,'stride':768,'conf':1e-5,'tile_iou':.60,'max_det':1000,'batch':6,'half':True}

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

DIAG=load_module('scripts/14_fn_diagnostic.py','geometry_official14')
get_starts=DIAG.get_starts

def read_json(path): return json.loads(Path(path).read_text())
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def val_items():
    manifest=read_json(ORIGINAL/'manifest.json')
    for k in ['tile_size','stride','conf','tile_iou','max_det','batch']: assert manifest[k]==PARAMS[k], (k,manifest[k])
    assert Path(manifest['model'])==WEIGHTS
    items=manifest['items'];names=[i['image_name'] for i in items]
    actual=sorted(p.name for p in IMAGES.iterdir() if p.suffix.lower() in DIAG.IMAGE_EXTENSIONS)
    assert len(items)==474 and len(set(names))==474 and sorted(names)==actual
    assert all(Path(i['cache_file']).name==i['cache_file'] for i in items)
    return manifest,items

def tile_origins(height,width,grid_mode):
    xs=get_starts(width,PARAMS['tile_size'],PARAMS['stride'])
    ys=get_starts(height,PARAMS['tile_size'],PARAMS['stride'])
    if grid_mode=='mirrored_y':
        ys=sorted({height-(s+min(PARAMS['tile_size'],height-s)) for s in ys})
    else: assert grid_mode=='original'
    return [(x,y,min(PARAMS['tile_size'],width-x),min(PARAMS['tile_size'],height-y)) for y in ys for x in xs]

def patch(image,meta,vertical_flip,pad=True):
    x,y,w,h=meta
    crop=image[y:y+h,x:x+w]
    if vertical_flip: crop=cv2.flip(crop,0)
    if pad and (w!=PARAMS['tile_size'] or h!=PARAMS['tile_size']):
        output=np.zeros((PARAMS['tile_size'],PARAMS['tile_size'],3),np.uint8)
        output[:h,:w]=crop;crop=output
    return np.ascontiguousarray(crop)

def inverse_local(boxes,valid_h):
    out=np.asarray(boxes,dtype=np.float32).copy()
    out[:,1]=valid_h-boxes[:,3];out[:,3]=valid_h-boxes[:,1]
    return out

def pack_rows(boxes,classes,scores,meta,vertical_flip):
    x,y,w,h=meta
    b=np.asarray(boxes,dtype=np.float32).copy()
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
    if vertical_flip:b=inverse_local(b,h)
    good=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);b=b[good]
    arr=np.zeros((len(b),14),np.float32)
    arr[:,0]=classes[good];arr[:,1]=scores[good]
    arr[:,6:10]=meta
    arr[:,2:6]=b+np.asarray([x,y,x,y],np.float32)
    # Store local coordinates consistent with the same float32 global rounding.
    arr[:,10:14]=arr[:,2:6]-arr[:,[6,7,6,7]]
    return arr

def audit(view):
    om,items=val_items();vid,grid,vflip=VIEWS[view]
    directory=ROOT/view/'cache';m=read_json(directory/'manifest.json')
    assert m['view_id']==vid and m['factor_orientation']==vflip and m['grid_mode']==grid
    assert m['weights']==str(WEIGHTS)
    for k,v in PARAMS.items():
        if k!='batch':assert m[k]==v,(k,m[k],v)
    assert m['image_count']==474 and len(m['items'])==474
    assert [x['image_name'] for x in m['items']]==[x['image_name'] for x in items]
    assert {p.name for p in directory.glob('*.npz')}=={x['cache_file'] for x in items}
    residual=0.;rows=0;nonzero=0;class_counts=np.zeros(9,np.int64);score_min=1.;score_max=0.
    for item,original in zip(m['items'],items):
        with np.load(directory/item['cache_file']) as z:
            a=z['candidates'];shape=z['image_shape']
        assert a.ndim==2 and a.shape[1]==14 and a.dtype==np.float32
        assert np.array_equal(shape,[original['height'],original['width']])
        assert len(a)==item['candidate_count'] and np.isfinite(a).all()
        origins=tile_origins(int(shape[0]),int(shape[1]),grid)
        assert item['tile_origins']==[list(t) for t in origins]
        if len(a):
            assert ((a[:,0]>=0)&(a[:,0]<9)&(a[:,0]==np.floor(a[:,0]))).all()
            assert ((a[:,1]>=PARAMS['conf'])&(a[:,1]<=1)).all()
            assert (a[:,4]>a[:,2]).all() and (a[:,5]>a[:,3]).all()
            assert (a[:,2:6]>=0).all() and (a[:,[2,4]]<=shape[1]).all() and (a[:,[3,5]]<=shape[0]).all()
            assert (a[:,10:14]>=0).all() and (a[:,[10,12]]<=a[:,8,None]).all() and (a[:,[11,13]]<=a[:,9,None]).all()
            assert set(map(tuple,np.unique(a[:,6:10],axis=0))).issubset(set(origins))
            r=float(np.max(np.abs(a[:,2:6].astype(np.float64)-a[:,10:14].astype(np.float64)-a[:,[6,7,6,7]].astype(np.float64))))
            residual=max(residual,r);nonzero+=int(np.count_nonzero(np.any(a[:,6:8]!=0,axis=1)))
            assert r<=1e-4,(item['image_name'],r)
            class_counts+=np.bincount(a[:,0].astype(int),minlength=9)
            score_min=min(score_min,float(a[:,1].min()));score_max=max(score_max,float(a[:,1].max()))
        rows+=len(a)
    assert rows==m['candidate_rows']
    result={'status':'PASS','view_id':vid,'image_count':474,'candidate_rows':rows,'coord_residual':residual,'nonzero_origin_rows':nonzero,'nan_inf_count':0,'class_counts':class_counts.tolist(),'score_min':score_min,'score_max':score_max,
        'checks':{'474_npz':'PASS','same_image_set_as_O':'PASS','schema_14_float32':'PASS','global_local_float64_residual':'PASS','finite_and_valid_boxes':'PASS','class_and_score_valid':'PASS','expected_tile_origins':'PASS'},'created_at':datetime.now(timezone.utc).isoformat()}
    write_json(ROOT/'audit'/f'cache_{view}.json',result)
    print('END cache_audit '+json.dumps(result),flush=True)
    return result

def build(view):
    assert read_json(ROOT/'exact_combo/baseline/PASS.json')['status']=='PASS'
    assert read_json(ROOT/'audit/geometry_tests.json')['status']=='PASS'
    order=list(VIEWS);pos=order.index(view)
    for earlier in order[:pos]:assert read_json(ROOT/'audit'/f'cache_{earlier}.json')['status']=='PASS'
    target=ROOT/view/'cache'
    if (target/'manifest.json').exists():
        print('REUSE complete cache '+view,flush=True);audit(view);return
    assert not target.exists() or not list(target.glob('*.npz')), 'Incomplete cache: STOP for investigation'
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available()
    om,items=val_items();vid,grid,vflip=VIEWS[view];target.mkdir(parents=True,exist_ok=True)
    start=time.time();torch.cuda.reset_peak_memory_stats();batch=PARAMS['batch'];oom=[]
    model=YOLO(str(WEIGHTS))
    manifest={'view_id':vid,'factor_orientation':vflip,'factor_grid':int(grid=='mirrored_y'),'grid_mode':grid,'weights':str(WEIGHTS),
        'weights_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        **PARAMS,'image_count':474,'images_count':474,'schema':DIAG.COLS,'columns':DIAG.COLS,'items':[], 'coordinate_space':'original_image',
        'local_coordinate_storage':'float32 global minus origin; audited in float64','image_source':str(IMAGES),'created_at':datetime.now(timezone.utc).isoformat(),
        'ground_truth_accessed':False,'test_hidden_run':False,'oom_events':oom}
    total=0
    print(f'START inference view={vid} images=474 grid={grid} vertical_flip={vflip}',flush=True)
    for index,item in enumerate(items,1):
        image=cv2.imread(str(IMAGES/item['image_name']),cv2.IMREAD_COLOR)
        assert image is not None
        h,w=image.shape[:2];assert (h,w)==(item['height'],item['width'])
        metas=tile_origins(h,w,grid);tiles=[patch(image,t,vflip) for t in metas]
        rows=[];begin=0
        while begin<len(tiles):
            count=min(batch,len(tiles)-begin)
            try:
                predictions=model.predict(tiles[begin:begin+count],imgsz=PARAMS['tile_size'],conf=PARAMS['conf'],iou=PARAMS['tile_iou'],max_det=PARAMS['max_det'],device='0',half=True,verbose=False)
            except torch.cuda.OutOfMemoryError:
                if batch==1:raise
                old=batch;batch=max(1,batch//2);oom.append({'image_index':index,'from_batch':old,'to_batch':batch})
                torch.cuda.empty_cache();print('OOM lowering batch '+json.dumps(oom[-1]),flush=True);continue
            for pred,meta in zip(predictions,metas[begin:begin+count]):
                if pred.boxes is None or not len(pred.boxes):continue
                boxes=pred.boxes.xyxy.detach().cpu().numpy();cls=pred.boxes.cls.detach().cpu().numpy();score=pred.boxes.conf.detach().cpu().numpy()
                rows.append(pack_rows(boxes,cls,score,meta,vflip))
            begin+=count
        arr=np.concatenate(rows) if rows else np.empty((0,14),np.float32)
        np.savez_compressed(target/item['cache_file'],candidates=arr,image_shape=np.asarray([h,w],np.int32))
        total+=len(arr);manifest['items'].append({'image_name':item['image_name'],'cache_file':item['cache_file'],'height':h,'width':w,'candidate_count':len(arr),'tile_origins':[list(t) for t in metas]})
        if index%10==0 or index==474:print(f'PROGRESS view={vid} {index}/474 candidates={total} runtime_seconds={time.time()-start:.2f} peak_gpu_mem_mb={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)
    torch.cuda.synchronize()
    manifest.update(candidate_rows=total,total_candidates=total,runtime_seconds=time.time()-start,runtime=time.time()-start,
        peak_gpu_mem_mb=torch.cuda.max_memory_allocated()/2**20,peak_gpu_reserved_mb=torch.cuda.max_memory_reserved()/2**20,batch=batch,half_actual=bool(model.predictor.model.fp16),completed_at=datetime.now(timezone.utc).isoformat())
    assert manifest['half_actual']
    write_json(target/'manifest.json',manifest);write_json(ROOT/view/'manifest.json',manifest)
    print(f'END inference view={vid} STATUS=PASS candidate_rows={total} runtime_seconds={manifest["runtime_seconds"]:.2f} peak_gpu_mem_mb={manifest["peak_gpu_mem_mb"]:.1f}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--view',choices=list(VIEWS),required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:audit(a.view)
    else:build(a.view)
